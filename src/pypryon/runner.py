"""Streaming driver for the portable Pryon graph.

The graph runs one engine cycle (6 LFBE frames) per ``session.run``. Its 25
LSTM lanes are <=1.5 s sliding windows: before cycle t the host zeroes lane
t % 25 of h and c (all lanes at t == 0), per the vendor LEDGER's lane-ring
contract. The full ring is returned because the decoder scores by lane age.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import onnxruntime as ort

from ._constants import CYCLE_FRAMES, LANES
from .graph import make_portable
from .modelset import ModelSet

# nemort_scorer_config.json output_map; fixed across this model generation
SOFTMAX_WW = "softmax_output_ww"
FC_SE = "fc_output_se"


def _concrete(shape) -> list[int]:
    return [d if isinstance(d, int) else 1 for d in shape]


def create_session(model_set: ModelSet) -> ort.InferenceSession:
    """Convert the model's graph and open it; safe to share across GraphRunners."""
    return ort.InferenceSession(make_portable(model_set.graph_path.read_bytes()),
                                providers=["CPUExecutionProvider"])


@dataclass
class Posteriors:
    """Engine-cycle outputs of one push; rows are 60 ms cycles.

    ``lanes`` is the softmax ring [n, 25, K+1] (classes ordered like
    words.shrunk.txt, background first); ``se_lanes`` the se ring [n, 25, 2];
    ``ww``/``se`` their per-cycle max over lanes.
    """

    ww: np.ndarray
    se: np.ndarray
    lanes: np.ndarray
    se_lanes: np.ndarray


class GraphRunner:
    """Stateful streaming driver: LFBE frames in, per-cycle posteriors out."""

    def __init__(self, model_set: ModelSet, session: ort.InferenceSession | None = None):
        self.session = session or create_session(model_set)
        self._input_name = model_set.input_name
        out_names = [o.name for o in self.session.get_outputs()]
        self._input_shapes = {i.name: _concrete(i.shape) for i in self.session.get_inputs()}

        # 2-D states are LSTM h/c rings (lane-zeroed per cycle); the rest are
        # streaming conv buffers, passed through unchanged
        self._lstm: list[tuple[str, str]] = []
        self._buffers: list[tuple[str, str]] = []
        for inp, out in zip(model_set.state_inputs, model_set.state_outputs):
            if out not in out_names:
                raise ValueError(f"state output {out!r} missing from graph")
            target = self._lstm if len(self._input_shapes[inp]) == 2 else self._buffers
            target.append((inp, out))
        self._out_pos = {name: pos for pos, name in enumerate(out_names)}
        self._num_classes = len(model_set.words)
        self._feats = model_set.fe.num_mel
        self.client_properties = dict(model_set.client_prop_defaults)
        self.reset()

    def reset(self) -> None:
        """Zero LSTM/conv state and the cycle counter."""
        self._states = {inp: np.zeros(self._input_shapes[inp], np.float32)
                        for inp, _ in self._lstm + self._buffers}
        self._cycle = 0
        self._pending = np.zeros((0, self._feats), np.float32)

    def push_client_property(self, name: str, value: int) -> None:
        self.client_properties[name] = value

    def push_frames(self, frames: np.ndarray) -> Posteriors:
        """Feed (n, num_mel) LFBE frames; partial cycles wait for the next push."""
        frames = np.asarray(frames)
        if frames.ndim != 2 or frames.shape[1] != self._feats:
            raise ValueError(f"expected (n, {self._feats}) frames, got {frames.shape}")
        frames = frames.astype(np.float32, copy=False)
        buf = np.concatenate([self._pending, frames]) if self._pending.size else frames
        n_cycles = buf.shape[0] // CYCLE_FRAMES
        self._pending = buf[n_cycles * CYCLE_FRAMES:]
        if n_cycles == 0:
            k = self._num_classes
            return Posteriors(np.zeros((0, k), np.float32), np.zeros((0, 2), np.float32),
                              np.zeros((0, LANES, k), np.float32),
                              np.zeros((0, LANES, 2), np.float32))
        lanes, ses = [], []
        for c in range(n_cycles):
            ww, se = self._run_cycle(buf[c * CYCLE_FRAMES:(c + 1) * CYCLE_FRAMES])
            lanes.append(ww)
            ses.append(se)
        lanes_v, se_v = np.stack(lanes), np.stack(ses)
        return Posteriors(ww=lanes_v.max(axis=1), se=se_v.max(axis=1),
                          lanes=lanes_v, se_lanes=se_v)

    def _run_cycle(self, cycle_frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        feeds = dict(self._states)
        feeds[self._input_name] = cycle_frames[None, :, :, None]
        for inp, _ in self._lstm:
            if self._cycle == 0:
                feeds[inp] = np.zeros_like(feeds[inp])
            else:
                state = feeds[inp].copy()
                state[self._cycle % LANES] = 0.0
                feeds[inp] = state
        results = self.session.run(None, feeds)
        for inp, out in self._lstm + self._buffers:
            self._states[inp] = results[self._out_pos[out]]
        self._cycle += 1
        return (results[self._out_pos[SOFTMAX_WW]][:, 0, :],
                results[self._out_pos[FC_SE]][:, 0, :])
