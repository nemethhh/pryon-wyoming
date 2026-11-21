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


def session_options() -> ort.SessionOptions:
    """One thread per session: the graphs are tiny, and idle ORT pools spin whole cores."""
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    return opts


def create_session(model_set: ModelSet) -> ort.InferenceSession:
    """Convert the model's graph and open it; safe to share across GraphRunners."""
    return ort.InferenceSession(make_portable(model_set.graph_path.read_bytes()),
                                sess_options=session_options(), providers=["CPUExecutionProvider"])


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
    """Stateful streaming driver for a single-graph model: LFBE frames in, cycles out."""

    def __init__(self, model_set: ModelSet, session: ort.InferenceSession | None = None):
        from .davs import GraphInput, GraphSpec
        from .pipeline import StageRunner

        self.session = session or create_session(model_set)
        shapes = {i.name: i.shape for i in self.session.get_inputs()}
        out_names = {o.name for o in self.session.get_outputs()}
        for out in model_set.state_outputs:
            if out not in out_names:
                raise ValueError(f"state output {out!r} missing from graph")
        # 2-D states are the LSTM h/c rings; the loader lists them before the conv buffers
        lstm = sum(len(shapes[name]) == 2 for name in model_set.state_inputs)
        spec = GraphSpec("model", model_set.graph_path, [GraphInput(model_set.input_name, "audio", None)],
                         {SOFTMAX_WW: None, FC_SE: None}, list(model_set.state_inputs),
                         list(model_set.state_outputs), lstm, CYCLE_FRAMES, model_set.fe.num_mel)
        self._stage = StageRunner(spec, self.session, {})
        self._input_name = model_set.input_name
        self._num_classes = len(model_set.words)
        self._feats = model_set.fe.num_mel
        self.client_properties = dict(model_set.client_prop_defaults)
        self.reset()

    def reset(self) -> None:
        """Zero LSTM/conv state and the cycle counter."""
        self._stage.reset()
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
            cycle = buf[c * CYCLE_FRAMES:(c + 1) * CYCLE_FRAMES]
            out = self._stage.step({self._input_name: cycle[None, :, :, None]})
            lanes.append(out[SOFTMAX_WW][:, 0, :])
            ses.append(out[FC_SE][:, 0, :])
        lanes_v, se_v = np.stack(lanes), np.stack(ses)
        return Posteriors(ww=lanes_v.max(axis=1), se=se_v.max(axis=1),
                          lanes=lanes_v, se_lanes=se_v)
