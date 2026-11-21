"""Streaming runner for the ported Pryon graph.

Drives the surgered/portable graph one engine cycle at a time: 6 LFBE frames
per session.run — the cycle is the engine's quantum (kw.cfg.json
scorer.score_upsampling: initialize.frame_count 6, skip-frame-count 5) and
the unit the vendor LSTM ring contract speaks in.  Multi-cycle calls would
skip the host-side lane reset between steps, so the runner never makes one.

Lane-ring contract (vendor pryon_lstm_analysis.py / pryon_quant_model
Model.stream): 25 LSTM lanes are ≤1.5 s sliding windows; before cycle t the
host zeroes lane t % 25 of h and c (all lanes at t == 0).  The full ring is
what the kw.cfg decoder consumes (vendor pryon_decoder.cycle_score scores by
lane age), so Posteriors carries it alongside the collapsed per-cycle views.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import onnxruntime as ort

from .graph import make_portable
from .modelset import ModelSet

# One engine cycle = this many LFBE frames (scorer.score_upsampling).
CYCLE_FRAMES = 6


def _concrete(shape) -> list[int]:
    """Replace dynamic dim_params (strings) with 1 — buffer batch dims."""
    return [d if isinstance(d, int) else 1 for d in shape]

# Graph tensor names of the two heads (nemort_scorer_config.json output_map;
# constant across this model generation — spec §"What is already known").
SOFTMAX_WW = "softmax_output_ww"
FC_SE = "fc_output_se"

# Portable-model cache: conversion is pure in the source bytes.
_PORT_CACHE: dict[bytes, bytes] = {}


def portable_model(graph_bytes: bytes) -> bytes:
    """make_portable with a process-local cache (Task 12 persists its own)."""
    out = _PORT_CACHE.get(graph_bytes)
    if out is None:
        out = make_portable(graph_bytes)
        _PORT_CACHE[graph_bytes] = out
    return out


@dataclass
class Posteriors:
    """One push's worth of engine-cycle outputs.

    Rows are engine cycles (60 ms).  ``lanes`` is the full softmax ring
    [n, 25, K+1] (classes ordered like words.shrunk.txt: background first);
    ``ww``/``se`` are the per-cycle per-class max over lanes.
    """

    ww: np.ndarray     # (n, K+1) float32
    se: np.ndarray     # (n, 2) float32
    lanes: np.ndarray  # (n, 25, K+1) float32
    se_lanes: np.ndarray = None  # (n, 25, 2) float32 — full se ring


class GraphRunner:
    """Stateful streaming driver: PCM-level FE output → per-cycle posteriors."""

    def __init__(self, model_set: ModelSet):
        self._model_set = model_set
        self._session = ort.InferenceSession(
            portable_model(model_set.graph_path.read_bytes()),
            providers=["CPUExecutionProvider"],
        )
        self._input_name = model_set.input_name
        out_names = [o.name for o in self._session.get_outputs()]
        input_shapes = {i.name: _concrete(i.shape)
                        for i in self._session.get_inputs()}
        self._input_shapes = input_shapes

        # Positional state pairing from the scorer config (lstm_components
        # + recurrent components); 2-D states are LSTM h/c rings (ring-zeroed
        # per cycle), the 4-D ones are streaming conv buffers (pass-through).
        self._lstm: list[tuple[str, str]] = []
        self._buffers: list[tuple[str, str]] = []
        for inp, out in zip(model_set.state_inputs, model_set.state_outputs):
            if out not in out_names:
                raise ValueError(f"state output {out!r} missing from graph")
            shape = input_shapes[inp]
            (self._lstm if len(shape) == 2 else self._buffers).append((inp, out))
        self._out_pos = {name: pos for pos, name in enumerate(out_names)}
        self._num_classes = len(model_set.words)  # incl. background <eps>
        self._feats = model_set.fe.num_mel
        self.reset()
        # Client properties (kw.cfg override knobs); start at the derived
        # defaults (0/unset matches no override — modelset.py provenance).
        self.client_properties = dict(model_set.client_prop_defaults)

    # -- state ------------------------------------------------------------

    def reset(self) -> None:
        """Zero LSTM/conv state and the cycle counter (no stale lanes)."""
        self._states = {
            inp: np.zeros(self._input_shapes[inp], np.float32)
            for inp, _ in self._lstm + self._buffers
        }
        self._cycle = 0
        self._pending = np.zeros((0, self._feats), np.float32)

    def push_client_property(self, name: str, value: int) -> None:
        """Store a client property; the decoder reads it for thresholds."""
        self.client_properties[name] = value

    # -- streaming ----------------------------------------------------------

    def push_frames(self, frames: np.ndarray) -> Posteriors:
        """Feed (n, num_mel) float32 LFBE frames; return this push's cycles.

        Arbitrary chunk sizes: partial cycles are buffered until 6 frames
        accumulate, so the posterior stream is identical to a one-call feed
        of the same frames.
        """
        frames = np.asarray(frames)
        if frames.ndim != 2 or frames.shape[1] != self._feats:
            raise ValueError(
                f"expected (n, {self._feats}) frames, got {frames.shape}")
        if frames.dtype != np.float32:
            frames = frames.astype(np.float32)
        buf = np.concatenate([self._pending, frames]) if self._pending.size else frames
        n_cycles = buf.shape[0] // CYCLE_FRAMES
        self._pending = buf[n_cycles * CYCLE_FRAMES:]
        if n_cycles == 0:
            k = self._num_classes
            return Posteriors(np.zeros((0, k), np.float32),
                              np.zeros((0, 2), np.float32),
                              np.zeros((0, 25, k), np.float32),
                              np.zeros((0, 25, 2), np.float32))

        lanes, ses = [], []
        for c in range(n_cycles):
            self._run_cycle(buf[c * CYCLE_FRAMES:(c + 1) * CYCLE_FRAMES], lanes, ses)
        lanes_v = np.stack(lanes)
        se_v = np.stack(ses)
        return Posteriors(
            ww=lanes_v.max(axis=1),
            se=se_v.max(axis=1),
            lanes=lanes_v,
            se_lanes=se_v,
        )

    # -- internals -----------------------------------------------------------

    def _run_cycle(self, cycle_frames: np.ndarray, lanes, ses) -> None:
        feeds = dict(self._states)
        feeds[self._input_name] = cycle_frames[None, :, :, None]
        # Host-side lane-ring zeroing (Model.stream):
        #   t == 0 → all lanes; else lane t % 25 (h and c both).
        for inp, _ in self._lstm:
            t = feeds[inp]
            if self._cycle == 0:
                feeds[inp] = np.zeros_like(t)
            else:
                t = t.copy()
                t[self._cycle % 25] = 0.0
                feeds[inp] = t
        results = self._session.run(None, feeds)
        for inp, out in self._lstm + self._buffers:
            self._states[inp] = results[self._out_pos[out]]
        lanes.append(results[self._out_pos[SOFTMAX_WW]][:, 0, :])  # [25, K+1]
        ses.append(results[self._out_pos[FC_SE]][:, 0, :])         # [25, 2]
        self._cycle += 1
