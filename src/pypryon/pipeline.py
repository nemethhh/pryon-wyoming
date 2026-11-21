"""Streaming stages: one ONNX graph run per step, carrying its recurrent state.

A DAVS set runs a shared encoder every 8 LFBE frames and feeds each embedding to its
decoders, one step each. LSTM states are lane rings: before step t lane t % lanes of
h and c is zeroed (all lanes at t == 0) — the vendor LEDGER's ring contract, which the
DAVS decoders share with the single-graph model.
"""
from __future__ import annotations

import numpy as np
import onnxruntime as ort

from .davs import DavsModelSet, GraphSpec
from .graph import initial_state, make_portable


def create_stage_session(spec: GraphSpec) -> tuple[ort.InferenceSession, dict[str, np.ndarray]]:
    """Portable session for one graph plus the stored ("primed") values of its states."""
    raw = spec.path.read_bytes()
    session = ort.InferenceSession(make_portable(raw, keep_inputs=spec.state_inputs),
                                   providers=["CPUExecutionProvider"])
    return session, initial_state(raw, spec.state_inputs)


class StageRunner:
    """One graph: feeds state + inputs per step, keeps state outputs for the next step."""

    def __init__(self, spec: GraphSpec, session: ort.InferenceSession,
                 initial: dict[str, np.ndarray]):
        self.spec = spec
        self.session = session
        self.initial = initial
        shapes = {i.name: [d if isinstance(d, int) else 1 for d in i.shape]
                  for i in session.get_inputs()}
        self._lstm = spec.state_inputs[:spec.lstm_states]
        self._zero = {name: np.zeros(shapes[name], np.float32) for name in spec.state_inputs}
        self._outputs = [o.name for o in session.get_outputs()]
        self.reset()

    def reset(self) -> None:
        self._state = {name: np.array(self.initial.get(name, zero), np.float32)
                       for name, zero in self._zero.items()}
        self.step_index = 0

    def step(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        feeds = dict(self._state)
        t = self.step_index
        for name in self._lstm:
            if t == 0:
                feeds[name] = np.zeros_like(feeds[name])
            else:
                lanes = feeds[name].copy()
                lanes[t % lanes.shape[0]] = 0.0
                feeds[name] = lanes
        feeds.update(inputs)
        results = dict(zip(self._outputs, self.session.run(None, feeds), strict=True))
        for inp, out in zip(self.spec.state_inputs, self.spec.state_outputs, strict=True):
            self._state[inp] = results[out]
        self.step_index += 1
        return results


class DavsPipeline:
    """LFBE frames in; per-step decoder outputs keyed ``"<scorer>/<component>"`` out."""

    def __init__(self, model_set: DavsModelSet,
                 sessions: dict[str, tuple[ort.InferenceSession, dict]] | None = None):
        self.model_set = model_set
        self.sessions = sessions or create_sessions(model_set)
        self.stages = {name: StageRunner(spec, *self.sessions[name])
                       for name, spec in [(model_set.encoder.name, model_set.encoder),
                                          *model_set.decoders.items()]}
        enc = model_set.encoder
        self._frames_per_step = enc.frames_per_step
        self._feat = enc.feat_dim
        self._enc_input = next(i.name for i in enc.inputs if i.source == "audio")
        self._artifacts = {comp: tensor or comp for comp, tensor in enc.outputs.items()}
        self.reset()

    def reset(self) -> None:
        for stage in self.stages.values():
            stage.reset()
        self._pending = np.zeros((0, self._feat), np.float32)

    def push_frames(self, frames: np.ndarray) -> dict[str, np.ndarray]:
        frames = np.asarray(frames, np.float32)
        if frames.ndim != 2 or frames.shape[1] != self._feat:
            raise ValueError(f"expected (n, {self._feat}) frames, got {frames.shape}")
        buf = np.concatenate([self._pending, frames]) if self._pending.size else frames
        n = buf.shape[0] // self._frames_per_step
        self._pending = buf[n * self._frames_per_step:]
        rows: dict[str, list[np.ndarray]] = {}
        for k in range(n):
            chunk = buf[k * self._frames_per_step:(k + 1) * self._frames_per_step]
            for key, value in self._step(chunk).items():
                rows.setdefault(key, []).append(value)
        if not rows:
            return self._empty()
        return {key: np.stack(values) for key, values in rows.items()}

    def _step(self, chunk: np.ndarray) -> dict[str, np.ndarray]:
        enc = self.stages[self.model_set.encoder.name]
        enc_out = enc.step({self._enc_input: chunk[None]})
        artifacts = {self._artifacts[comp]: enc_out[comp] for comp in self._artifacts}
        out = {}
        for name, spec in self.model_set.decoders.items():
            feeds = {i.name: artifacts[i.tensor] for i in spec.inputs}
            results = self.stages[name].step(feeds)
            for comp in spec.outputs:
                out[f"{name}/{comp}"] = results[comp][0]   # drop the batch axis
        return out

    def _empty(self) -> dict[str, np.ndarray]:
        out = {}
        for name, spec in self.model_set.decoders.items():
            sess = self.sessions[name][0]
            shapes = {o.name: o.shape for o in sess.get_outputs()}
            for comp in spec.outputs:
                tail = [d if isinstance(d, int) else 1 for d in shapes[comp][1:]]
                out[f"{name}/{comp}"] = np.zeros([0, *tail], np.float32)
        return out


def create_sessions(model_set: DavsModelSet) -> dict[str, tuple[ort.InferenceSession, dict]]:
    """Sessions for every graph of a DAVS set; shareable across Spotters."""
    return {spec.name: create_stage_session(spec)
            for spec in [model_set.encoder, *model_set.decoders.values()]}
