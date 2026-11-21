"""End-to-end spotter: int16 PCM in, keyword detections out.

Input contract: int16, mono, 16 kHz, 1-D. A sample rate cannot be detected
from raw PCM, so anything else silently shifts every index and score.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from ._constants import FRAME_SAMPLES
from .davs import DavsModelSet
from .decoder import START_CYCLE, Orchestra, cycle_score
from .fe import frame_count, lfbe_frames
from .modelset import KeywordSpec, ModelSet, load_model_set
from .runner import GraphRunner

ACCEPT = 2
NEAR_MISS = 0
ACCEPT_AUXILIARY = 3   # DAVS verifier accepts (libpryon reports them as RESULT type 2)


@dataclass(frozen=True)
class PryonDetection:
    keyword: str
    detection_type: int       # ACCEPT (2) or NEAR_MISS (0)
    score: float
    begin_sample: int
    end_sample: int

    @property
    def type_name(self) -> str:
        return {ACCEPT: "Accept", NEAR_MISS: "NearMiss",
                ACCEPT_AUXILIARY: "AcceptAuxiliary"}[self.detection_type]

    @property
    def is_accept(self) -> bool:
        return self.detection_type in (ACCEPT, ACCEPT_AUXILIARY)


def _with_thresholds(specs: list[KeywordSpec],
                     thresholds: Mapping[str, float] | None) -> list[KeywordSpec]:
    if not thresholds:
        return specs
    names = [s.name for s in specs]
    for name in thresholds:
        if name not in names:
            raise ValueError(f"unknown keyword {name!r}; model has: {', '.join(names)}")
    return [replace(s, accept=float(thresholds[s.name])) if s.name in thresholds else s
            for s in specs]


class Spotter:
    """Streaming keyword spotter over one Pryon model directory (single-graph or DAVS).

    ``thresholds`` overrides the kw.cfg accept threshold per keyword (a DAVS verifier
    moves with its keyword). A shared ``session`` (``Spotter.session`` of another
    Spotter on the same model) skips reconverting the graphs.
    """

    def __init__(self, model: str | os.PathLike | ModelSet | DavsModelSet, *,
                 thresholds: Mapping[str, float] | None = None, session=None):
        model_set = model if isinstance(model, (ModelSet, DavsModelSet)) else \
            load_model_set(Path(model).expanduser())
        self._fe_cfg = model_set.fe
        if isinstance(model_set, DavsModelSet):
            self._backend = _DavsBackend(model_set, thresholds, session)
        else:
            self._backend = _GraphBackend(model_set, thresholds, session)
        self.reset()

    @property
    def session(self):
        return self._backend.session

    @property
    def keywords(self) -> list[str]:
        return self._backend.keywords

    @property
    def thresholds(self) -> dict[str, float]:
        """Effective accept threshold per keyword (before client-property overrides)."""
        return self._backend.thresholds

    @property
    def last_scores(self) -> dict[str, np.ndarray]:
        """Per-keyword classifier score of every engine step the last feed() completed."""
        return self._backend.last_scores

    def reset(self) -> None:
        """Clear all streaming state; sample indices restart at 0."""
        self._backend.reset()
        self._pending_pcm = np.zeros(0, np.int16)
        self._history = {k: [] for k in self.keywords}

    def set_client_property(self, name: str, value: int) -> None:
        """Set a kw.cfg client property (e.g. AlarmState=1 selects its override threshold)."""
        self._backend.client_properties[name] = value

    def feed(self, pcm: np.ndarray) -> list[PryonDetection]:
        """Feed int16 mono 16 kHz PCM of any length; returns detections completed by it."""
        pcm = np.asarray(pcm)
        if pcm.dtype != np.int16:
            raise ValueError(f"expected int16 PCM, got dtype {pcm.dtype}")
        if pcm.ndim != 1:
            raise ValueError(f"expected 1-D mono PCM, got shape {pcm.shape}")
        if pcm.size == 0:
            raise ValueError("empty PCM chunk")
        buf = np.concatenate([self._pending_pcm, pcm])
        n = frame_count(buf.size, self._fe_cfg)
        if n == 0:
            self._pending_pcm = buf
            self._backend.clear_scores()
            return []
        frames = lfbe_frames(buf, self._fe_cfg)
        # windows overlap, so the next frame starts at n*hop, not after the last window
        self._pending_pcm = buf[n * FRAME_SAMPLES:].copy()
        events = self._backend.push(frames)
        for name, scores in self._backend.last_scores.items():
            history = self._history[name]
            history.extend(scores.tolist())
            del history[:-64]
        return events

    def scores_tail(self, n: int = 1) -> dict[str, float]:
        """Per-keyword classifier score of the newest step; n > 1 averages the last n."""
        if n < 1:
            raise ValueError("n must be >= 1")
        return {name: float(np.mean(h[-n:])) if h else 0.0 for name, h in self._history.items()}


class _GraphBackend:
    """The single-graph generation (firmware ALEXA): 6-frame cycles, 25-lane ring."""

    def __init__(self, model_set: ModelSet, thresholds, session):
        self._specs = _with_thresholds(model_set.keywords, thresholds)
        self._runner = GraphRunner(model_set, session=session)
        self.client_properties = self._runner.client_properties
        self.keywords = [s.name for s in self._specs]
        self.thresholds = {s.name: s.accept for s in self._specs}

    @property
    def session(self):
        return self._runner.session

    def reset(self) -> None:
        self._runner.reset()
        self._orchestra = Orchestra(self._specs)
        self._cycles_seen = 0
        self.clear_scores()

    def clear_scores(self) -> None:
        self.last_scores = {s.name: np.zeros(0) for s in self._specs}

    def push(self, frames: np.ndarray) -> list[PryonDetection]:
        posteriors = self._runner.push_frames(frames)
        rows = posteriors.lanes.shape[0]
        self.last_scores = {}
        for spec in self._specs:
            self.last_scores[spec.name] = np.array([
                cycle_score(posteriors.lanes[r], t, spec.class_index, spec.trim_tail, spec.window)
                if t >= START_CYCLE else 0.0
                for r, t in enumerate(range(self._cycles_seen, self._cycles_seen + rows))])
        self._cycles_seen += rows
        events = self._orchestra.push(posteriors, self.client_properties)
        return [PryonDetection(e.keyword, e.detection_type, float(e.score),
                               max(0, int(e.begin_frame)) * FRAME_SAMPLES,
                               int(e.end_frame) * FRAME_SAMPLES)
                for e in events]


class _DavsBackend:
    """A DAVS set: shared encoder every 8 frames, decoders per step, DAVS rules."""

    def __init__(self, model_set: DavsModelSet, thresholds, session):
        from .pipeline import DavsPipeline
        from .rules import DavsOrchestra

        spotters = list(model_set.spotters)
        probe = DavsOrchestra(spotters, frames_per_step=model_set.encoder.frames_per_step)
        if thresholds:
            for name in thresholds:
                if name not in probe.keywords:
                    raise ValueError(f"unknown keyword {name!r}; model has: {', '.join(probe.keywords)}")
            base = {s.name: s.accept for s in spotters if not s.auxiliary}
            spotters = [replace(s, accept=float(thresholds[s.name]) + (s.accept - base[s.name] if s.auxiliary else 0.0))
                        if s.name in thresholds else s for s in spotters]
        self._pipeline = DavsPipeline(model_set, sessions=session)
        self._orchestra = DavsOrchestra(spotters, frames_per_step=model_set.encoder.frames_per_step)
        self.keywords = self._orchestra.keywords
        self.thresholds = {s.name: s.accept for s in spotters if not s.auxiliary and s.name in self.keywords}
        self.client_properties = dict(model_set.client_prop_defaults)

    @property
    def session(self):
        return self._pipeline.sessions

    def reset(self) -> None:
        self._pipeline.reset()
        self._orchestra.reset()
        self.clear_scores()

    def clear_scores(self) -> None:
        self.last_scores = {k: np.zeros(0) for k in self.keywords}

    def push(self, frames: np.ndarray) -> list[PryonDetection]:
        rows = self._pipeline.push_frames(frames)
        events = self._orchestra.push(rows, self.client_properties)
        self.last_scores = self._orchestra.last_scores
        return [PryonDetection(e.keyword, e.detection_type, float(e.score), int(e.begin_sample),
                               int(e.end_sample)) for e in events]
