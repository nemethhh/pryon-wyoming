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
import onnxruntime as ort

from ._constants import FRAME_SAMPLES
from .decoder import START_CYCLE, Orchestra, cycle_score
from .fe import frame_count, lfbe_frames
from .modelset import KeywordSpec, ModelSet, load_model_set
from .runner import GraphRunner

ACCEPT = 2
NEAR_MISS = 0


@dataclass(frozen=True)
class PryonDetection:
    keyword: str
    detection_type: int       # ACCEPT (2) or NEAR_MISS (0)
    score: float
    begin_sample: int
    end_sample: int

    @property
    def type_name(self) -> str:
        return {ACCEPT: "Accept", NEAR_MISS: "NearMiss"}[self.detection_type]


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
    """Streaming keyword spotter over one Pryon model directory.

    ``thresholds`` overrides the kw.cfg accept threshold per keyword. A shared
    ``session`` (from another Spotter on the same model) skips reconverting the graph.
    """

    def __init__(self, model: str | os.PathLike | ModelSet, *,
                 thresholds: Mapping[str, float] | None = None,
                 session: ort.InferenceSession | None = None):
        model_set = model if isinstance(model, ModelSet) else \
            load_model_set(Path(model).expanduser())
        self._fe_cfg = model_set.fe
        self._specs = _with_thresholds(model_set.keywords, thresholds)
        self._runner = GraphRunner(model_set, session=session)
        self.reset()

    @property
    def session(self) -> ort.InferenceSession:
        return self._runner.session

    @property
    def keywords(self) -> list[str]:
        return [s.name for s in self._specs]

    @property
    def thresholds(self) -> dict[str, float]:
        """Effective accept threshold per keyword (before client-property overrides)."""
        return {s.name: s.accept for s in self._specs}

    def reset(self) -> None:
        """Clear all streaming state; sample indices restart at 0."""
        self._runner.reset()
        self._orchestra = Orchestra(self._specs)
        self._pending_pcm = np.zeros(0, np.int16)
        self._cycles_seen = 0
        self._history = {s.name: [] for s in self._specs}
        self.last_scores: dict[str, np.ndarray] = {s.name: np.zeros(0) for s in self._specs}

    def set_client_property(self, name: str, value: int) -> None:
        """Set a kw.cfg client property (e.g. AlarmState=1 selects its override threshold)."""
        self._runner.push_client_property(name, value)

    def feed(self, pcm: np.ndarray) -> list[PryonDetection]:
        """Feed int16 mono 16 kHz PCM of any length; returns detections completed by it.

        ``last_scores`` then holds the per-keyword classifier score of every engine
        cycle this call completed.
        """
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
            self.last_scores = {s.name: np.zeros(0) for s in self._specs}
            return []
        frames = lfbe_frames(buf, self._fe_cfg)
        # windows overlap, so the next frame starts at n*hop, not after the last window
        self._pending_pcm = buf[n * FRAME_SAMPLES:].copy()

        posteriors = self._runner.push_frames(frames)
        self._record_scores(posteriors)
        events = self._orchestra.push(posteriors, self._runner.client_properties)
        return [PryonDetection(e.keyword, e.detection_type, float(e.score),
                               max(0, int(e.begin_frame)) * FRAME_SAMPLES,
                               int(e.end_frame) * FRAME_SAMPLES)
                for e in events]

    def scores_tail(self, n: int = 1) -> dict[str, float]:
        """Per-keyword classifier score of the newest cycle; n > 1 averages the last n."""
        if n < 1:
            raise ValueError("n must be >= 1")
        return {name: float(np.mean(h[-n:])) if h else 0.0 for name, h in self._history.items()}

    def _record_scores(self, posteriors) -> None:
        rows = posteriors.lanes.shape[0]
        self.last_scores = {}
        for spec in self._specs:
            scores = np.array([
                cycle_score(posteriors.lanes[r], t, spec.class_index, spec.trim_tail, spec.window)
                if t >= START_CYCLE else 0.0
                for r, t in enumerate(range(self._cycles_seen, self._cycles_seen + rows))])
            self.last_scores[spec.name] = scores
            history = self._history[spec.name]
            history.extend(scores.tolist())
            del history[:-64]
        self._cycles_seen += rows
