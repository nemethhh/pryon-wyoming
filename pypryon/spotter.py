"""End-to-end spotter: int16 PCM → PryonDetection events.

Composes the LFBE front end (fe.py), the streaming graph runner
(runner.py) and the kw.cfg orchestra (decoder.py).  PCM is buffered across
feeds so arbitrary chunk sizes give one continuous, chunk-independent
detection stream; sample indices are cumulative from the first feed after
construction/reset (the caller's sample-index space) at 160 samples per
10 ms LFBE frame.

Input contract: int16, mono, 16 kHz, 1-D (one feed = one contiguous chunk).
A sample rate cannot be detected from raw PCM — anything but 16 kHz audio
shifts every index and score; the contract is stated here and in the README
rather than guessed at runtime.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .decoder import Orchestra, cycle_score
from .fe import frame_count, lfbe_frames
from .modelset import KeywordSpec, load_model_set
from .runner import GraphRunner

_FRAME_SAMPLES = 160          # 10 ms hop


@dataclass(frozen=True)
class PryonDetection:
    keyword: str
    detection_type: int       # 2 = Accept, 0 = NearMiss
    score: float
    begin_sample: int
    end_sample: int

    @property
    def type_name(self) -> str:
        return {2: "Accept", 0: "NearMiss"}[self.detection_type]


def _with_thresholds(specs: list[KeywordSpec],
                     thresholds: dict[str, float] | None) -> list[KeywordSpec]:
    if not thresholds:
        return specs
    return [replace(s, accept=float(thresholds[s.name])) if s.name in thresholds
            else s for s in specs]


class Spotter:
    """Streaming wake-word spotter over one Pryon model directory."""

    def __init__(self, model_dir: Path, *, thresholds: dict[str, float] | None = None):
        model_set = load_model_set(Path(model_dir))
        self._fe_cfg = model_set.fe
        self._specs = _with_thresholds(model_set.keywords, thresholds)
        self._runner = GraphRunner(model_set)
        self._orchestra = Orchestra(self._specs)
        self._scores: dict[str, deque[float]] = {k.name: deque(maxlen=64)
                                                 for k in self._specs}
        self._cycles_seen = 0
        self.reset()

    # -- lifecycle -----------------------------------------------------------

    def reset(self) -> None:
        """Clear all pipeline state (FE buffer, LSTM lanes, decoder)."""
        self._runner.reset()
        self._orchestra = Orchestra(self._specs)
        self._pending_pcm = np.zeros(0, np.int16)
        self._cycles_seen = 0
        for d in self._scores.values():
            d.clear()

    def set_client_property(self, name: str, value: int) -> None:
        """Set a kw.cfg client property (e.g. AlarmState=1 lowers ALEXA's
        accept threshold to its override value)."""
        self._runner.push_client_property(name, value)

    # -- streaming -------------------------------------------------------------

    def feed(self, pcm: np.ndarray) -> list[PryonDetection]:
        """Feed int16 mono 16 kHz PCM; returns detections from this chunk."""
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
            return []
        frames = lfbe_frames(buf, self._fe_cfg)
        # The next feed's frames must start on the 160-sample grid: frame n
        # begins at n*160 (windows overlap the previous frame's tail by 240
        # samples), so the pending tail is everything from n*160 on — NOT
        # from the last frame's end, which is off-grid and would phase-shift
        # the whole frame stream.
        self._pending_pcm = buf[n * 160:].copy()

        posteriors = self._runner.push_frames(frames)
        self._record_scores(posteriors)
        events = self._orchestra.push(posteriors, self._runner.client_properties)
        return [PryonDetection(e.keyword, e.detection_type, e.score,
                               e.begin_frame * _FRAME_SAMPLES,
                               e.end_frame * _FRAME_SAMPLES)
                for e in events]

    def scores_tail(self, n: int = 1) -> dict[str, float]:
        """Per-keyword smoothed classifier score; n>1 averages the last n."""
        if n < 1:
            raise ValueError("n must be >= 1")
        out = {}
        for name, d in self._scores.items():
            tail = list(d)[-n:]
            out[name] = float(np.mean(tail)) if tail else 0.0
        return out

    # -- internals -----------------------------------------------------------

    def _record_scores(self, posteriors) -> None:
        rows = posteriors.lanes.shape[0]
        if rows == 0:
            return
        t = self._cycles_seen + rows - 1      # global cycle of the newest row
        for spec in self._specs:
            self._scores[spec.name].append(cycle_score(
                posteriors.lanes[-1], t, spec.class_index, spec.trim_tail,
                spec.window))
        self._cycles_seen += rows
