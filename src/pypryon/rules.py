"""DAVS keyword rules: per-step decoder scores to Accept / AcceptAuxiliary / NearMiss.

Fitted against libpryon (qemu oracle) on alexa-en-US:

* ``moving-average-max-pool`` window 1 scores a step as the max over lanes; the
  verifier (``DETECTION_AUXILIARY``) spotter is an alternative accept path that is only
  checked when the primary does not accept, and an accept of either locks both;
* accepts fire at the first step at or above the active threshold, ending at the
  step's last sample, and lock the keyword for its lock period (frames);
* a near miss is the peak over the 8 steps from the first step at or above notify,
  emitted at (peak step end + lock period) unless an accept comes first;
* ``exponential-decay`` smooths the per-step score as s = rate*s + (1-rate)*x.

``begin_sample`` is approximate (the start of the winning lane's window); libpryon
derives it at frame resolution from the alignments head.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ._constants import FRAME_SAMPLES
from .davs import SpotterSpec

_LOGGER = logging.getLogger(__name__)

ACCEPT = 2
NEAR_MISS = 0
ACCEPT_AUXILIARY = 3
NEAR_MISS_WINDOW_STEPS = 8
_WARNED: set[tuple] = set()


@dataclass(frozen=True)
class DavsDetection:
    keyword: str
    detection_type: int
    score: float
    step: int
    begin_sample: int
    end_sample: int


class MaxPool:
    def __init__(self, window: int = 1):
        self.window = window

    def score(self, row: np.ndarray, class_index: int, t: int = 0) -> float:
        col = row[:, class_index].astype(np.float64)
        if self.window <= 1:
            return float(col.max())
        lanes = col.size
        by_age = col[(t - np.arange(lanes)) % lanes]      # age 0 first
        n = by_age.size
        return float(max(by_age[i:i + self.window].mean() for i in range(n - self.window + 1)))


class ExponentialDecay:
    def __init__(self, rate: float):
        self.rate = rate
        self.reset()

    def reset(self) -> None:
        self.state = 0.0

    def update(self, x: float) -> float:
        self.state = self.rate * self.state + (1.0 - self.rate) * float(x)
        return self.state


def _classifier(spec: SpotterSpec):
    kind = spec.classifier.get("type")
    if len(spec.components) != 1:
        return None
    if kind == "moving-average-max-pool":
        return MaxPool(int(spec.classifier.get("window-size", 1)))
    if kind == "exponential-decay":
        return ExponentialDecay(float(spec.classifier.get("decay-rate", 0.0)))
    return None


class _Keyword:
    def __init__(self, primary: SpotterSpec, auxiliary: SpotterSpec | None):
        self.primary, self.auxiliary = primary, auxiliary
        self.cls = _classifier(primary)
        self.aux_cls = _classifier(auxiliary) if auxiliary else None
        self.reset()

    def reset(self) -> None:
        self.lock_until = -1
        self.run: list | None = None      # [start step, peak score, peak step, begin]
        self.pending: tuple | None = None  # (score, peak step, begin, end sample)
        for c in (self.cls, self.aux_cls):
            if isinstance(c, ExponentialDecay):
                c.reset()


class DavsOrchestra:
    """Streaming decision layer over DavsPipeline outputs (keys ``"<scorer>/<component>"``)."""

    def __init__(self, spotters: list[SpotterSpec], *, frames_per_step: int = 8,
                 words: dict[int, str] | None = None):
        self.step_samples = frames_per_step * FRAME_SAMPLES
        primaries: dict[str, SpotterSpec] = {}
        aux: dict[str, SpotterSpec] = {}
        for spec in spotters:
            (aux if spec.auxiliary else primaries).setdefault(spec.name, spec)
        self._keywords: list[_Keyword] = []
        for name, spec in primaries.items():
            kw = _Keyword(spec, aux.get(name))
            if kw.cls is None or (kw.auxiliary is not None and kw.aux_cls is None):
                key = (name, spec.classifier.get("type"))
                if key not in _WARNED:  # every connection builds an orchestra
                    _WARNED.add(key)
                    _LOGGER.warning("Keyword %s uses an unsupported classifier (%s); not served", *key)
                continue
            self._keywords.append(kw)
        if words:  # the model's word order puts the wake word first (ALEXA 1 before STOP)
            order = {w: i for i, w in words.items()}
            self._keywords.sort(key=lambda kw: order.get(kw.primary.name, len(order)))
        self._step = 0
        self.last_scores: dict[str, np.ndarray] = {kw.primary.name: np.zeros(0) for kw in self._keywords}

    @property
    def keywords(self) -> list[str]:
        return [kw.primary.name for kw in self._keywords]

    def reset(self) -> None:
        for kw in self._keywords:
            kw.reset()
        self._step = 0

    @staticmethod
    def _threshold(spec: SpotterSpec, props: dict[str, int]) -> float:
        for prop, equals, accept in spec.overrides:
            if props.get(prop, 0) == equals:
                return accept
        return spec.accept

    def _score(self, cls, spec: SpotterSpec, rows: dict, r: int, t: int) -> tuple[float, int]:
        comp = spec.components[0]
        row = rows[f"{spec.scorer}/{comp.component}"][r]
        if isinstance(cls, ExponentialDecay):
            return cls.update(float(row[:, comp.class_index].max())), t
        lane = int(np.argmax(row[:, comp.class_index]))
        age = (t - lane) % row.shape[0]
        return cls.score(row, comp.class_index, t), t - age

    def push(self, rows: dict[str, np.ndarray], client_props: dict[str, int] | None = None
             ) -> list[DavsDetection]:
        props = client_props or {}
        n = next(iter(rows.values())).shape[0] if rows else 0
        events: list[DavsDetection] = []
        self._scores: dict[str, list[float]] = {kw.primary.name: [] for kw in self._keywords}
        for r in range(n):
            t = self._step + r
            for kw in self._keywords:
                self._step_keyword(kw, rows, r, t, props, events)
        self._step += n
        self.last_scores = {k: np.array(v) for k, v in self._scores.items()}
        return events

    def _step_keyword(self, kw: _Keyword, rows, r, t, props, events) -> None:
        p, a = kw.primary, kw.auxiliary
        end = (t + 1) * self.step_samples
        score, win_start = self._score(kw.cls, p, rows, r, t)
        self._scores[p.name].append(score)
        aux_score = self._score(kw.aux_cls, a, rows, r, t) if a else None
        if kw.pending and end >= kw.pending[3]:
            s, peak_t, begin, nm_end = kw.pending
            events.append(DavsDetection(p.name, NEAR_MISS, s, peak_t, begin, nm_end))
            kw.pending = None
        if end > kw.lock_until:
            fired = None
            if score >= self._threshold(p, props):
                fired = (ACCEPT, score, win_start)
            elif aux_score is not None and aux_score[0] >= self._threshold(a, props):
                fired = (ACCEPT_AUXILIARY, aux_score[0], aux_score[1])
            if fired:
                kind, s, start = fired
                events.append(DavsDetection(p.name, kind, s, t, max(0, start * self.step_samples), end))
                kw.lock_until = end + p.lock * FRAME_SAMPLES
                kw.run = kw.pending = None
                return
        if p.notify is None:
            return
        if kw.run is not None:
            start, peak, peak_t, begin = kw.run
            if score > peak:
                kw.run = [start, score, t, max(0, win_start * self.step_samples)]
            if t >= start + NEAR_MISS_WINDOW_STEPS - 1:
                _, peak, peak_t, begin = kw.run
                kw.pending = (peak, peak_t, begin, (peak_t + 1) * self.step_samples + p.lock * FRAME_SAMPLES)
                kw.run = None
        elif kw.pending is None and end > kw.lock_until and score >= p.notify:
            kw.run = [t, score, t, max(0, win_start * self.step_samples)]
