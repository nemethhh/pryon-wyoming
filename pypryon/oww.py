"""openWakeWord-adjacent adapter: `PryonModel.predict(x) -> dict[str,float]`.

Mirrors openWakeWord's Model contract closely enough to sit beside it in a
Wyoming/HA pipeline: feed int16 16 kHz mono chunks (the reference cadence
is one 80 ms frame — 1280 samples — per call; any chunk size works because
the underlying Spotter buffers), read per-keyword continuous scores, and
consume detection events off `.detections` (emptied on read).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .spotter import PryonDetection, Spotter


class PryonModel:
    """One Pryon model directory behind an openWakeWord-style interface."""

    def __init__(self, model_dir: Path, name: str | None = None):
        self.name = name or Path(model_dir).name
        self._spotter = Spotter(model_dir)
        self._pending: list[PryonDetection] = []

    def predict(self, x: np.ndarray) -> dict[str, float]:
        """Feed int16 PCM; returns per-keyword smoothed classifier score."""
        self._pending.extend(self._spotter.feed(x))
        return self._spotter.scores_tail()

    @property
    def detections(self) -> list[PryonDetection]:
        """Detections since the last read (consumed on read)."""
        out, self._pending = self._pending, []
        return out

    def reset(self) -> None:
        self._spotter.reset()
        self._pending.clear()
