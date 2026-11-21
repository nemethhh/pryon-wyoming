"""Load 16 kHz mono s16 audio files (WAV, or headerless raw s16le)."""
from __future__ import annotations

import os
import wave
from pathlib import Path

import numpy as np

from ._constants import SAMPLE_RATE


def load_pcm16(path: str | os.PathLike) -> np.ndarray:
    """Return the file's samples as int16; ``.wav`` must be 16 kHz mono 16-bit."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"{path}: no such file")
    if path.suffix.lower() != ".wav":
        size = path.stat().st_size
        if size % 2:
            raise ValueError(f"{path.name}: odd byte count {size}, not s16le raw PCM")
        return np.fromfile(path, "<i2").astype(np.int16)
    try:
        with wave.open(str(path), "rb") as w:
            rate, channels, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
            data = w.readframes(w.getnframes())
    except (wave.Error, EOFError) as exc:
        raise ValueError(f"{path.name}: not a readable WAV file ({exc})") from exc
    if rate != SAMPLE_RATE:
        raise ValueError(f"{path.name} is {rate} Hz; pypryon takes 16000 Hz (resample first)")
    if channels != 1:
        raise ValueError(f"{path.name} has {channels} channels; pypryon takes mono")
    if width != 2:
        raise ValueError(f"{path.name} is {8 * width}-bit; pypryon takes 16-bit")
    return np.frombuffer(data, "<i2").astype(np.int16)
