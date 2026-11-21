"""Shared test configuration.

Tests that need a real Pryon model read it from ``PYPRYON_MODEL_DIR`` (an
extracted ALEXA keyword directory) and skip without it; the live qemu oracle
additionally needs ``PYPRYON_ORACLE_ROOTFS``.
"""
from __future__ import annotations

import os
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
ESPEAK_WAV = REPO_ROOT / "tests/data/alexa_espeak.wav"

# dev-only oracle tooling, not part of the package
sys.path.insert(0, str(REPO_ROOT / "tools/oracle"))


def _env_dir(name: str) -> Path | None:
    value = os.environ.get(name)
    if not value:
        return None
    path = Path(value).expanduser().resolve()
    return path if path.is_dir() else None


MODEL_DIR = _env_dir("PYPRYON_MODEL_DIR")

requires_model = pytest.mark.skipif(
    MODEL_DIR is None, reason="set PYPRYON_MODEL_DIR to an extracted ALEXA model dir"
)


ORACLE_ROOTFS = _env_dir("PYPRYON_ORACLE_ROOTFS")
DAVS_MODEL_DIR = _env_dir("PYPRYON_DAVS_MODEL_DIR")
# parity clips: the author's recordings, else locally generated TTS clips (tools/make_tts_clips.py)
CLIPS_DIR = _env_dir("PYPRYON_CLIPS_DIR") or (
    REPO_ROOT / ".cache/tts-clips" if (REPO_ROOT / ".cache/tts-clips").is_dir() else None)

requires_davs = pytest.mark.skipif(
    DAVS_MODEL_DIR is None, reason="set PYPRYON_DAVS_MODEL_DIR to an extracted DAVS alexa-en-US set")
requires_davs_oracle = pytest.mark.skipif(
    DAVS_MODEL_DIR is None or ORACLE_ROOTFS is None,
    reason="set PYPRYON_DAVS_MODEL_DIR and PYPRYON_ORACLE_ROOTFS for the live DAVS oracle")

requires_oracle = pytest.mark.skipif(
    MODEL_DIR is None or ORACLE_ROOTFS is None,
    reason="set PYPRYON_MODEL_DIR and PYPRYON_ORACLE_ROOTFS for the live oracle",
)


@pytest.fixture
def model_dir() -> Path:
    if MODEL_DIR is None:
        pytest.skip("set PYPRYON_MODEL_DIR to an extracted ALEXA model dir")
    return MODEL_DIR


def load_wav16(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2)
        return np.frombuffer(w.readframes(w.getnframes()), "<i2").copy()
