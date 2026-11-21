"""LFBE front end vs the bit-exact C golden vector (vendor tests/golden_lfbe.json).

The NumPy port is close, not bit-exact (float64 FFT vs Ne10 float32); per-element
tolerance is 1e-4, and oracle parity is the real gate.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pypryon.fe import frame_count, lfbe_frames
from pypryon.modelset import FeConfig

VENDOR_GOLDEN = Path(__file__).resolve().parents[1] / (
    "vendor/pryon-native/tests/golden_lfbe.json"
)

# ALEXA pryon.config front-end parameters
ALEXA_FE = FeConfig(
    num_mel=64,
    frame_ms=25,
    shift_ms=10,
    preemph=0.96875,
    low_hz=80,
    high_hz=7200,
    fft_pow2=512,
    remove_dc=True,
    window="povey",
    use_energy=0,
)


def golden_pcm(rule: dict) -> np.ndarray:
    """Synthesize the fixture PCM per the JSON rule (vendor test_lfbe_frontend.py)."""
    return (
        (np.arange(rule["samples"], dtype=np.int64) * rule["multiplier"])
        % rule["modulus"]
        - rule["subtract"]
    ).astype(np.int16)


def test_golden_vector_matches_within_1e4() -> None:
    vector = json.loads(VENDOR_GOLDEN.read_text())
    pcm = golden_pcm(vector["input"])
    frames = lfbe_frames(pcm, ALEXA_FE)
    expected = np.asarray(vector["output"], np.float32)
    assert frames.dtype == np.float32
    assert frames.shape == expected.shape == (3, 64)
    diff = np.abs(frames.astype(np.float64) - expected.astype(np.float64))
    assert diff.max() <= 1e-4, f"max abs diff {diff.max()}"


def test_silence_is_log_floor() -> None:
    # Zero PCM: every mel energy floors at log(max(e,1)) = log(1) = 0.
    frames = lfbe_frames(np.zeros(560, np.int16), ALEXA_FE)
    assert frames.shape == (2, 64)
    assert np.all(frames == 0.0)


@pytest.mark.parametrize(
    ("pcm_len", "expected"),
    [
        (0, 0),
        (399, 0),          # one sample short of a single frame
        (400, 1),          # exactly one 25 ms frame
        (560, 2),          # frames at 0 and 160
        (719, 2),          # golden clip minus one sample: third frame incomplete
        (720, 3),          # golden clip: starts 0/160/320
        (132720, 828),     # runner chunk-independence anchor: (828-1)*160+400
    ],
)
def test_frame_count_exact(pcm_len: int, expected: int) -> None:
    assert frame_count(pcm_len) == expected
    assert frame_count(pcm_len, ALEXA_FE) == expected


def test_partial_tail_is_dropped_not_padded() -> None:
    # 720 samples yield 3 frames; a 400-sample call yields 1 — no zero padding
    # of the tail (C loop: s0 + FL <= ns).
    frames = lfbe_frames(golden_pcm({"samples": 400, "multiplier": 73,
                                     "modulus": 20001, "subtract": 10000}), ALEXA_FE)
    assert frames.shape == (1, 64)
