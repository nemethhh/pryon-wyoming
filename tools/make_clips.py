#!/usr/bin/env python3
"""Generate the synthetic negative clips for the oracle parity suite.

All clips are fully synthetic (no recordings): deterministic numpy streams
seeded with fixed seeds, written as s16le 16 kHz mono raw PCM to the output
directory (tests/clips/ by default, which is git-ignored; the committed
artifacts are the oracle JSON fixtures, not audio).

Usage: venv/bin/python tools/make_clips.py [--out DIR]
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

SR = 16_000
DURATION_S = 30.0


def silence(n: int) -> np.ndarray:
    return np.zeros(n, np.int16)


def sine(n: int) -> np.ndarray:
    t = np.arange(n) / SR
    return (6000.0 * np.sin(2 * np.pi * 440.0 * t)).astype(np.int16)


def white_noise(n: int) -> np.ndarray:
    return (np.random.default_rng(1717).normal(0, 3000, n)).astype(np.int16)


def tts_hum(n: int) -> np.ndarray:
    """Voice-like hum: 120 Hz glottal harmonics with formant emphasis and a
    3 Hz syllabic envelope — spectral motion without any real speech."""
    t = np.arange(n) / SR
    f0 = 120.0 + 6.0 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    sig = np.zeros(n)
    for k, amp in ((1, 1.0), (2, 0.6), (3, 0.8), (4, 0.4), (5, 0.3), (6, 0.2)):
        formant = 1.0 + 0.7 * np.sin(2 * np.pi * 0.9 * t + k)
        sig += amp * formant * np.sin(k * phase)
    envelope = 0.55 + 0.45 * np.clip(np.sin(2 * np.pi * 3.0 * t), 0, 1)
    return (4500.0 * envelope * sig / np.abs(sig).max() * 0.9).astype(np.int16)


CLIPS: dict[str, object] = {
    "silence_30s.raw": silence,
    "sine_30s.raw": sine,
    "noise_30s.raw": white_noise,
    "hum_30s.raw": tts_hum,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "tests/clips")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    n = int(SR * DURATION_S)
    for name, fn in CLIPS.items():
        pcm = fn(n)
        path = args.out / name
        pcm.tofile(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"{name}: {pcm.size} samples, sha256 {digest[:16]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
