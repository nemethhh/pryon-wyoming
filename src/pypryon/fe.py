"""NumPy port of the stock Pryon LFBE front end.

Semantics mirror ``vendor/pryon-native/pryon_lfbe.c`` — the float32
reimplementation measured bit-exact against the device library (vendor
LEDGER.md, "Frontend"):

* int16 -> float32 frames, 400 samples / 160 hop from sample 0, no padding;
* DC removal (per-frame mean), preemphasis 0.96875 applied descending on the
  un-windowed frame (first sample scaled by 1 - coeff), then the Povey
  window (0.5 - 0.5*cos(2*pi*i/(L-1)))**0.85 computed in float64 and rounded
  to float32;
* rFFT-512 (Kaldi packing), power spectrum re^2 + im^2 with DC/Nyquist
  energy in bins 0/256;
* Kaldi MelBanks, 64 bins 80..7200 Hz, MelScale 1127*log(1 + f/700), no HTK
  bin-0 quirk, triangle weights over the 257 power bins;
* log with floor max(e, 1).

Achieved tolerance vs the bit-exact C golden vector
(vendor/pryon-native/tests/golden_lfbe.json): max abs diff 3.9e-6 per element
(the C path's float32 Ne10 FFT / msun logf vs numpy float64 FFT / np.log);
asserted <= 1e-4 in tests/test_fe_golden.py.  End-to-end oracle parity
(Task 11) is the behavioral gate.

All constants carry their pryon.config provenance (Task 4 / modelset.py);
the 16 kHz sample rate is the generation-wide fe.audio_analysis.samplerate
(not stored in FeConfig — every model dir of this generation is 16 kHz).
"""
from __future__ import annotations

import numpy as np

from .modelset import FeConfig

_SAMPLE_RATE = 16000  # pryon.config fe.audio_analysis.samplerate (spec §"known")

# Generation defaults (ALEXA pryon.config); used when no cfg is passed to
# frame_count().
DEFAULT_FE = FeConfig(
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


def _frame_length(cfg: FeConfig) -> int:
    # pryon.config window length: frame_ms * rate / 1000 = 25 ms * 16 = 400
    # (pre round_to_power_of_two; cfg.fft_pow2 is the rounded FFT size).
    return round(cfg.frame_ms * _SAMPLE_RATE / 1000)


def _hop(cfg: FeConfig) -> int:
    # pryon.config fe.audio_analysis.frame_shift_milli * rate / 1000 = 160
    return round(cfg.shift_ms * _SAMPLE_RATE / 1000)


def frame_count(pcm_len: int, cfg: FeConfig | None = None) -> int:
    """Complete 25 ms frames in ``pcm_len`` samples (tail dropped, no padding).

    Mirrors pryon_lfbe.c ``for (s0 = 0; s0 + FL <= ns; s0 += HOP)``.
    """
    if cfg is None:
        cfg = DEFAULT_FE
    fl, hop = _frame_length(cfg), _hop(cfg)
    if pcm_len < fl:
        return 0
    return (pcm_len - fl) // hop + 1


def _povey_window(length: int) -> np.ndarray:
    # pryon_lfbe.c: pow(0.5 - 0.5*cos(2*pi*i/(FL-1)), 0.85) in double, stored
    # as float (Kaldi ProcessWindow povey window).
    i = np.arange(length, dtype=np.float64)
    return ((0.5 - 0.5 * np.cos(2.0 * np.pi * i / (length - 1))) ** 0.85).astype(
        np.float32
    )


def _mel_bank(cfg: FeConfig) -> np.ndarray:
    # Kaldi MelBanks float weights (pryon_lfbe.c main()): MelScale
    # 1127*log(1 + f/700), num_mel+2 linspace edges low..high Hz, triangle
    # weights over the n_fft/2+1 power bins, no HTK bin-0 quirk.
    n_bins = cfg.fft_pow2 // 2 + 1
    bin_hz = np.arange(n_bins, dtype=np.float64) * _SAMPLE_RATE / cfg.fft_pow2

    def mel(f: np.ndarray) -> np.ndarray:
        return 1127.0 * np.log1p(f / 700.0)

    grid = mel(bin_hz)
    edges = np.linspace(mel(np.float64(cfg.low_hz)), mel(np.float64(cfg.high_hz)),
                        cfg.num_mel + 2)
    lo, ce, hi = edges[:-2], edges[1:-1], edges[2:]
    rise = (grid[:, None] - lo) / (ce - lo)
    fall = (hi - grid[:, None]) / (hi - ce)
    return np.maximum(0.0, np.minimum(rise, fall))  # (n_bins, num_mel)


def lfbe_frames(pcm: np.ndarray, cfg: FeConfig) -> np.ndarray:
    """int16 mono 16 kHz PCM -> (n_frames, cfg.num_mel) float32 LFBE frames."""
    pcm = np.asarray(pcm)
    if pcm.ndim != 1:
        raise ValueError(f"expected 1-D PCM, got shape {pcm.shape}")
    if pcm.dtype != np.int16:
        raise ValueError(f"expected int16 PCM, got dtype {pcm.dtype}")
    if cfg.window != "povey":
        raise ValueError(f"unsupported window {cfg.window!r} (only 'povey')")

    fl, hop, n_fft = _frame_length(cfg), _hop(cfg), cfg.fft_pow2
    n = frame_count(pcm.size, cfg)
    if n == 0:
        return np.zeros((0, cfg.num_mel), np.float32)

    # Frame math in float32, mirroring the C path's float32 ProcessWindow.
    fr = np.lib.stride_tricks.sliding_window_view(pcm, fl)[::hop].astype(np.float32)
    fr = fr[:n].copy()
    if cfg.remove_dc:
        # C: mean = -sdot(w, one)/FL; w[i] += mean.
        fr -= fr.mean(axis=1, keepdims=True)
    # Preemphasis descending on the un-windowed frame: w[i] -= c*w[i-1] for
    # i > 0 (w[i-1] still original), w[0] -= c*w[0].  The RHS is fully
    # evaluated before assignment, so original values are read.
    fr[:, 1:] = fr[:, 1:] - np.float32(cfg.preemph) * fr[:, :-1]
    fr[:, 0] *= np.float32(1.0 - cfg.preemph)
    fr *= _povey_window(fl)

    # Zero-pad to n_fft; numpy computes the rFFT in float64 (pocketfft) —
    # the C path's float32 Ne10 FFT is the dominant residual vs golden.
    spec = np.fft.rfft(fr.astype(np.float64), n=n_fft)
    power = spec.real**2 + spec.imag**2  # (n, n_fft/2 + 1), DC/Nyquist included

    energy = power @ _mel_bank(cfg)
    # msun logf in C; np.log here. Floor max(e, 1) -> 0.
    return np.log(np.maximum(energy, 1.0)).astype(np.float32)
