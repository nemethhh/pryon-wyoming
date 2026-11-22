"""Synthesize "Alexa" parity clips with Piper TTS (16 kHz mono s16 WAV + raw).

Run with:  uv run --no-project --python 3.12 --with piper-tts==1.8.0 --with scipy \
               python tools/make_tts_clips.py [--out .cache/tts-clips]

Voices come from rhasspy/piper-voices and are cached under <out>/voices. Piper samples
noise, so clips differ between runs: generate once and keep the directory; the parity
tests cache oracle results per clip sha256.
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import tempfile
import urllib.request
import wave
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

VOICES_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US"
VOICES = {"lessac-medium": "lessac/medium", "amy-medium": "amy/medium", "ryan-high": "ryan/high"}
TEXTS = ["Alexa.", "Alexa, what time is it?", "Alexa, turn on the kitchen lights.", "Okay. Alexa, stop."]
RATE = 16000


def _voice(cache: Path, name: str) -> Path:
    model = cache / f"en_US-{name}.onnx"
    for path in (model, model.with_suffix(".onnx.json")):
        if not path.exists():
            url = f"{VOICES_URL}/{VOICES[name]}/{path.name}"
            print(f"downloading {url}", file=sys.stderr)
            urllib.request.urlretrieve(url, path)
    return model


def _synthesize(model: Path, text: str) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "clip.wav"
        subprocess.run([sys.executable, "-m", "piper", "-m", str(model), "-f", str(out)],
                       input=text.encode(), check=True, capture_output=True)
        with wave.open(str(out)) as w:
            rate = w.getframerate()
            pcm = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float64)
    if rate != RATE:
        pcm = resample_poly(pcm, RATE, rate)
    pad = np.zeros(RATE)  # 1 s of silence each side
    return np.clip(np.concatenate([pad, pcm * 0.7, pad]), -32768, 32767).astype("<i2")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path(".cache/tts-clips"))
    args = ap.parse_args()
    voices = args.out / "voices"
    voices.mkdir(parents=True, exist_ok=True)
    sums = []
    n = 0
    for name in VOICES:
        model = _voice(voices, name)
        for text in TEXTS:
            n += 1
            stem = f"{n:02d}-{name}"
            pcm = _synthesize(model, text)
            with wave.open(str(args.out / f"{stem}.wav"), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(RATE)
                w.writeframes(pcm.tobytes())
            sums.append(f"{hashlib.sha256(pcm.tobytes()).hexdigest()}  {stem}.wav")
            print(f"{stem}: {text!r} {pcm.size / RATE:.2f} s")
    (args.out / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
