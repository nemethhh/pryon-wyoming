"""openWakeWord-adjacent adapter + CLI tests (Task 12)."""
from __future__ import annotations

import os
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

from pypryon import Spotter
from pypryon.oww import PryonModel

REPO_ROOT = Path(__file__).resolve().parent.parent
EDA_DIR = Path(os.environ.get("EDA_DIR", REPO_ROOT.parent / "echo-dot-assist")).resolve()
MODEL_DIR = EDA_DIR / "firmware/donut/rootfs/system/local/models/keyword/en-US/ALEXA"
CLIP = EDA_DIR / "testdata/alexa_espeak.raw"

pytestmark = pytest.mark.skipif(not CLIP.exists(), reason="espeak clip absent")

CHUNK = 1280          # 80 ms at 16 kHz


def test_predict_scores_and_detections() -> None:
    model = PryonModel(MODEL_DIR)
    pcm = np.fromfile(CLIP, "<i2")
    scores = []
    for i in range(0, pcm.size, CHUNK):
        scores.append(model.predict(pcm[i:i + CHUNK]))
    alexa = [s["ALEXA"] for s in scores]
    assert max(alexa) >= 0.9
    assert any(s["STOP"] < 0.9 for s in scores)
    dets = model.detections
    assert dets, "no detections consumed"
    assert any(d.detection_type == 2 for d in dets)
    assert model.detections == []      # consumed on read


def test_predict_rejects_float() -> None:
    model = PryonModel(MODEL_DIR)
    with pytest.raises(ValueError, match="int16"):
        model.predict(np.zeros(CHUNK, np.float32))


def _cli(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    return subprocess.run([sys.executable, "-m", "pypryon.cli", *args],
                          capture_output=True, text=True, cwd=cwd or REPO_ROOT,
                          env=env)


def test_cli_run_raw() -> None:
    proc = _cli("run", str(MODEL_DIR), str(CLIP))
    assert proc.returncode == 0, proc.stderr
    assert "ALEXA" in proc.stdout
    assert "Accept" in proc.stdout


def test_cli_run_scores_flag() -> None:
    proc = _cli("run", str(MODEL_DIR), str(CLIP), "--scores")
    assert proc.returncode == 0, proc.stderr
    assert "STOP" in proc.stdout          # scores include every keyword


def test_cli_rejects_48k_wav(tmp_path: Path) -> None:
    wav = tmp_path / "clip48k.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(48000)
        w.writeframes(np.zeros(4800, np.int16).tobytes())
    proc = _cli("run", str(MODEL_DIR), str(wav))
    assert proc.returncode != 0
    assert "16000" in proc.stderr or "16 kHz" in proc.stderr


def test_cli_accepts_16k_wav(tmp_path: Path) -> None:
    pcm = np.fromfile(CLIP, "<i2")
    wav = tmp_path / "clip16k.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm.tobytes())
    proc = _cli("run", str(MODEL_DIR), str(wav))
    assert proc.returncode == 0, proc.stderr
    assert "ALEXA" in proc.stdout


def test_cli_convert_writes_cache_not_model_dir(tmp_path: Path) -> None:
    before = sorted(p.name for p in MODEL_DIR.iterdir())
    work = tmp_path / "work"
    work.mkdir()
    proc = _cli("convert", str(MODEL_DIR), cwd=work)
    assert proc.returncode == 0, proc.stderr
    cache = work / ".pypryon-cache"
    files = list(cache.glob("*.onnx"))
    assert len(files) == 1, proc.stdout
    assert files[0].stat().st_size > 1_000_000
    assert sorted(p.name for p in MODEL_DIR.iterdir()) == before   # untouched


def test_bench_five_x_realtime() -> None:
    import time
    clip = REPO_ROOT / "tests/clips/noise_30s.raw"
    if not clip.exists():
        subprocess.run([sys.executable, str(REPO_ROOT / "tools/make_clips.py")],
                       check=True, capture_output=True)
    t0 = time.monotonic()
    proc = _cli("run", str(MODEL_DIR), str(clip))
    elapsed = time.monotonic() - t0
    assert proc.returncode == 0, proc.stderr
    rate = 30.0 / elapsed
    print(f"\npypryon run on 30 s clip: {elapsed:.2f}s wall → {rate:.1f}× realtime")
    assert rate >= 5.0, f"{rate:.1f}× realtime < 5×"
