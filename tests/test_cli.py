"""pypryon CLI on the synthetic model."""
from __future__ import annotations

import os
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

from pypryon import __version__

from conftest import REPO_ROOT
from synthmodel import build_model_dir


@pytest.fixture(scope="module")
def synth_dir(tmp_path_factory) -> Path:
    return build_model_dir(tmp_path_factory.mktemp("models") / "SYNTH")


def _cli(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src"))
    return subprocess.run([sys.executable, "-m", "pypryon.cli", *args], capture_output=True,
                          text=True, cwd=cwd or REPO_ROOT, env=env)


def _wav(path: Path, rate=16000, channels=1, width=2, frames=1600) -> Path:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(b"\x00" * frames * channels * width)
    return path


def _last_line(stderr: str) -> str:
    # onnxruntime may log host hardware probing first on some machines
    return stderr.strip().splitlines()[-1] if stderr.strip() else ""


def test_version_flag() -> None:
    proc = _cli("--version")
    assert proc.returncode == 0
    assert proc.stdout.strip() == f"pypryon {__version__}"


def test_run_synthetic_wav(synth_dir, tmp_path) -> None:
    proc = _cli("run", str(synth_dir), str(_wav(tmp_path / "ok.wav")), "--scores")
    assert proc.returncode == 0, proc.stderr
    assert "score HEY" in proc.stdout and "score STOP" in proc.stdout


def test_convert_output_flag(synth_dir, tmp_path) -> None:
    out = tmp_path / "sub" / "portable.onnx"
    proc = _cli("convert", str(synth_dir), "-o", str(out))
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == str(out)
    ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])


def _odd_raw(tmp_path: Path) -> Path:
    p = tmp_path / "odd.raw"
    p.write_bytes(b"\x00" * 3201)
    return p


@pytest.mark.parametrize(("make", "needle"), [
    (lambda d: _wav(d / "stereo.wav", channels=2), "mono"),
    (lambda d: _wav(d / "8bit.wav", width=1), "16-bit"),
    (lambda d: _wav(d / "44k.wav", rate=44100), "44100"),
    (_odd_raw, "odd"),
    (lambda d: d / "missing.wav", "missing.wav"),
])
def test_cli_bad_audio_exit_2(synth_dir, tmp_path, make, needle) -> None:
    proc = _cli("run", str(synth_dir), str(make(tmp_path)))
    assert proc.returncode == 2, proc.stderr
    assert "Traceback" not in proc.stderr
    assert _last_line(proc.stderr).startswith("pypryon: error:"), proc.stderr
    assert needle in _last_line(proc.stderr)


def test_cli_not_a_model_dir_exit_2(tmp_path) -> None:
    proc = _cli("run", str(tmp_path), str(_wav(tmp_path / "ok.wav")))
    assert proc.returncode == 2
    assert "Traceback" not in proc.stderr
    assert _last_line(proc.stderr).startswith("pypryon: error:"), proc.stderr


def test_load_pcm16_raw_roundtrip(tmp_path) -> None:
    from pypryon.audio import load_pcm16
    pcm = np.arange(-50, 50, dtype=np.int16)
    raw = tmp_path / "x.raw"
    pcm.astype("<i2").tofile(raw)
    np.testing.assert_array_equal(load_pcm16(raw), pcm)


def test_convert_davs_set_is_a_clear_error(tmp_path) -> None:
    from synthmodel import build_davs_model_dir

    d = build_davs_model_dir(tmp_path / "DAVS")
    proc = _cli("convert", str(d), "-o", str(tmp_path / "x.onnx"))
    assert proc.returncode == 2
    assert "Traceback" not in proc.stderr
    assert "DAVS" in _last_line(proc.stderr)
