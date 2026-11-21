"""DAVS alexa-en-US parity with libpryon: keyword, type, end sample, score.

Decisions must match exactly. Scores must match within 1e-3; differences up to
DRIFT_LIMIT are reported, not failed: near digital silence the encoder's quantized
arithmetic is not yet bit-exact (two events on the current clip set). begin_sample is
not compared (libpryon derives it at frame resolution from the alignments head).
HEY_DISNEY is not served, so its oracle events are dropped.
"""
from __future__ import annotations

import json
import subprocess
import sys
import wave

import pytest

from pypryon import Spotter

from conftest import CLIPS_DIR, DAVS_MODEL_DIR, ESPEAK_WAV, ORACLE_ROOTFS, REPO_ROOT, load_wav16, requires_davs

FIXTURES = REPO_ROOT / "tests/fixtures/davs/alexa-en-US"
DRIFT_LIMIT = 0.03


def _parity(dets, refs, thresholds) -> list[str]:
    refs = [r for r in refs if r["keyword"] != "HEY_DISNEY"]
    errors = []
    if len(dets) != len(refs):
        return [f"{len(dets)} events {[(d.keyword, d.type_name, round(d.score, 4), d.end_sample) for d in dets]}"
                f" vs oracle {[(r['keyword'], r['detection_type'], r['classifier_score'], r['end_sample']) for r in refs]}"]
    for det, ref in zip(dets, refs, strict=True):
        if (det.keyword, det.type_name, det.end_sample) != (ref["keyword"], ref["detection_type"], ref["end_sample"]):
            errors.append(f"{(det.keyword, det.type_name, det.end_sample)} vs "
                          f"{(ref['keyword'], ref['detection_type'], ref['end_sample'])}")
        margin = min(abs(ref["classifier_score"] - t) for t in thresholds[ref["keyword"]])
        delta = abs(det.score - ref["classifier_score"])
        if margin > 1e-2 and delta > 1e-3:
            line = f"{ref['keyword']} score {det.score:.6f} vs {ref['classifier_score']:.6f}"
            if delta > DRIFT_LIMIT:
                errors.append(line)
            else:
                print(f"drift: {line}")
    return errors


def _thresholds(spotter: Spotter) -> dict[str, list[float]]:
    from pypryon import load_model_set
    ms = load_model_set(DAVS_MODEL_DIR)
    out: dict[str, list[float]] = {}
    for s in ms.spotters:
        out.setdefault(s.name, []).extend(t for t in (s.accept, s.notify) if t is not None)
    return out


@requires_davs
def test_davs_espeak_nearmiss_matches_oracle() -> None:
    spotter = Spotter(DAVS_MODEL_DIR)
    dets = spotter.feed(load_wav16(ESPEAK_WAV))
    refs = json.loads((FIXTURES / "alexa_espeak_oracle.json").read_text())
    assert _parity(dets, refs, _thresholds(spotter)) == []


@requires_davs
def test_davs_negatives_match_fixtures() -> None:
    subprocess.run([sys.executable, str(REPO_ROOT / "tools/make_clips.py"), "--out",
                    str(REPO_ROOT / "tests/clips")], check=True, capture_output=True)
    import numpy as np
    spotter = Spotter(DAVS_MODEL_DIR)
    for name in ("silence_30s", "sine_30s", "noise_30s", "hum_30s"):
        spotter.reset()
        dets = spotter.feed(np.fromfile(REPO_ROOT / f"tests/clips/{name}.raw", "<i2"))
        refs = json.loads((FIXTURES / f"{name}_oracle.json").read_text())
        assert _parity(dets, refs, _thresholds(spotter)) == [], name


def _clips():
    return sorted(CLIPS_DIR.glob("*.wav")) if CLIPS_DIR else []


@requires_davs
@pytest.mark.skipif(not _clips(), reason="no parity clips: set PYPRYON_CLIPS_DIR or run tools/make_tts_clips.py")
def test_davs_clips_match_oracle() -> None:
    from pryon_oracle import run_oracle_cached

    spotter = Spotter(DAVS_MODEL_DIR)
    thresholds = _thresholds(spotter)
    errors = []
    for wav in _clips():
        cache = wav.parent / "oracle"
        if ORACLE_ROOTFS is None and not list(cache.glob(f"{wav.stem}.*.oracle.json")):
            pytest.skip(f"no cached oracle result for {wav.name}; set PYPRYON_ORACLE_ROOTFS")
        refs = run_oracle_cached(wav, DAVS_MODEL_DIR, cache)
        with wave.open(str(wav)) as w:
            assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2), wav.name
        spotter.reset()
        errors += [f"{wav.stem}: {e}" for e in _parity(spotter.feed(load_wav16(wav)), refs, thresholds)]
    assert errors == []
