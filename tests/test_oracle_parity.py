"""Oracle parity suite (Task 11) — the M1+M2 acceptance gate.

Every clip (the espeak ALEXA clip plus the four synthetic negatives from
tools/make_clips.py) is run through BOTH the pinned oracle (real libpryon
under qemu-arm via pypryon.oracle) and the pypryon Spotter; events must
match on (keyword, type, begin/end sample) with |Δscore| ≤ 1e-3 whenever
the oracle event's margin to any active threshold (accept or notify,
including overrides) exceeds 1e-2.  Knife-edge events (margin < 1e-2) are
logged as `boundary` and not failed (Global Constraints / spec goal 2).

The parsed oracle runs are committed as JSON fixtures (derived data); with
the firmware present the live oracle must reproduce them, and the Spotter
is compared against the live run.  Without $EDA_DIR the live runs skip
with a message and the Spotter is still checked against the committed
fixtures on the regenerable synthetic clips.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pypryon import Spotter
from pypryon.modelset import load_model_set

REPO_ROOT = Path(__file__).resolve().parent.parent
EDA_DIR = Path(os.environ.get("EDA_DIR", REPO_ROOT.parent / "echo-dot-assist")).resolve()
MODEL_DIR = EDA_DIR / "firmware/donut/rootfs/system/local/models/keyword/en-US/ALEXA"
ESPRESSO_CLIP = EDA_DIR / "testdata/alexa_espeak.raw"

FIXTURES = Path(__file__).parent / "fixtures"
CLIPS_DIR = Path(__file__).parent / "clips"

SYNTHETIC_CLIPS = ["silence_30s.raw", "sine_30s.raw", "noise_30s.raw", "hum_30s.raw"]


def _ensure_clips() -> dict[str, Path]:
    """(Re)generate the synthetic clips deterministically; return paths."""
    subprocess.run([sys.executable, str(REPO_ROOT / "tools/make_clips.py"),
                    "--out", str(CLIPS_DIR)], check=True, capture_output=True)
    return {name: CLIPS_DIR / name for name in SYNTHETIC_CLIPS}


def _keyword_thresholds() -> dict[str, list[float]]:
    specs = load_model_set(MODEL_DIR).keywords
    return {s.name: [t for t in (s.accept, s.notify) if t is not None]
            for s in specs}


def _assert_event_parity(got, ref, thresholds, clip, errors, boundary):
    if got.keyword != ref["keyword"] or \
            got.type_name != ref["detection_type"]:
        errors.append(f"{clip}: type/keyword {got.keyword}/{got.type_name} "
                      f"vs oracle {ref['keyword']}/{ref['detection_type']}")
        return
    margin = min(abs(ref["classifier_score"] - t)
                 for t in thresholds[ref["keyword"]])
    if margin > 1e-2:
        for attr, key in (("score", "classifier_score"),
                          ("begin_sample", "begin_sample"),
                          ("end_sample", "end_sample")):
            if getattr(got, attr) != ref[key] and attr != "score":
                errors.append(f"{clip}: {attr} {getattr(got, attr)} != {ref[key]}")
        if abs(got.score - ref["classifier_score"]) > 1e-3:
            errors.append(f"{clip}: score {got.score:.6f} vs "
                          f"{ref['classifier_score']:.6f} (Δ>1e-3)")
    else:
        boundary.append(f"{clip}: {ref['detection_type']} "
                        f"{got.score:.6f} vs {ref['classifier_score']:.6f}, "
                        f"margin {margin:.4f}")


@pytest.fixture(scope="session")
def synthetic_clips():
    return _ensure_clips()


@pytest.fixture(scope="session")
def live_oracle(synthetic_clips) -> dict[str, list[dict]] | None:
    """Live oracle runs per clip; None when firmware is absent."""
    if not (MODEL_DIR / "pryon.manifest").exists():
        return None
    from pypryon.oracle import run_oracle
    runs = {}
    clips = {"alexa_espeak.raw": ESPRESSO_CLIP, **synthetic_clips}
    for name, path in clips.items():
        if not path.exists():
            pytest.skip(f"clip {name} absent")
        runs[name] = run_oracle(path)
    return runs


def test_live_oracle_reproduces_fixtures(live_oracle, synthetic_clips) -> None:
    if live_oracle is None:
        pytest.skip("$EDA_DIR firmware absent — live oracle unavailable")
    clips = {"alexa_espeak.raw": ESPRESSO_CLIP, **synthetic_clips}
    for name in clips:
        fixture = FIXTURES / f"{Path(name).stem}_oracle.json"
        assert fixture.exists(), f"missing fixture {fixture.name}"
        assert live_oracle[name] == json.loads(fixture.read_text()), name


def test_spotter_matches_live_oracle(live_oracle, synthetic_clips) -> None:
    if live_oracle is None:
        pytest.skip("$EDA_DIR firmware absent — live oracle unavailable")
    thresholds = _keyword_thresholds()
    clips = {"alexa_espeak.raw": ESPRESSO_CLIP, **synthetic_clips}
    errors: list[str] = []
    boundary: list[str] = []
    for name, path in clips.items():
        dets = Spotter(MODEL_DIR).feed(np.fromfile(path, "<i2"))
        refs = live_oracle[name]
        if len(dets) != len(refs):
            errors.append(f"{name}: {len(dets)} events vs oracle {len(refs)}")
            continue
        for det, ref in zip(dets, refs):
            _assert_event_parity(det, ref, thresholds, name, errors, boundary)
    for line in boundary:
        print(f"boundary: {line}")
    assert not errors, errors


def test_spotter_matches_committed_fixtures_offline(synthetic_clips) -> None:
    """Firmware-free path: synthetic clips vs committed oracle JSON."""
    thresholds = _keyword_thresholds()
    errors: list[str] = []
    boundary: list[str] = []
    for name, path in synthetic_clips.items():
        fixture = FIXTURES / f"{Path(name).stem}_oracle.json"
        assert fixture.exists(), fixture.name
        dets = Spotter(MODEL_DIR).feed(np.fromfile(path, "<i2"))
        refs = json.loads(fixture.read_text())
        if len(dets) != len(refs):
            errors.append(f"{name}: {len(dets)} events vs fixture {len(refs)}")
            continue
        for det, ref in zip(dets, refs):
            _assert_event_parity(det, ref, thresholds, name, errors, boundary)
    for line in boundary:
        print(f"boundary: {line}")
    assert not errors, errors
