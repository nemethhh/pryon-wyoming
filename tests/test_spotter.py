"""Spotter behavior on the synthetic model (no firmware needed)."""
from __future__ import annotations

import dataclasses
import os
import stat
from pathlib import Path

import numpy as np
import pytest

from pypryon import Spotter

from synthmodel import build_model_dir

NOISE = (np.random.default_rng(5).normal(0, 3000, 16000 * 10)).astype(np.int16)
LOW = {"HEY": 1e-4}  # below every synthetic HEY score: forces accepts


@pytest.fixture(scope="module")
def synth_dir(tmp_path_factory) -> Path:
    return build_model_dir(tmp_path_factory.mktemp("models") / "SYNTH")


def _astuples(events):
    return [dataclasses.astuple(e) for e in events]


def test_last_scores_one_per_cycle(synth_dir) -> None:
    sp = Spotter(synth_dir)
    sp.feed(NOISE[:9600 + 240])          # 10 complete cycles (+ the 25 ms window tail)
    for name in ("HEY", "STOP"):
        scores = sp.last_scores[name]
        assert scores.shape == (10,)
        assert (scores[:5] == 0.0).all() and (scores[5:] > 0.0).all()
    sp.feed(NOISE[:500])
    assert all(v.shape == (0,) for v in sp.last_scores.values())


def test_keywords_and_thresholds(synth_dir) -> None:
    sp = Spotter(synth_dir, thresholds={"STOP": 0.7})
    assert sp.keywords == ["HEY", "STOP"]
    assert sp.thresholds == {"HEY": 0.9, "STOP": 0.7}


def test_unknown_threshold_keyword_rejected(synth_dir) -> None:
    with pytest.raises(ValueError, match=r"'HAY'.*HEY, STOP"):
        Spotter(synth_dir, thresholds={"HAY": 0.5})


def test_model_dir_forms_and_read_only(synth_dir, tmp_path, monkeypatch) -> None:
    Spotter(str(synth_dir) + "/")
    home = tmp_path / "home"
    ro = build_model_dir(home / "models" / "RO")
    monkeypatch.setenv("HOME", str(home))
    for p in [ro, *ro.iterdir()]:
        p.chmod(p.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    try:
        before = sorted(os.listdir(ro))
        sp = Spotter("~/models/RO")
        sp.feed(NOISE[:16000])
        assert sorted(os.listdir(ro)) == before
    finally:
        for p in [ro, *ro.iterdir()]:
            p.chmod(p.stat().st_mode | stat.S_IWUSR)


def test_detections_have_nonnegative_int_samples(synth_dir) -> None:
    events = Spotter(synth_dir, thresholds=LOW).feed(NOISE)
    assert events
    for e in events:
        assert type(e.begin_sample) is int and type(e.end_sample) is int
        assert 0 <= e.begin_sample < e.end_sample


def test_chunked_equals_whole_synthetic(synth_dir) -> None:
    whole = Spotter(synth_dir, thresholds=LOW).feed(NOISE)
    sp = Spotter(synth_dir, thresholds=LOW)
    out, fed = [], 0
    rng = np.random.default_rng(0)
    while fed < NOISE.size:
        n = int(rng.choice([1, 159, 961, 1280, 13_333]))
        out += sp.feed(NOISE[fed:fed + n])
        fed += n
    assert _astuples(out) == _astuples(whole)


def test_reset_restores_fresh_behavior_synthetic(synth_dir) -> None:
    sp = Spotter(synth_dir, thresholds=LOW)
    first = _astuples(sp.feed(NOISE))
    sp.reset()
    assert _astuples(sp.feed(NOISE)) == first
