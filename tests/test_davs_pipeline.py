"""Generalized streaming pipeline: encoder stage (8 frames/step) feeding decoder stages."""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from pypryon import Spotter, load_model_set
from pypryon.graph import initial_state
from pypryon.pipeline import DavsPipeline, StageRunner

from synthmodel import DAVS_LANES, build_davs_model_dir, build_model_dir

FRAMES = np.random.default_rng(4).normal(8, 3, (8 * 30 + 5, 64)).astype(np.float32)


@pytest.fixture(scope="module")
def davs_set(tmp_path_factory):
    return load_model_set(build_davs_model_dir(tmp_path_factory.mktemp("models") / "DAVS"))


def test_davs_chunk_independence(davs_set) -> None:
    whole = DavsPipeline(davs_set).push_frames(FRAMES)
    assert whole["ww_alexa/ww_det_scores"].shape == (30, DAVS_LANES, 3)
    assert whole["ww_stop/mww/posteriors"].shape == (30, 1, 4)   # [steps, lanes, classes]
    pipe = DavsPipeline(davs_set)
    parts: dict[str, list] = {}
    rng = np.random.default_rng(1)
    i = 0
    while i < len(FRAMES):
        n = int(rng.choice([1, 3, 7, 8, 13, 40]))
        for key, arr in pipe.push_frames(FRAMES[i:i + n]).items():
            parts.setdefault(key, []).append(arr)
        i += n
    for key, arr in whole.items():
        np.testing.assert_array_equal(np.concatenate(parts[key]), arr, err_msg=key)


def test_reset_restores_primed_state(davs_set) -> None:
    fresh = DavsPipeline(davs_set).push_frames(FRAMES[:16])
    pipe = DavsPipeline(davs_set)
    pipe.push_frames(FRAMES)
    pipe.reset()
    again = pipe.push_frames(FRAMES[:16])
    for key in fresh:
        np.testing.assert_array_equal(again[key], fresh[key], err_msg=key)
    buf = "conv_buffer_input/Concat:2"
    stored = initial_state(davs_set.encoder.path.read_bytes(), [buf])[buf]
    np.testing.assert_array_equal(pipe.stages["WW_encoder"].initial[buf], stored)
    assert np.any(stored != 0)


class _Recorder:
    def __init__(self, session):
        self.session, self.feeds = session, []

    def __getattr__(self, name):
        return getattr(self.session, name)

    def run(self, outputs, feeds):
        self.feeds.append({k: np.array(v) for k, v in feeds.items()})
        return self.session.run(outputs, feeds)


def test_lane_ring_16(davs_set) -> None:
    pipe = DavsPipeline(davs_set)
    stage = pipe.stages["ww_alexa"]
    stage.session = _Recorder(stage.session)
    pipe.push_frames(FRAMES)
    feeds = stage.session.feeds
    assert (feeds[0]["scores_h0"] == 0).all()
    for t in range(1, len(feeds)):
        h = feeds[t]["scores_h0"]
        assert (h[t % DAVS_LANES] == 0).all(), t
        assert np.abs(np.delete(h, t % DAVS_LANES, axis=0)).sum() > 0, t


def test_stage_runner_rejects_wrong_input(davs_set) -> None:
    pipe = DavsPipeline(davs_set)
    with pytest.raises(ValueError, match="frames"):
        pipe.push_frames(np.zeros((8, 63), np.float32))
    assert isinstance(pipe.stages["WW_encoder"], StageRunner)


OLD_BASELINE = [("HEY", 2, 0.000438966, 0, 5760), ("HEY", 2, 0.000445816, 12640, 22080),
                ("HEY", 2, 0.000445816, 29920, 38400), ("HEY", 2, 0.000445816, 41440, 54720),
                ("HEY", 2, 0.000445816, 62560, 71040), ("HEY", 2, 0.000445816, 78880, 87360),
                ("HEY", 2, 0.000445816, 94240, 103680), ("HEY", 2, 0.000445816, 111520, 120000),
                ("HEY", 2, 0.000445816, 127840, 136320), ("HEY", 2, 0.000445816, 144160, 152640)]


def test_old_spotter_unchanged(tmp_path: Path) -> None:
    noise = (np.random.default_rng(5).normal(0, 3000, 16000 * 10)).astype(np.int16)
    events = Spotter(build_model_dir(tmp_path / "SYNTH"), thresholds={"HEY": 1e-4}).feed(noise)
    got = [dataclasses.astuple(e) for e in events]
    assert [(k, t, round(s, 9), b, e) for k, t, s, b, e in got] == OLD_BASELINE
