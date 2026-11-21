"""Streaming graph runner tests (Task 8).

The runner drives the ported graph one engine cycle (6 LFBE frames) per
session.run — the cycle is the engine's quantum (scorer.score_upsampling:
initialize.frame_count 6, skip-frame-count 5; provenance: kw.cfg.json /
vendor LEDGER) — carrying LSTM h/c + conv buffers, and doing the host-side
lane-ring zeroing of the vendor LSTM contract (Model.stream:
``h[t % 25 if t else slice(None)] = 0``, pryon_lstm_analysis.py: each lane
a ≤1.5 s sliding window, host resets lanes ring-wise).

Posteriors carry the full 25-lane softmax ring (``lanes``) in addition to
the per-cycle per-class max views (``ww``/``se``): the kw.cfg decoder rule
is defined over the ring by lane age (vendor pryon_decoder.cycle_score),
which a collapsed max cannot reproduce.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from pypryon.fe import lfbe_frames
from pypryon.modelset import load_model_set
from pypryon.runner import GraphRunner

from conftest import MODEL_DIR, REPO_ROOT, requires_model

pytestmark = requires_model


def golden_pcm(n_samples: int) -> np.ndarray:
    """LCG golden rule (vendor tests / Task 5): i*73 % 20001 - 10000."""
    return ((np.arange(n_samples, dtype=np.int64) * 73) % 20001 - 10000).astype(np.int16)


@pytest.fixture(scope="module")
def model_set():
    return load_model_set(MODEL_DIR)


@pytest.fixture(scope="module")
def frames828(model_set) -> np.ndarray:
    # (828-1)*160+400 = 132,720 samples → exactly 828 LFBE frames
    return lfbe_frames(golden_pcm(132_720), model_set.fe)


def test_chunk_independence(model_set, frames828) -> None:
    assert frames828.shape == (828, 64)
    whole = GraphRunner(model_set).push_frames(frames828)
    assert whole.lanes.shape == (138, 25, 3)
    assert whole.ww.shape == (138, 3)

    rng = np.random.default_rng(3)
    runner = GraphRunner(model_set)
    parts = []
    fed = 0
    while fed < 828:
        n = int(rng.choice([1, 7, 160, 10_000]))
        chunk = frames828[fed:fed + n]
        parts.append(runner.push_frames(chunk))
        fed += chunk.shape[0]
    lanes = np.concatenate([p.lanes for p in parts])
    ww = np.concatenate([p.ww for p in parts])
    se = np.concatenate([p.se for p in parts])

    np.testing.assert_array_equal(lanes, whole.lanes)
    np.testing.assert_array_equal(ww, whole.ww)
    np.testing.assert_array_equal(se, whole.se)


def test_reset_leaves_no_stale_state(model_set, frames828) -> None:
    runner = GraphRunner(model_set)
    first = runner.push_frames(frames828[:360])
    runner.reset()
    again = runner.push_frames(frames828[:360])
    np.testing.assert_array_equal(first.lanes, again.lanes)

    # after reset the runner is indistinguishable from a fresh one on new audio
    other = lfbe_frames(golden_pcm(20_000), model_set.fe)
    runner.reset()
    post_reset = runner.push_frames(other)
    fresh = GraphRunner(model_set).push_frames(other)
    np.testing.assert_array_equal(post_reset.lanes, fresh.lanes)
    np.testing.assert_array_equal(post_reset.se, fresh.se)


def test_partial_chunks_buffer_until_cycle(model_set, frames828) -> None:
    runner = GraphRunner(model_set)
    assert runner.push_frames(frames828[:5]).ww.shape == (0, 3)   # < 1 cycle
    out = runner.push_frames(frames828[5:7])                      # 7 total: cycle 0
    assert out.ww.shape == (1, 3)


def test_parity_vs_vendor_model_stream(model_set, frames828) -> None:
    os.environ["PRYON_MODEL"] = str(model_set.graph_path)
    sys.path.insert(0, str(REPO_ROOT / "vendor/pryon-native"))
    from pryon_quant_model import Model

    frames = frames828[:360]  # 60 cycles
    want = list(Model(str(model_set.graph_path)).stream(frames))
    got = GraphRunner(model_set).push_frames(frames)

    assert got.lanes.shape == (60, 25, 3)
    np.testing.assert_allclose(
        got.lanes, np.stack([w["softmax_ww"] for w in want]),
        atol=1e-5, err_msg="lanes vs Model.stream")
    np.testing.assert_allclose(
        got.se, np.stack([w["se"].max(0) for w in want]),
        atol=1e-5, err_msg="se collapse vs Model.stream")


def test_client_property_roundtrip(model_set) -> None:
    runner = GraphRunner(model_set)
    assert runner.client_properties == model_set.client_prop_defaults
    runner.push_client_property("AlarmState", 1)
    assert runner.client_properties["AlarmState"] == 1
