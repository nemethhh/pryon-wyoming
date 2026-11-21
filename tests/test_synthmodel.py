"""The synthetic Pryon-format model exercises the loader, the surgery and the runner without firmware."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import pytest

from pypryon.graph import make_portable
from pypryon.modelset import load_model_set
from pypryon.runner import GraphRunner
from synthmodel import build_model_dir
from test_graph_surgery import _checker_minus_hybrid, _int32_shape_inputs, _load_and_classify

EXPECTED_INPUTS = {"input_feats", "streaming_conv_buffer_input_1",
                   "input_h_ww", "input_c_ww", "input_h_se", "input_c_se"}


@pytest.fixture(scope="module")
def synth_dir(tmp_path_factory) -> Path:
    return build_model_dir(tmp_path_factory.mktemp("models") / "SYNTH")


def test_load_synthetic_model_dir(synth_dir) -> None:
    ms = load_model_set(synth_dir)
    assert [k.name for k in ms.keywords] == ["HEY", "STOP"]
    assert ms.words == {0: "<eps>", 1: "HEY", 2: "STOP"}
    assert len(ms.state_inputs) == 5 and ms.state_inputs[-1] == "streaming_conv_buffer_input_1"


def test_surgery_fixes_every_quirk(synth_dir) -> None:
    raw = (synth_dir / "model.onnx").read_bytes()
    with pytest.raises(onnx.checker.ValidationError):
        _checker_minus_hybrid(onnx.load_from_string(raw))
    out = make_portable(raw, rewrite_hybrid=False)
    model = onnx.load_from_string(out)
    assert _int32_shape_inputs(model.graph) == []
    assert {i.name for i in model.graph.input} == EXPECTED_INPUTS
    _checker_minus_hybrid(model)
    assert _load_and_classify(out) == "hybrid"
    assert make_portable(out, rewrite_hybrid=False) == out


def test_portable_runs_in_ort(synth_dir) -> None:
    runner = GraphRunner(load_model_set(synth_dir))
    frames = np.random.default_rng(3).normal(8, 3, (30 * 6, 64)).astype(np.float32)
    post = runner.push_frames(frames)
    assert post.lanes.shape == (30, 25, 3) and post.se_lanes.shape == (30, 25, 2)
    assert np.isfinite(post.lanes).all() and np.isfinite(post.se_lanes).all()
    np.testing.assert_allclose(post.lanes.sum(axis=-1), 1.0, atol=1e-5)
