"""Mechanical graph surgery tests (Task 6).

The real-graph test exercises the shipped ALEXA ONNX from the
echo-dot-assist checkout ($EDA_DIR, else ../echo-dot-assist): after
``make_portable`` the graph must be mechanically loadable — int32
shape-position inputs cast to int64, initializer names out of graph.input
(keeping the true runtime/state inputs), onnx.checker clean — with stock
onnxruntime still refusing it *only* over the unknown Hybrid* ops (the
ORT_LOAD_ERROR_HYBRID condition Task 7 resolves).

A firmware-free synthetic test pins the same mechanics on a graph small
enough that ORT runs it end-to-end after surgery.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import helper, numpy_helper

from pypryon.graph import ORT_LOAD_ERROR_HYBRID, make_portable

# Task 7 grew make_portable's default to also rewrite the Hybrid* ops; the
# mechanical-only layer it builds on is still exposed (and tested here) via
# rewrite_hybrid=False.

REPO_ROOT = Path(__file__).resolve().parent.parent
EDA_DIR = Path(os.environ.get("EDA_DIR", REPO_ROOT.parent / "echo-dot-assist")).resolve()
ALEXA_ONNX = (
    EDA_DIR
    / "firmware/donut/rootfs/system/local/models/keyword/en-US/ALEXA"
    / "runtime_stateless_standard.streaming.fixed.int16.onnx"
)

# Shape-input positions that ONNX/ORT require to be int64 (opset 11): Reshape
# `shape` (input 1), Slice `starts`/`ends`/`axes` (inputs 1-3), Tile `repeats`.
INT64_POSITIONS = {"Reshape": (1,), "Slice": (1, 2, 3), "Tile": (1,)}

# The shipped graph's true runtime inputs: features + 4 LSTM states + 5
# streaming conv buffers (nemort_scorer_config.json; empty placeholder
# initializers in the ONNX are exporter artifacts, not constants).
EXPECTED_INPUTS = {
    "input_feats",
    "input_h_ww", "input_c_ww", "input_h_se", "input_c_se",
    "streaming_conv_buffer_input_7",
    "streaming_conv_buffer_input_15",
    "streaming_conv_buffer_input_23",
    "streaming_conv_buffer_input_30",
    "streaming_conv_buffer_input_37",
}


def _int32_shape_inputs(graph) -> list[str]:
    init_dtypes = {i.name: i.data_type for i in graph.initializer}
    bad = []
    for node in graph.node:
        for pos in INT64_POSITIONS.get(node.op_type, ()):
            if pos < len(node.input):
                name = node.input[pos]
                if init_dtypes.get(name) == onnx.TensorProto.INT32:
                    bad.append(f"{node.name}:{pos}:{name}")
    return bad


def _checker_minus_hybrid(model: onnx.ModelProto) -> None:
    """Structural check with Hybrid* ops on a scratch custom domain.

    Hybrid* ops are declared in the default ai.onnx domain (spec §"What is
    already known"), which no stock checker accepts; renaming their domain
    for the check makes the checker tolerate them as custom ops while still
    validating every other node, initializer and io declaration.
    """
    import copy

    checked = copy.deepcopy(model)
    for node in checked.graph.node:
        if node.op_type.startswith("Hybrid"):
            node.domain = "ai.pypryon.checker"
    checked.opset_import.append(helper.make_opsetid("ai.pypryon.checker", 1))
    onnx.checker.check_model(checked)


def _load_and_classify(data: bytes) -> str:
    """'ok', 'hybrid' (unknown Hybrid* ops only) or the raw error text."""
    try:
        ort.InferenceSession(data, providers=["CPUExecutionProvider"])
        return "ok"
    except Exception as exc:  # noqa: BLE001 - classification is the test
        return "hybrid" if ORT_LOAD_ERROR_HYBRID in str(exc) else str(exc)


@pytest.mark.skipif(not ALEXA_ONNX.exists(), reason="firmware model dir absent")
class TestAlexaSurgery:
    def test_mechanically_clean_and_only_hybrid_ops_block_ort(self) -> None:
        out = make_portable(ALEXA_ONNX.read_bytes(), rewrite_hybrid=False)
        model = onnx.load_from_string(out)
        graph = model.graph

        # int64 shape positions: no int32 initializer feeds them anymore.
        assert _int32_shape_inputs(graph) == []

        # initializer names gone from graph.input; true inputs kept.
        init_names = {
            i.name for i in graph.initializer
            if i.raw_data or i.float_data or i.int32_data or i.int64_data
        }
        graph_inputs = {i.name for i in graph.input}
        assert graph_inputs & init_names == set()
        assert graph_inputs == EXPECTED_INPUTS

        # structurally valid per onnx.checker.  The stock model fails it
        # over the empty placeholder initializers and a spurious Shape
        # attribute; Hybrid* ops are undeclarable in the default domain, so
        # structural checking runs with them on a scratch custom domain
        # (which the checker skips) — everything else is fully validated.
        _checker_minus_hybrid(model)

        # ops otherwise untouched: same ops in the same order once the
        # inserted Cast nodes (the prescribed int64 casts) are ignored, and
        # Hybrid* nodes have identical inputs (their int32 zp/sums consts
        # stay int32).
        orig = onnx.load_from_string(ALEXA_ONNX.read_bytes())
        orig_ops = [n for n in orig.graph.node if n.op_type != "Cast"]
        new_ops = [n for n in graph.node if n.op_type != "Cast"]
        assert [(n.name, n.op_type) for n in new_ops] == \
            [(n.name, n.op_type) for n in orig_ops]
        for ours, theirs in zip(new_ops, orig_ops):
            if theirs.op_type.startswith("Hybrid"):
                assert ours.input == theirs.input, ours.name

        # ORT now rejects the graph only over the unknown Hybrid* ops.
        assert _load_and_classify(out) == "hybrid"

    def test_idempotent(self) -> None:
        once = make_portable(ALEXA_ONNX.read_bytes(), rewrite_hybrid=False)
        assert make_portable(once, rewrite_hybrid=False) == once


def _synthetic_model() -> bytes:
    init = [
        numpy_helper.from_array(np.array([3, 2], np.int32), "shape_i32"),
        numpy_helper.from_array(np.array([0], np.int32), "starts_i32"),
        numpy_helper.from_array(np.array([1], np.int32), "ends_i32"),
        numpy_helper.from_array(np.array([1], np.int32), "axes_i32"),
    ]
    x = helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [3, 2])
    z = helper.make_tensor_value_info("Z", onnx.TensorProto.FLOAT, [2, 1])
    nodes = [
        helper.make_node("Reshape", ["X", "shape_i32"], ["Y"], name="reshape"),
        helper.make_node("Slice", ["X", "starts_i32", "ends_i32", "axes_i32"],
                         ["Z"], name="slice"),
    ]
    # Exporter-style duplication: initializers re-declared as graph inputs.
    ins = [x] + [
        helper.make_tensor_value_info(i.name, onnx.TensorProto.INT32, list(i.dims))
        for i in init
    ]
    graph = helper.make_graph(nodes, "synthetic", ins, [y, z], init)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)]
                             ).SerializeToString()


def test_synthetic_graph_runs_after_surgery() -> None:
    out = make_portable(_synthetic_model())
    model = onnx.load_from_string(out)
    assert _int32_shape_inputs(model.graph) == []
    onnx.checker.check_model(model)
    assert {i.name for i in model.graph.input} == {"X"}

    # With no Hybrid ops present the surgered graph must actually load and
    # compute — proving the Cast rewiring preserves semantics.
    assert _load_and_classify(out) == "ok"
    sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    y, z = sess.run(None, {"X": x})
    np.testing.assert_array_equal(y, x.reshape(3, 2))
    np.testing.assert_array_equal(z, x[:, :1])
