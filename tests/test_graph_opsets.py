"""Graph rewrite on newer opsets (DAVS graphs are opset 15-22), biased HybridGemm, primed state."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto as T
from onnx import helper
from test_hybrid_ops import _lstm_reference

from pypryon.graph import initial_state, make_portable

from conftest import DAVS_MODEL_DIR, REPO_ROOT, requires_davs
from synthmodel import build_davs_model_dir, hybrid_conv_graph, hybrid_gemm_graph, hybrid_lstm_graph

sys.path.insert(0, str(REPO_ROOT / "vendor/pryon-native"))
from pryon_quant_model import Model, quant_matmul  # noqa: E402

RNG = np.random.default_rng(11)


def _session(model: bytes, **kwargs) -> ort.InferenceSession:
    return ort.InferenceSession(make_portable(model, **kwargs), providers=["CPUExecutionProvider"])


def _weights(n_out: int, k: int):
    return (RNG.integers(-127, 128, (n_out, k), dtype=np.int8),
            RNG.uniform(0.002, 0.02, n_out).astype(np.float32),
            RNG.integers(-3, 4, n_out).astype(np.int32))


@pytest.mark.parametrize("opset", [11, 13, 15, 18, 21])
def test_axes_form_matches_opset(opset) -> None:
    # every node in the attribute form older exporters (and the shipped encoder) use
    nodes = [helper.make_node("Unsqueeze", ["x"], ["u"], axes=[0]),
             helper.make_node("ReduceSum", ["u"], ["s"], axes=[1], keepdims=1),
             helper.make_node("ReduceMax", ["s"], ["m"], axes=[2], keepdims=0),
             helper.make_node("Squeeze", ["m"], ["y"], axes=[0])]
    graph = helper.make_graph(nodes, "axes", [helper.make_tensor_value_info("x", T.FLOAT, [3, 4])],
                              [helper.make_tensor_value_info("y", T.FLOAT, None)])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 8
    x = RNG.normal(size=(3, 4)).astype(np.float32)
    (y,) = _session(model.SerializeToString()).run(None, {"x": x})
    np.testing.assert_allclose(y, x.sum(axis=0, keepdims=True).max(axis=1), rtol=1e-6)


@pytest.mark.parametrize("trans_b", [1, 0])
def test_hybrid_gemm_bias_and_transb(trans_b) -> None:
    w, sw, zw = _weights(6, 10)
    bias = RNG.normal(0, 0.5, (1, 6)).astype(np.float32)
    x = RNG.normal(0, 2, (5, 10)).astype(np.float32)
    model = hybrid_gemm_graph(w, sw, zw, 10, bias=bias, trans_b=trans_b, opset=15)
    (got,) = _session(model).run(None, {"x": x})
    want = quant_matmul(x, w, sw, zw, {"constant_rows": 0}) + bias
    np.testing.assert_allclose(got, want, atol=1e-5)


@pytest.mark.parametrize("x_lanes", [16, 1])
def test_hybrid_lstm_opset15_16_lanes(x_lanes) -> None:
    n_in, hid, lanes = 24, 20, 16
    w, sw_w, zw_w = _weights(4 * hid, n_in)
    r, sw_r, zw_r = _weights(4 * hid, hid)
    b = RNG.normal(0, 0.1, 8 * hid).astype(np.float32)
    x = RNG.normal(0, 1, (1, x_lanes, n_in)).astype(np.float32)
    h0 = RNG.normal(0, 0.5, (1, lanes, hid)).astype(np.float32)
    c0 = RNG.normal(0, 0.5, (1, lanes, hid)).astype(np.float32)
    model = hybrid_lstm_graph(w[None], sw_w[None], zw_w[None], r[None], sw_r[None], zw_r[None], b[None],
                              n_in, hid, lanes=lanes, x_lanes=x_lanes, opset=15)
    y, yh, yc = _session(model).run(None, {"x": x, "h0": h0, "c0": c0})
    want_y, want_h, want_c = _lstm_reference(np.broadcast_to(x, (1, lanes, n_in)),
                                             w, sw_w, zw_w, r, sw_r, zw_r, b, h0[0], c0[0])
    np.testing.assert_allclose(y[:, 0], want_y, atol=1e-5)
    np.testing.assert_allclose(yh[0], want_h, atol=1e-5)
    np.testing.assert_allclose(yc[0], want_c, atol=1e-5)


def test_hybrid_conv_opset21() -> None:
    w, sw, zw = _weights(6, 3 * 4)
    w = w.reshape(6, 3, 4, 1)
    bias = RNG.normal(0, 0.1, 6).astype(np.float32)
    x = RNG.normal(0, 1, (1, 3, 4, 1)).astype(np.float32)
    (got,) = _session(hybrid_conv_graph(w, sw, zw, bias, (4, 1), (1, 1), x.shape, opset=21)).run(None, {"x": x})
    import types
    ref = types.SimpleNamespace(init={"w": w, "sw": sw, "zw": zw, "b": bias})
    want = Model.conv(ref, x[0], ["x", "w", "sw", "zw", "sums", "b"],
                      {"kernel_shape": [4, 1], "strides": [1, 1]}, {"constant_rows": 0})
    np.testing.assert_allclose(got[0], want, atol=1e-5)


@pytest.fixture(scope="module")
def davs_dir(tmp_path_factory) -> Path:
    return build_davs_model_dir(tmp_path_factory.mktemp("models") / "DAVS")


def test_primed_state_initial_values(davs_dir) -> None:
    raw = (davs_dir / "encoder/encoder.onnx").read_bytes()
    buf = "conv_buffer_input/Concat:2"
    sess = _session(raw, keep_inputs=[buf])
    assert buf in [i.name for i in sess.get_inputs()]
    stored = onnx.numpy_helper.to_array(next(i for i in onnx.load_from_string(raw).graph.initializer
                                              if i.name == buf))
    state = initial_state(raw, [buf])
    assert np.any(state[buf] != 0)
    np.testing.assert_array_equal(state[buf], stored)
    out = sess.run(None, {"input_136": np.zeros((1, 8, 64), np.float32), buf: state[buf]})
    assert all(np.isfinite(o).all() for o in out)


def test_duplicate_inputs_deduped(davs_dir) -> None:
    raw = (davs_dir / "multi-wakeword-wakeworddecoder/stop.onnx").read_bytes()
    sess = _session(raw, keep_inputs=["conv_buffer_input/Concat:3"])
    names = [i.name for i in sess.get_inputs()]
    assert sorted(names) == ["conv_buffer_input/Concat:3", "input_6"]


@requires_davs
def test_real_davs_graphs_load() -> None:
    from pypryon import load_model_set

    ms = load_model_set(DAVS_MODEL_DIR)
    for spec in [ms.encoder, *ms.decoders.values()]:
        raw = spec.path.read_bytes()
        sess = _session(raw, keep_inputs=spec.state_inputs)
        state = initial_state(raw, spec.state_inputs)
        feeds = {}
        for inp in sess.get_inputs():
            shape = [d if isinstance(d, int) else 1 for d in inp.shape]
            feeds[inp.name] = state.get(inp.name, np.zeros(shape, np.float32))
        for out in sess.run(None, feeds):
            assert np.isfinite(out).all(), spec.name


def test_hybrid_gemm_3d_input() -> None:
    w, sw, zw = _weights(2, 20)
    bias = RNG.normal(0, 0.5, (1, 2)).astype(np.float32)
    g = onnx.load_from_string(hybrid_gemm_graph(w, sw, zw, 20, bias=bias, trans_b=1, opset=15))
    for vi in (g.graph.input[0], g.graph.output[0]):
        vi.type.tensor_type.ClearField("shape")
    x = RNG.normal(0, 2, (1, 16, 20)).astype(np.float32)
    (got,) = _session(g.SerializeToString()).run(None, {"x": x})
    want = quant_matmul(x[0], w, sw, zw, {"constant_rows": 0}) + bias
    assert got.shape == (1, 16, 2)
    np.testing.assert_allclose(got[0], want, atol=1e-5)
