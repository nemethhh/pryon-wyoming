"""Each Hybrid* rewrite in make_portable must match the vendored NumPy CPUNK reference."""
from __future__ import annotations

import sys
import types

import numpy as np
import onnxruntime as ort
import pytest

from conftest import REPO_ROOT
from pypryon.graph import make_portable
from pypryon.modelset import ModelError
from synthmodel import LANES, batchnorm_graph, hybrid_conv_graph, hybrid_gemm_graph, \
    hybrid_lstm_graph

sys.path.insert(0, str(REPO_ROOT / "vendor/pryon-native"))
from pryon_quant_model import Model, quant_matmul  # noqa: E402

RNG = np.random.default_rng(42)


def _run(model: bytes, feeds: dict) -> list[np.ndarray]:
    sess = ort.InferenceSession(make_portable(model), providers=["CPUExecutionProvider"])
    return sess.run(None, feeds)


def _weights(n_out: int, k: int):
    w = RNG.integers(-127, 128, (n_out, k), dtype=np.int8)
    sw = RNG.uniform(0.002, 0.02, n_out).astype(np.float32)
    zw = RNG.integers(-3, 4, n_out).astype(np.int32)
    return w, sw, zw


def test_gemm_matches_cpunk_reference() -> None:
    w, sw, zw = _weights(7, 10)
    x = RNG.normal(0, 2, (9, 10)).astype(np.float32)
    x[4] = 1.5  # constant row takes the exact-float fallback
    (got,) = _run(hybrid_gemm_graph(w, sw, zw, 10), {"x": x})
    want = quant_matmul(x, w, sw, zw, {"constant_rows": 0})
    np.testing.assert_allclose(got, want, atol=1e-6)


@pytest.mark.parametrize(("kernel", "strides", "in_shape"), [
    ((8, 5), (8, 3), (1, 1, 16, 64)),
    ((1, 20), (1, 1), (1, 3, 2, 20)),
])
def test_conv_matches_reference(kernel, strides, in_shape) -> None:
    c_in = in_shape[1]
    w, sw, zw = _weights(6, c_in * kernel[0] * kernel[1])
    w = w.reshape(6, c_in, *kernel)
    bias = RNG.normal(0, 0.1, 6).astype(np.float32)
    x = RNG.normal(0, 1, in_shape).astype(np.float32)
    (got,) = _run(hybrid_conv_graph(w, sw, zw, bias, kernel, strides, in_shape), {"x": x})
    ref = types.SimpleNamespace(init={"w": w, "sw": sw, "zw": zw, "b": bias})
    want = Model.conv(ref, x[0], ["x", "w", "sw", "zw", "sums", "b"],
                      {"kernel_shape": list(kernel), "strides": list(strides)},
                      {"constant_rows": 0})
    np.testing.assert_allclose(got[0], want, atol=1e-5)


def _lstm_reference(x, w, sw_w, zw_w, r, sw_r, zw_r, b, h, c):
    """Float64 cell math, gate order i,o,f,c, both bias halves added."""
    hid = r.shape[-1]
    stats = {"constant_rows": 0}
    ys = []
    for xt in x:
        g = (quant_matmul(xt, w, sw_w, zw_w, stats) + quant_matmul(h, r, sw_r, zw_r, stats)
             + b[:4 * hid].astype(np.float64) + b[4 * hid:].astype(np.float64))
        it, ot, ft, ct = np.split(g, 4, axis=1)
        sig = lambda v: 1 / (1 + np.exp(-v))  # noqa: E731
        c64 = sig(ft) * c + sig(it) * np.tanh(ct)
        h = (sig(ot) * np.tanh(c64)).astype(np.float32)
        c = c64.astype(np.float32)
        ys.append(h)
    return np.stack(ys), h, c


def test_lstm_matches_reference_input_ne_hidden() -> None:
    n_in, hid, steps = 12, 16, 3
    w, sw_w, zw_w = _weights(4 * hid, n_in)
    r, sw_r, zw_r = _weights(4 * hid, hid)
    b = RNG.normal(0, 0.1, 8 * hid).astype(np.float32)
    x = RNG.normal(0, 1, (steps, LANES, n_in)).astype(np.float32)
    h0 = RNG.normal(0, 0.5, (1, LANES, hid)).astype(np.float32)
    c0 = RNG.normal(0, 0.5, (1, LANES, hid)).astype(np.float32)
    model = hybrid_lstm_graph(w[None], sw_w[None], zw_w[None], r[None], sw_r[None], zw_r[None],
                              b[None], n_in, hid)
    y, yh, yc = _run(model, {"x": x, "h0": h0, "c0": c0})
    want_y, want_h, want_c = _lstm_reference(x, w, sw_w, zw_w, r, sw_r, zw_r, b, h0[0], c0[0])
    np.testing.assert_allclose(y[:, 0], want_y, atol=1e-5)
    np.testing.assert_allclose(yh[0], want_h, atol=1e-5)
    np.testing.assert_allclose(yc[0], want_c, atol=1e-5)


def test_batchnorm_matches_fused_epilogue() -> None:
    c = 5
    gamma, beta, mean = (RNG.normal(1, 0.2, c).astype(np.float32) for _ in range(3))
    var = RNG.uniform(0.5, 2, c).astype(np.float32)
    x = RNG.normal(0, 1, (1, c, 3, 4)).astype(np.float32)
    (got,) = _run(batchnorm_graph(gamma, beta, mean, var, 1e-3, x.shape), {"x": x})
    ref = types.SimpleNamespace(init={"g": gamma, "b": beta, "m": mean, "v": var})
    want = Model.bn(ref, x[0], ["x", "g", "b", "m", "v"], np.float32(1e-3))
    np.testing.assert_allclose(got[0], want, atol=1e-6)


def test_malformed_hybrid_conv_raises_model_error() -> None:
    w, sw, zw = _weights(4, 9)
    model = hybrid_conv_graph(w.reshape(4, 1, 3, 3), sw, zw, np.zeros(4, np.float32),
                              (3, 3), (1, 1), (1, 1, 8, 8), dilations=(2, 2))
    with pytest.raises(ModelError, match="dilations"):
        make_portable(model)
