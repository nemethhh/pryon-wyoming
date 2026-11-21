"""The ported ALEXA graph loads in stock onnxruntime and matches the vendor NumPy
emulation (``pryon_quant_model.Model``) to <=1e-5 per output, cycle by cycle.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import onnxruntime as ort

from pypryon.graph import make_portable

from conftest import MODEL_DIR, REPO_ROOT, requires_model

ALEXA_ONNX = MODEL_DIR / "runtime_stateless_standard.streaming.fixed.int16.onnx" if MODEL_DIR else None

pytestmark = requires_model

# The 5 streaming conv buffer shapes as declared by the graph inputs.
CONV_BUFFERS = {
    "streaming_conv_buffer_input_7": (1, 1, 6, 64),
    "streaming_conv_buffer_input_15": (1, 96, 2, 20),
    "streaming_conv_buffer_input_23": (1, 128, 1, 9),
    "streaming_conv_buffer_input_30": (1, 128, 1, 6),
    "streaming_conv_buffer_input_37": (1, 160, 1, 4),
}
BUFFER_OUTPUTS = {k: k.replace("input", "output") for k in CONV_BUFFERS}


def _session() -> ort.InferenceSession:
    data = make_portable(ALEXA_ONNX.read_bytes())
    return ort.InferenceSession(data, providers=["CPUExecutionProvider"])


def _zero_states() -> dict[str, np.ndarray]:
    feeds = {
        "input_h_ww": np.zeros((25, 100), np.float32),
        "input_c_ww": np.zeros((25, 100), np.float32),
        "input_h_se": np.zeros((25, 100), np.float32),
        "input_c_se": np.zeros((25, 100), np.float32),
    }
    feeds.update({k: np.zeros(s, np.float32) for k, s in CONV_BUFFERS.items()})
    return feeds


def test_ort_loads_and_runs_25_zero_frames() -> None:
    sess = _session()
    feeds = _zero_states()
    feeds["input_feats"] = np.zeros((1, 25, 64, 1), np.float32)
    names = [o.name for o in sess.get_outputs()]
    out = dict(zip(names, sess.run(None, feeds), strict=True))

    # 25 input frames -> 4 LSTM steps through the 6:1 streaming conv stack
    # (conv1 T=6+N, maxpool /2, conv2 /3 -> N//6).
    assert out["softmax_output_ww"].shape == (25, 4, 3)
    assert out["fc_output_se"].shape == (25, 4, 2)
    for key in ("softmax_output_ww", "fc_output_se"):
        assert np.isfinite(out[key]).all(), key
    assert np.allclose(out["softmax_output_ww"].sum(-1), 1.0, atol=1e-5)

    # LSTM states come back in the input shapes.
    for key in ("lstm_ww_1", "lstm_ww_1_1", "lstm_se_1", "lstm_se_1_1"):
        assert out[key].shape == (25, 100), key
        assert np.isfinite(out[key]).all(), key
    # Streaming conv buffers roll with their declared shapes.
    for inp, outp in BUFFER_OUTPUTS.items():
        assert out[outp].shape == CONV_BUFFERS[inp], outp
        assert np.isfinite(out[outp]).all(), outp


def test_layer_parity_vs_vendor_emulation() -> None:
    os.environ["PRYON_MODEL"] = str(ALEXA_ONNX)
    sys.path.insert(0, str(REPO_ROOT / "vendor/pryon-native"))
    from pryon_quant_model import Model

    rng = np.random.default_rng(7)
    cycles = 12
    frames = rng.normal(2.0, 3.0, (cycles * 6, 64)).astype(np.float32)
    frames[18:24] = 0.0  # one silent cycle: log-floor / constant-ish rows

    vendor = Model(str(ALEXA_ONNX))
    stream = vendor.stream(frames)
    sess = _session()

    states = _zero_states()
    h_ww = np.zeros((25, 100), np.float32)
    c_ww = np.zeros((25, 100), np.float32)
    h_se = np.zeros((25, 100), np.float32)
    c_se = np.zeros((25, 100), np.float32)

    for t in range(cycles):
        feeds = dict(states)
        feeds["input_feats"] = frames[6 * t:6 * t + 6][None, :, :, None]
        # Host-side lane ring zeroing, exactly Model.stream's rule:
        # t == 0 zeroes all lanes, later cycles zero lane t % 25.
        h_in, c_in = h_ww.copy(), c_ww.copy()
        h_in[:] = 0 if t == 0 else h_in
        if t:
            h_in[t % 25] = 0
            c_in[t % 25] = 0
        feeds["input_h_ww"], feeds["input_c_ww"] = h_in, c_in
        h_in_se, c_in_se = h_se.copy(), c_se.copy()
        h_in_se[:] = 0 if t == 0 else h_in_se
        if t:
            h_in_se[t % 25] = 0
            c_in_se[t % 25] = 0
        feeds["input_h_se"], feeds["input_c_se"] = h_in_se, c_in_se

        names = [o.name for o in sess.get_outputs()]
        got = dict(zip(names, sess.run(None, feeds), strict=True))
        want = next(stream)

        np.testing.assert_allclose(
            got["softmax_output_ww"][:, 0, :], want["softmax_ww"],
            atol=1e-5, err_msg=f"softmax cycle {t}")
        np.testing.assert_allclose(
            got["fc_output_se"][:, 0, :], want["se"],
            atol=1e-5, err_msg=f"se cycle {t}")
        np.testing.assert_allclose(
            got["lstm_ww_1"], want["h_ww"], atol=1e-5, err_msg=f"h_ww cycle {t}")
        np.testing.assert_allclose(
            got["lstm_se_1"], want["h_se"], atol=1e-5, err_msg=f"h_se cycle {t}")

        h_ww, c_ww = got["lstm_ww_1"], got["lstm_ww_1_1"]
        h_se, c_se = got["lstm_se_1"], got["lstm_se_1_1"]
        states.update({k: got[v] for k, v in BUFFER_OUTPUTS.items()})
