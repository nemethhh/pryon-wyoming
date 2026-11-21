"""A synthetic Pryon-format model directory for tests (random weights, no Amazon bytes).

The graph keeps the real model's I/O contract and the exporter quirks
``make_portable`` fixes, at a tiny size: Hybrid* ops in the default domain,
int32 shape constants, ``Shape`` nodes with a ``dtype`` attribute, initializers
re-declared as graph inputs, and an empty placeholder initializer shadowing the
streaming conv buffer input.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto as T
from onnx import helper
from onnx import numpy_helper as nh

LANES = 25
MEL = 64
BUF_ROWS = 2

PRYON_CONFIG = """\
fe.type = "LFBE"
fe.audio_analysis.dither = 0.0
fe.audio_analysis.frame_length_milli = 25
fe.audio_analysis.frame_shift_milli = 10
fe.audio_analysis.preemphasis_coeff = 0.96875
fe.audio_analysis.remove_dc_offset = 1
fe.audio_analysis.round_to_power_of_two = 1
fe.audio_analysis.samplerate = 16000
fe.audio_analysis.window_type = "povey"
fe.cep.use_energy = 0
fe.mel_fbank.high_freq = 7200
fe.mel_fbank.low_freq = 80
fe.mel_fbank.num_bins = 64
scorer.nemort_config_json_filepath = "nemort_scorer_config.json"
keyword_spotter.config_filepath = "kw.cfg.json"
id2word.mapping_filepath = "words.shrunk.txt"
"""


class _Graph:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.nodes: list[onnx.NodeProto] = []
        self.inits: list[onnx.TensorProto] = []
        self.value_info: list[onnx.ValueInfoProto] = []
        self.n = 0

    def name(self, tag: str) -> str:
        self.n += 1
        return f"synth/{tag}_{self.n}"

    def init(self, arr, tag: str) -> str:
        name = self.name(tag)
        self.inits.append(nh.from_array(np.asarray(arr), name))
        return name

    def i32(self, values, tag: str = "i32") -> str:
        return self.init(np.array(values, np.int32), tag)

    def op(self, op_type: str, inputs, n_out: int = 1, **attrs) -> list[str]:
        outs = [self.name(op_type.lower()) for _ in range(n_out)]
        self.nodes.append(helper.make_node(op_type, list(inputs), outs,
                                           name=self.name(op_type), **attrs))
        return outs

    def shape_with_dtype(self, x: str) -> str:
        out = self.name("shape")
        node = helper.make_node("Shape", [x], [out], name=self.name("Shape"))
        node.attribute.append(helper.make_attribute("dtype", T.INT32))
        self.nodes.append(node)
        return out

    def quant_weights(self, n_out: int, k: int):
        w = self.rng.integers(-127, 128, (n_out, k), dtype=np.int8)
        sw = self.rng.uniform(0.002, 0.02, n_out).astype(np.float32)
        zw = self.rng.integers(-3, 4, n_out).astype(np.int32)
        sums = w.astype(np.int32).sum(axis=1)
        return w, sw, zw, sums

    def hybrid_conv(self, x: str, c_in: int, c_out: int, kernel, strides) -> str:
        w, sw, zw, sums = self.quant_weights(c_out, c_in * kernel[0] * kernel[1])
        bias = self.rng.normal(0, 0.1, c_out).astype(np.float32)
        return self.op("HybridConv", [x, self.init(w, "w"), self.init(sw, "sw"),
                                      self.init(zw, "zw"), self.init(sums, "sums"),
                                      self.init(bias, "b")],
                       data_format=b"NCHW", dilations=[1, 1], strides=list(strides),
                       kernel_shape=list(kernel), group=1)[0]

    def batchnorm(self, x: str, c: int) -> str:
        r = self.rng
        return self.op("BatchNormalization", [
            x, self.init(r.uniform(0.5, 1.5, c).astype(np.float32), "g"),
            self.init(r.normal(0, 0.1, c).astype(np.float32), "beta"),
            self.init(r.normal(0, 0.2, c).astype(np.float32), "mu"),
            self.init(r.uniform(0.5, 2.0, c).astype(np.float32), "var")],
            epsilon=0.001)[0]

    def hybrid_gemm(self, x: str, k: int, n_out: int) -> str:
        w, sw, zw, sums = self.quant_weights(n_out, k)
        return self.op("HybridGemm", [x, self.init(w, "w"), self.init(sw, "sw"),
                                      self.init(zw, "zw"), self.init(sums, "sums")])[0]

    def hybrid_lstm(self, x: str, n_in: int, hidden: int, h0: str, c0: str) -> list[str]:
        w, sw, zw, sums = self.quant_weights(4 * hidden, n_in)
        r, swr, zwr, sumsr = self.quant_weights(4 * hidden, hidden)
        b = self.rng.normal(0, 0.1, 8 * hidden).astype(np.float32)
        ins = [x, self.init(w[None], "W"), self.init(sw[None], "swW"),
               self.init(zw[None], "zwW"), self.init(sums[None], "sumsW"),
               self.init(r[None], "R"), self.init(swr[None], "swR"),
               self.init(zwr[None], "zwR"), self.init(sumsr[None], "sumsR"),
               self.init(b[None], "B"), "", h0, c0]
        return self.op("HybridLSTM", ins, n_out=3, direction=b"forward",
                       hidden_size=hidden, activations=[b"Sigmoid", b"Tanh", b"Tanh"])

    def model(self, inputs, outputs, *, duplicate_inits: bool = True) -> bytes:
        ins = list(inputs)
        if duplicate_inits:  # exporter style: every initializer is also a graph input
            declared = {i.name for i in ins}
            ins += [helper.make_tensor_value_info(i.name, i.data_type, list(i.dims))
                    for i in self.inits if i.name not in declared]
        graph = helper.make_graph(self.nodes, "synth", ins, outputs, self.inits,
                                  value_info=self.value_info)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
        model.ir_version = 6
        return model.SerializeToString()


def _vi(name, elem, shape):
    return helper.make_tensor_value_info(name, elem, shape)


def _head(g: _Graph, lstm_y: str, hidden: int, sizes: list[int]) -> str:
    """LSTM Y [T,1,25,h] -> per-lane dense stack -> [25,T,sizes[-1]] (pre-activation)."""
    y = g.op("Squeeze", [lstm_y], axes=[1])[0]                     # [T,25,h]
    y = g.op("Transpose", [y], perm=[1, 0, 2])[0]                  # [25,T,h]
    k = hidden
    for i, n_out in enumerate(sizes):
        rows = g.op("Reshape", [y, g.i32([-1, k])])[0]
        mm = g.hybrid_gemm(rows, k, n_out)
        y = g.op("Reshape", [mm, g.i32([LANES, -1, n_out])])[0]
        y = g.op("Add", [y, g.init(g.rng.normal(0, 0.1, n_out).astype(np.float32), "bias")])[0]
        if i < len(sizes) - 1:
            y = g.op("Relu", [y])[0]
        k = n_out
    return y


def build_graph(seed: int = 0, n_classes: int = 3, lstm_in: int = 12, hidden: int = 16) -> bytes:
    """The model ONNX: input_feats [1,T,64,1] (T a multiple of 6) -> ww softmax + se head."""
    g = _Graph(np.random.default_rng(seed))
    buf_in, buf_out = "streaming_conv_buffer_input_1", "streaming_conv_buffer_output_1"

    x = g.op("Sub", ["input_feats", g.init(np.full((1, MEL, 1), 8.0, np.float32), "mean")])[0]
    x = g.op("Mul", [x, g.init(np.full((1, MEL, 1), 0.25, np.float32), "invstd")])[0]
    x = g.op("Reshape", [x, g.i32([1, 1, -1, MEL])])[0]
    cat = g.op("Concat", [buf_in, x], axis=2)[0]                   # [1,1,T+2,64]
    g.value_info.append(_vi(cat, T.FLOAT, [1, 1, "T2", MEL]))
    rows = g.op("Slice", [g.shape_with_dtype(cat), g.i32([2]), g.i32([3]), g.i32([0])])[0]
    g.op("Slice", [cat, g.i32([-BUF_ROWS]), rows, g.i32([2])])
    g.nodes[-1].output[0] = buf_out

    c1 = 8
    x = g.hybrid_conv(cat, 1, c1, (3, 5), (1, 3))                  # [1,8,T,20]
    x = g.op("Relu", [x])[0]
    x = g.op("MaxPool", [x], kernel_shape=[6, 1], strides=[6, 1])[0]   # [1,8,T/6,20]
    x = g.batchnorm(x, c1)
    g.value_info.append(_vi(x, T.FLOAT, [1, c1, "T3", 20]))
    x = g.hybrid_conv(x, c1, lstm_in, (1, 20), (1, 1))            # [1,C,T/6,1]
    x = g.op("Relu", [x])[0]
    x = g.batchnorm(x, lstm_in)

    x = g.op("Transpose", [x], perm=[0, 2, 3, 1])[0]               # [1,T',1,C]
    shp = g.op("Cast", [g.shape_with_dtype(x)], to=T.INT32)[0]
    t_dim = g.op("Slice", [shp, g.i32([1]), g.i32([2]), g.i32([0])])[0]
    new_shape = g.op("Concat", [g.i32([1]), t_dim, g.i32([lstm_in])], axis=0)[0]
    new_shape = g.op("Cast", [new_shape], to=T.INT32)[0]
    x = g.op("Reshape", [x, new_shape])[0]                         # [1,T',C]
    x = g.op("Tile", [x, g.i32([LANES, 1, 1])])[0]                 # [25,T',C]
    x = g.op("Transpose", [x], perm=[1, 0, 2])[0]                  # [T',25,C]

    outputs = []
    for branch, sizes, final in (("ww", [hidden, n_classes], "softmax_output_ww"),
                                 ("se", [hidden, 2], "fc_output_se")):
        h0 = g.op("Unsqueeze", [f"input_h_{branch}"], axes=[0])[0]
        c0 = g.op("Unsqueeze", [f"input_c_{branch}"], axes=[0])[0]
        y, yh, yc = g.hybrid_lstm(x, lstm_in, hidden, h0, c0)
        g.op("Squeeze", [yh], axes=[0])
        g.nodes[-1].output[0] = f"lstm_{branch}_1"
        g.op("Squeeze", [yc], axes=[0])
        g.nodes[-1].output[0] = f"lstm_{branch}_1_1"
        pre = _head(g, y, hidden, sizes)
        if branch == "ww":
            g.op("Softmax", [pre], axis=2)
        else:
            g.op("Identity", [pre])
        g.nodes[-1].output[0] = final
        outputs += [_vi(final, T.FLOAT, [LANES, "T4", sizes[-1]]),
                    _vi(f"lstm_{branch}_1", T.FLOAT, [LANES, hidden]),
                    _vi(f"lstm_{branch}_1_1", T.FLOAT, [LANES, hidden])]
    outputs.append(_vi(buf_out, T.FLOAT, [1, 1, BUF_ROWS, MEL]))

    # exporter artifact: an empty initializer shadowing the conv-buffer input
    g.inits.append(helper.make_tensor(buf_in, T.FLOAT, [0], []))
    inputs = [_vi("input_feats", T.FLOAT, [1, "T", MEL, 1]),
              _vi(buf_in, T.FLOAT, [1, 1, BUF_ROWS, MEL])]
    inputs += [_vi(f"input_{s}_{b}", T.FLOAT, [LANES, hidden])
               for b in ("ww", "se") for s in ("h", "c")]
    return g.model(inputs, outputs)


def build_model_dir(dest: Path, *, seed: int = 0, keywords: tuple[str, ...] = ("HEY", "STOP"),
                    lstm_in: int = 12, hidden: int = 16) -> Path:
    """Write a complete synthetic model directory to ``dest`` and return it."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "model.onnx").write_bytes(
        build_graph(seed, n_classes=len(keywords) + 1, lstm_in=lstm_in, hidden=hidden))
    (dest / "pryon.manifest").write_text(json.dumps({
        "object_type_name": "ModelSetManifest", "object_type_version": 1,
        "default_model_name": "pryon",
        "models": {"pryon": {"config_file_path": "pryon.config"}}}, indent=2))
    (dest / "pryon.config").write_text(PRYON_CONFIG)
    (dest / "nemort_scorer_config.json").write_text(json.dumps({
        "model_filepath": "model.onnx",
        "model_component_mapping": {
            "input_map": [{"component_name": "input_feats"}],
            "output_map": [{"component_name": "softmax_output_ww"},
                           {"component_name": "fc_output_se"}]},
        "lstm_components": [
            {"input_component": f"input_{s}_{b}",
             "output_component": f"lstm_{b}_1" + ("_1" if s == "c" else "")}
            for b in ("ww", "se") for s in ("h", "c")],
        "recurrent_components_input": ["streaming_conv_buffer_input_1"],
        "recurrent_components_output": ["streaming_conv_buffer_output_1"]}, indent=2))
    spotters = []
    for i, name in enumerate(keywords):
        thresholds = {"accept-threshold": 0.9 if i == 0 else 0.55}
        if i == 0:
            thresholds["notify-threshold"] = 0.6
            thresholds["overrides"] = [{"accept-threshold": 0.8, "clientProperties": [
                {"equals": 1, "name": "AlarmState"}]}]
        spotters.append({
            "name": name,
            "classifier": {"type": "moving-average-max-pool", "trim-tail": 8, "window-size": 3},
            "spotter": {
                "classification-thresholds": thresholds,
                "word-component-array": [{"scorer-index": i + 1,
                                          "keyword-start-scorer-index": 0,
                                          "keyword-end-scorer-index": 1}],
                "keyword-feature-window-length": 100,
                "lock-period": 100}})
    (dest / "kw.cfg.json").write_text(json.dumps(
        {"keyword-orchestra": {"buffer-size": 200, "keyword-spotters": spotters}}, indent=2))
    (dest / "words.shrunk.txt").write_text(
        "".join(f"{w} {i}\n" for i, w in enumerate(("<eps>",) + tuple(keywords))))
    return dest


# -- single-op graphs for rewrite parity ------------------------------------------------

def _single(g: _Graph, inputs, outputs) -> bytes:
    return g.model(inputs, outputs, duplicate_inits=False)


def hybrid_gemm_graph(w, sw, zw, k: int) -> bytes:
    g = _Graph(np.random.default_rng(0))
    sums = w.astype(np.int32).sum(axis=1)
    y = g.op("HybridGemm", ["x", g.init(w, "w"), g.init(sw, "sw"), g.init(zw, "zw"),
                            g.init(sums, "sums")])[0]
    g.nodes[-1].output[0] = "y"
    return _single(g, [_vi("x", T.FLOAT, ["M", k])], [_vi("y", T.FLOAT, ["M", w.shape[0]])])


def hybrid_conv_graph(w, sw, zw, bias, kernel, strides, in_shape, dilations=(1, 1)) -> bytes:
    g = _Graph(np.random.default_rng(0))
    sums = w.reshape(w.shape[0], -1).astype(np.int32).sum(axis=1)
    g.op("HybridConv", ["x", g.init(w, "w"), g.init(sw, "sw"), g.init(zw, "zw"),
                        g.init(sums, "sums"), g.init(bias, "b")],
         data_format=b"NCHW", dilations=list(dilations), strides=list(strides),
         kernel_shape=list(kernel), group=1)
    g.nodes[-1].output[0] = "y"
    return _single(g, [_vi("x", T.FLOAT, list(in_shape))], [_vi("y", T.FLOAT, None)])


def hybrid_lstm_graph(w, sw_w, zw_w, r, sw_r, zw_r, b, n_in: int, hidden: int) -> bytes:
    g = _Graph(np.random.default_rng(0))
    sums_w = w.astype(np.int32).sum(axis=-1)
    sums_r = r.astype(np.int32).sum(axis=-1)
    ins = ["x", g.init(w, "W"), g.init(sw_w, "swW"), g.init(zw_w, "zwW"), g.init(sums_w, "sW"),
           g.init(r, "R"), g.init(sw_r, "swR"), g.init(zw_r, "zwR"), g.init(sums_r, "sR"),
           g.init(b, "B"), "", "h0", "c0"]
    g.op("HybridLSTM", ins, n_out=3, direction=b"forward", hidden_size=hidden)
    g.nodes[-1].output[:] = ["y", "yh", "yc"]
    return _single(g, [_vi("x", T.FLOAT, ["T", LANES, n_in]),
                       _vi("h0", T.FLOAT, [1, LANES, hidden]),
                       _vi("c0", T.FLOAT, [1, LANES, hidden])],
                   [_vi("y", T.FLOAT, ["T", 1, LANES, hidden]),
                    _vi("yh", T.FLOAT, [1, LANES, hidden]),
                    _vi("yc", T.FLOAT, [1, LANES, hidden])])


def batchnorm_graph(gamma, beta, mean, var, eps: float, shape) -> bytes:
    g = _Graph(np.random.default_rng(0))
    g.op("BatchNormalization", ["x", g.init(gamma, "g"), g.init(beta, "b"), g.init(mean, "m"),
                                g.init(var, "v")], epsilon=eps)
    g.nodes[-1].output[0] = "y"
    return _single(g, [_vi("x", T.FLOAT, list(shape))], [_vi("y", T.FLOAT, list(shape))])
