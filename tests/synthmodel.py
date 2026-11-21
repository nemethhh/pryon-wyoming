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

    def model(self, inputs, outputs, *, duplicate_inits: bool = True, opset: int = 11,
              ir: int = 6, extra_inputs=()) -> bytes:
        ins = list(inputs)
        if duplicate_inits:  # exporter style: every initializer is also a graph input
            declared = {i.name for i in ins}
            ins += [helper.make_tensor_value_info(i.name, i.data_type, list(i.dims))
                    for i in self.inits if i.name not in declared]
        graph = helper.make_graph(self.nodes, "synth", ins, outputs, self.inits,
                                  value_info=self.value_info)
        graph.input.extend(extra_inputs)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
        model.ir_version = ir
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
    # like the real graph: a no-op Tile, so the LSTM input broadcasts across lanes
    x = g.op("Tile", [x, g.i32([1, 1, 1])])[0]                     # [1,T',C]
    x = g.op("Transpose", [x], perm=[1, 0, 2])[0]                  # [T',1,C]

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

def _single(g: _Graph, inputs, outputs, opset: int = 11) -> bytes:
    return g.model(inputs, outputs, duplicate_inits=False, opset=opset, ir=6 if opset <= 11 else 8)


def hybrid_gemm_graph(w, sw, zw, k: int, *, bias=None, trans_b: int | None = None,
                      opset: int = 11) -> bytes:
    """``w`` is [N, K] (stored transposed to [K, N] when ``trans_b == 0``)."""
    g = _Graph(np.random.default_rng(0))
    sums = w.astype(np.int32).sum(axis=1)
    stored = w.T.copy() if trans_b == 0 else w
    ins = ["x", g.init(stored, "w"), g.init(sw, "sw"), g.init(zw, "zw"), g.init(sums, "sums")]
    if bias is not None:
        ins.append(g.init(bias, "bias"))
    attrs = {} if trans_b is None else {"transB": trans_b}
    g.op("HybridGemm", ins, **attrs)
    g.nodes[-1].output[0] = "y"
    return _single(g, [_vi("x", T.FLOAT, ["M", k])], [_vi("y", T.FLOAT, ["M", w.shape[0]])], opset)


def hybrid_conv_graph(w, sw, zw, bias, kernel, strides, in_shape, dilations=(1, 1),
                      opset: int = 11) -> bytes:
    g = _Graph(np.random.default_rng(0))
    sums = w.reshape(w.shape[0], -1).astype(np.int32).sum(axis=1)
    g.op("HybridConv", ["x", g.init(w, "w"), g.init(sw, "sw"), g.init(zw, "zw"),
                        g.init(sums, "sums"), g.init(bias, "b")],
         data_format=b"NCHW", dilations=list(dilations), strides=list(strides),
         kernel_shape=list(kernel), group=1)
    g.nodes[-1].output[0] = "y"
    return _single(g, [_vi("x", T.FLOAT, list(in_shape))], [_vi("y", T.FLOAT, None)], opset)


def hybrid_lstm_graph(w, sw_w, zw_w, r, sw_r, zw_r, b, n_in: int, hidden: int, *,
                      lanes: int = LANES, x_lanes: int | None = None, opset: int = 11) -> bytes:
    g = _Graph(np.random.default_rng(0))
    sums_w = w.astype(np.int32).sum(axis=-1)
    sums_r = r.astype(np.int32).sum(axis=-1)
    ins = ["x", g.init(w, "W"), g.init(sw_w, "swW"), g.init(zw_w, "zwW"), g.init(sums_w, "sW"),
           g.init(r, "R"), g.init(sw_r, "swR"), g.init(zw_r, "zwR"), g.init(sums_r, "sR"),
           g.init(b, "B"), "", "h0", "c0"]
    g.op("HybridLSTM", ins, n_out=3, direction=b"forward", hidden_size=hidden)
    g.nodes[-1].output[:] = ["y", "yh", "yc"]
    return _single(g, [_vi("x", T.FLOAT, ["T", x_lanes or lanes, n_in]),
                       _vi("h0", T.FLOAT, [1, lanes, hidden]),
                       _vi("c0", T.FLOAT, [1, lanes, hidden])],
                   [_vi("y", T.FLOAT, ["T", 1, lanes, hidden]),
                    _vi("yh", T.FLOAT, [1, lanes, hidden]),
                    _vi("yc", T.FLOAT, [1, lanes, hidden])], opset)


def batchnorm_graph(gamma, beta, mean, var, eps: float, shape) -> bytes:
    g = _Graph(np.random.default_rng(0))
    g.op("BatchNormalization", ["x", g.init(gamma, "g"), g.init(beta, "b"), g.init(mean, "m"),
                                g.init(var, "v")], epsilon=eps)
    g.nodes[-1].output[0] = "y"
    return _single(g, [_vi("x", T.FLOAT, list(shape))], [_vi("y", T.FLOAT, list(shape))])


# -- synthetic DAVS set (shared encoder + decoders), mirroring alexa-en-US's layout ---------

DAVS_LANES = 16
DAVS_EMB = 24


def _i64(g: _Graph, values, tag="i64") -> str:
    return g.init(np.array(values, np.int64), tag)


def _davs_encoder(seed: int) -> bytes:
    """[1,8,64] LFBE frames -> [1,1,E] embedding; one primed conv buffer, opset 15."""
    g = _Graph(np.random.default_rng(seed))
    buf_in, buf_out = "conv_buffer_input/Concat:2", "conv_buffer_output/Slice:2"
    x = g.op("Sub", ["input_136", g.init(np.full((1, MEL), 8.0, np.float32), "mean")])[0]
    x = g.op("Mul", [x, g.init(np.full((1, MEL), 0.25, np.float32), "invstd")])[0]
    x = g.op("Squeeze", [x, _i64(g, [0])])[0]                       # [8,64]
    cat = g.op("Concat", [buf_in, x], axis=0)[0]                    # [10,64]
    g.op("Slice", [cat, _i64(g, [8]), _i64(g, [10]), _i64(g, [0])])
    g.nodes[-1].output[0] = buf_out
    taps = []
    for start in (0, 2, 4, 6):                                      # stride-2 depthwise taps
        win = g.op("Slice", [cat, _i64(g, [start]), _i64(g, [start + 4]), _i64(g, [0])])[0]
        prod = g.op("Mul", [win, g.init(g.rng.uniform(0.5, 1.5, (4, MEL)).astype(np.float32), "B")])[0]
        # exporter artifact: axes as an attribute although opset 15 wants an input
        node = helper.make_node("ReduceSum", [prod], [g.name("rsum")], name=g.name("ReduceSum"),
                                axes=[0], keepdims=1)
        g.nodes.append(node)
        taps.append(node.output[0])
    rows = g.op("Concat", taps, axis=0)[0]                          # [4,64]
    hidden = 32
    for k_in, n_out in ((MEL, hidden), (hidden, DAVS_EMB)):
        w, sw, zw, sums = g.quant_weights(n_out, k_in)
        bias = g.rng.normal(0, 0.1, (1, n_out)).astype(np.float32)
        rows = g.op("HybridGemm", [rows, g.init(w, "w"), g.init(sw, "sw"), g.init(zw, "zw"),
                                   g.init(sums, "sums"), g.init(bias, "bias")], transB=1)[0]
        rows = g.op("Relu", [rows])[0]
    pooled = g.op("ReduceMax", [rows], axes=[0], keepdims=1)[0]     # [1,E]
    g.op("Reshape", [pooled, _i64(g, [1, 1, DAVS_EMB])])
    g.nodes[-1].output[0] = "encoder_embeddings"
    primed = nh.from_array(g.rng.normal(1.0, 0.5, (BUF_ROWS, MEL)).astype(np.float32), buf_in)
    g.inits.append(primed)
    return g.model([_vi("input_136", T.FLOAT, [1, 8, MEL])],
                   [_vi("encoder_embeddings", T.FLOAT, [1, 1, DAVS_EMB]),
                    _vi(buf_out, T.FLOAT, [BUF_ROWS, MEL])], opset=15, ir=8)


def _davs_alexa_decoder(seed: int, lanes_first: bool = False) -> bytes:
    """Embedding -> two 16-lane LSTM rings: det/vrf scores [1,16,3] and alignments [1,16,2]."""
    g = _Graph(np.random.default_rng(seed + 1))
    x = g.op("Tile", ["embedding", _i64(g, [1, DAVS_LANES, 1])])[0]    # [1,16,E] (T=1)
    outputs, extra = [], []
    for ring, hid, heads in (("scores", 20, ("ww_det_scores", "ww_vrf_scores")),
                             ("alignments", 8, ("alignments",))):
        h0 = g.op("Unsqueeze", [f"{ring}_h0", _i64(g, [0])])[0]
        c0 = g.op("Unsqueeze", [f"{ring}_c0", _i64(g, [0])])[0]
        y, yh, yc = g.hybrid_lstm(x, DAVS_EMB, hid, h0, c0)
        g.op("Squeeze", [yh, _i64(g, [0])])
        g.nodes[-1].output[0] = f"{ring}_hn"
        g.op("Squeeze", [yc, _i64(g, [0])])
        g.nodes[-1].output[0] = f"{ring}_cn"
        lanes = g.op("Squeeze", [y, _i64(g, [0, 1])])[0]           # [16,h]
        for head in heads:
            n_out = 2 if head == "alignments" else 3
            mm = g.hybrid_gemm(lanes, hid, n_out)
            mm = g.op("Add", [mm, g.init(g.rng.normal(0, 0.5, n_out).astype(np.float32), "b")])[0]
            sm = g.op("Softmax", [mm], axis=-1)[0]
            # alexa-de-DE style decoders emit [lanes, 1, classes]
            g.op("Unsqueeze", [sm, _i64(g, [1 if lanes_first else 0])])
            g.nodes[-1].output[0] = head
            shape = [DAVS_LANES, 1, n_out] if lanes_first else [1, DAVS_LANES, n_out]
            outputs.append(_vi(head, T.FLOAT, shape))
        outputs += [_vi(f"{ring}_hn", T.FLOAT, [DAVS_LANES, hid]),
                    _vi(f"{ring}_cn", T.FLOAT, [DAVS_LANES, hid])]
        extra += [_vi(f"{ring}_h0", T.FLOAT, [DAVS_LANES, hid]),
                  _vi(f"{ring}_c0", T.FLOAT, [DAVS_LANES, hid])]
    return g.model([_vi("embedding", T.FLOAT, [1, 1, DAVS_EMB]), *extra], outputs,
                   duplicate_inits=False, opset=15, ir=8)


def _davs_stop_decoder(seed: int) -> bytes:
    """Embedding -> conv over 4 steps -> mww/posteriors [1,1,4], mww/start_end [1,1,2]; opset 21."""
    g = _Graph(np.random.default_rng(seed + 2))
    buf_in, buf_out = "conv_buffer_input/Concat:3", "conv_buffer_output/Slice:3"
    x = g.op("Reshape", ["input_6", _i64(g, [1, DAVS_EMB, 1, 1])])[0]
    cat = g.op("Concat", [buf_in, x], axis=2)[0]                    # [1,E,4,1]
    g.op("Slice", [cat, _i64(g, [1]), _i64(g, [4]), _i64(g, [2])])
    g.nodes[-1].output[0] = buf_out
    g.value_info.append(_vi(cat, T.FLOAT, [1, DAVS_EMB, 4, 1]))
    conv = g.hybrid_conv(cat, DAVS_EMB, 10, (4, 1), (1, 1))         # [1,10,1,1]
    conv = g.op("Relu", [conv])[0]
    flat = g.op("Reshape", [conv, _i64(g, [1, 10])])[0]
    for name, n_out in (("mww/posteriors", 4), ("mww/start_end", 2)):
        w, sw, zw, sums = g.quant_weights(n_out, 10)
        bias = g.rng.normal(0, 0.5, (1, n_out)).astype(np.float32)
        mm = g.op("HybridGemm", [flat, g.init(w, "w"), g.init(sw, "sw"), g.init(zw, "zw"),
                                 g.init(sums, "sums"), g.init(bias, "bias")], transB=1)[0]
        sm = g.op("Softmax", [mm], axis=-1)[0]
        g.op("Unsqueeze", [sm, _i64(g, [0])])
        g.nodes[-1].output[0] = name
    g.inits.append(nh.from_array(np.zeros((1, DAVS_EMB, 3, 1), np.float32), buf_in))
    buf_vi = _vi(buf_in, T.FLOAT, [1, DAVS_EMB, 3, 1])
    return g.model([_vi("input_6", T.FLOAT, [1, 1, DAVS_EMB])],
                   [_vi("mww/posteriors", T.FLOAT, [1, 1, 4]), _vi("mww/start_end", T.FLOAT, [1, 1, 2]),
                    _vi(buf_out, T.FLOAT, [1, DAVS_EMB, 3, 1])],
                   opset=21, ir=10, extra_inputs=[buf_vi])  # duplicated input, as shipped


def _thresholds(accept, notify=None, overrides=(), **extra) -> dict:
    out = {"accept-threshold": accept, **extra}
    if notify is not None:
        out["notify-threshold"] = notify
    if overrides:
        out["overrides"] = [{"accept-threshold": a, "clientProperties": [{"equals": 1, "name": n}]}
                            for n, a in overrides]
    return out


def _component(index: int, scores: str, start_end: str, **extra) -> dict:
    return {"scorer-index": index,
            "scorer-output-component": {"name": scores, "posterior-type": "decoding-scores"},
            "keyword-start-scorer-index": 0, "keyword-end-scorer-index": 1,
            "keyword-start-end-component": {"name": start_end, "posterior-type": "time-alignment-scores"},
            **extra}


def _scorer(name: str, nemort: str, frames: int, feat: int, outputs: int, skip: int) -> dict:
    return {"name": name, "scorer.model_filetype": "nemort-base", "scorer.model_filepath": nemort,
            "scorer.initialize.frame_count": frames, "scorer.feat.dimensions": feat,
            "scorer.num_dnn_outputs": outputs, "scorer.score_upsampling.enable": 1,
            "scorer.score_upsampling.default_skip_frame_count": skip,
            "config_type": "scorer-config", "config_version": 1}


def build_davs_model_dir(dest: Path, *, seed: int = 0, lanes_first: bool = False) -> Path:
    """Write a synthetic DAVS set with alexa-en-US's file layout and return it."""
    dest = Path(dest)
    dirs = ["audiofrontendconfig", "encoder", "alexa-wakeworddecoder", "multi-wakeword-wakeworddecoder",
            "dvaddecoder", "watermarkconfig", "BDPGeneratedFiles"]
    for d in dirs:
        (dest / d).mkdir(parents=True, exist_ok=True)

    def write(rel: str, text: str) -> None:
        (dest / rel).write_text(text)

    def write_json(rel: str, obj) -> None:
        write(rel, json.dumps(obj, indent=2))

    write_json("pryon.manifest", {"object_type_name": "ModelSetManifest", "object_type_version": 1,
                                  "default_model_name": "pryon",
                                  "models": {"pryon": {"config_file_path": "pryon.config"}}})
    write("pryon.config", "".join(f'INCLUDE "{d}/pryon.config"\n' for d in dirs))
    fe = "\n".join(line for line in PRYON_CONFIG.splitlines() if line.startswith("fe."))
    write("audiofrontendconfig/pryon.config", fe + "\n")
    write("watermarkconfig/pryon.config", 'watermark.config_filepath = "watermark.cfg.json"\n')
    write("dvaddecoder/pryon.config", "# scorer listed by BDPGeneratedFiles/decoders.cfg.json\n")
    write("alexa-wakeworddecoder/pryon.config",
          "#scorer.multi.config_filepath_list += alexa_decoder.cfg.json Commented out by BDP\n")
    write("multi-wakeword-wakeworddecoder/pryon.config", "# merged into BDPGeneratedFiles\n")
    write("encoder/pryon.config", 'search.decoder_type = "kaldi-key-phrase"\n'
                                  '#keyword_spotter.config_filepath = "kw.cfg.json" Commented out\n'
                                  'scorer.single.config_filepath = "encoder.cfg.json"\n')
    write_json("encoder/encoder.cfg.json", {"config_type": "multi_scorer_config", "config_version": 1,
                                            "scorers": [_scorer("WW_encoder", "nemort_scorer_config_encoder.json",
                                                                8, MEL, DAVS_EMB, 7)]})
    write_json("encoder/nemort_scorer_config_encoder.json", {
        "object_type_name": "nemort_scorer_config", "file_format": "ONNX",
        "model_filepath": "encoder.onnx",
        "inputs": [{"input_name": "input_136", "input_source": "audio", "channel_type": "beamformed"}],
        "outputs": [{"component_name": "encoder_embeddings", "output_type": "decoding-scores",
                     "tensor_name": "encoder_output"}],
        "recurrent_components_input": ["conv_buffer_input/Concat:2"],
        "recurrent_components_output": ["conv_buffer_output/Slice:2"],
        "recurrent_state_initialization": False})
    (dest / "encoder/encoder.onnx").write_bytes(_davs_encoder(seed))
    write_json("alexa-wakeworddecoder/nemort_scorer_config_decoder_alexa.json", {
        "object_type_name": "nemort_scorer_config", "file_format": "ONNX", "model_filepath": "alexa.onnx",
        "inputs": [{"input_name": "embedding", "input_source": "tensor_artifact", "tensor_name": "encoder_output"}],
        "outputs": [{"component_name": c, "output_type": t} for c, t in
                    (("ww_det_scores", "decoding-scores"), ("ww_vrf_scores", "decoding-scores"),
                     ("alignments", "time-alignment-scores"))],
        "lstm_components": [{"input_component": f"{r}_{s}0", "output_component": f"{r}_{s}n"}
                            for r in ("scores", "alignments") for s in ("h", "c")],
        "recurrent_state_initialization": False})
    (dest / "alexa-wakeworddecoder/alexa.onnx").write_bytes(_davs_alexa_decoder(seed, lanes_first))
    write_json("multi-wakeword-wakeworddecoder/nemort_scorer_config_decoder_stop.json", {
        "object_type_name": "nemort_scorer_config", "file_format": "ONNX", "model_filepath": "stop.onnx",
        "inputs": [{"input_name": "input_6", "input_source": "tensor_artifact", "tensor_name": "encoder_output"}],
        "outputs": [{"component_name": "mww/posteriors", "output_type": "decoding-scores"},
                    {"component_name": "mww/start_end", "output_type": "time-alignment-scores"}],
        "recurrent_components_input": ["conv_buffer_input/Concat:3"],
        "recurrent_components_output": ["conv_buffer_output/Slice:3"],
        "recurrent_state_initialization": False})
    (dest / "multi-wakeword-wakeworddecoder/stop.onnx").write_bytes(_davs_stop_decoder(seed))
    write("BDPGeneratedFiles/pryon.config", "# generated by BDP\n"
          'id2word.mapping_filepath = "words.shrunk.txt"\n'
          'keyword_spotter.op_config_filepath = "op.cfg.json"\n'
          'scorer.multi.config_filepath = "decoders.cfg.json"\n'
          'keyword_spotter.config_filepath = "kw.cfg.json"\n')
    write_json("BDPGeneratedFiles/decoders.cfg.json", {"config_type": "multi_scorer_config", "config_version": 1,
        "scorers": [
            _scorer("dvad", "../dvaddecoder/missing_dvad.json", 1, DAVS_EMB, 2, 0),  # never loaded
            _scorer("ww_stop", "../multi-wakeword-wakeworddecoder/nemort_scorer_config_decoder_stop.json",
                    1, DAVS_EMB, 4, 0),
            _scorer("ww_alexa", "../alexa-wakeworddecoder/nemort_scorer_config_decoder_alexa.json",
                    1, DAVS_EMB, 3, 0)]})
    overrides = (("AlarmState", 0.67), ("AudioPlayerState", 0.86))
    spotters = [
        {"classifier": {"type": "exponential-decay", "decay-rate": 0.6}, "name": "STOP", "scorers": ["ww_stop"],
         "spotter": {"classification-thresholds": _thresholds(0.6),
                     "word-component-array": [_component(1, "mww/posteriors", "mww/start_end")],
                     "keyword-feature-window-length": 100, "lock-period": 100}},
        {"classifier": {"type": "weighted-geometric-average-aggregator", "weights": [1, 1]},
         "name": "HEY_DISNEY", "scorers": ["ww_stop"],
         "spotter": {"classification-thresholds": _thresholds(0.43, 0.2),
                     "word-component-array": [
                         _component(2, "mww/posteriors", "mww/start_end", **{"word-score-history-size": 100}),
                         _component(3, "mww/posteriors", "mww/start_end", **{"word-score-history-size": 1})],
                     "posterior-smoothers": [{"type": "exponential-decay", "decay-rate": 0.0},
                                             {"type": "exponential-decay", "decay-rate": 0.3}],
                     "keyword-feature-window-length": 100, "lock-period": 100}},
        {"classifier": {"type": "moving-average-max-pool", "window-size": 1}, "name": "ALEXA",
         "scorers": ["ww_alexa"],
         "spotter": {"classification-thresholds": _thresholds(
             0.921, 0.278, overrides, **{"min-accept-threshold": 0.845, "max-accept-threshold": 0.929,
                                         "probabilistic-accept-threshold": 0.527,
                                         "probabilistic-accept-probability": 0.001,
                                         "probabilistic-accept-lock-period": 50}),
                     "word-component-array": [_component(1, "ww_det_scores", "alignments")],
                     "keyword-feature-window-length": 100, "lock-period": 132}},
        {"classifier": {"type": "moving-average-max-pool", "window-size": 1}, "name": "ALEXA",
         "scorers": ["ww_alexa"],
         "spotter": {"spotter-type": "DETECTION_AUXILIARY",
                     "classification-thresholds": _thresholds(0.898, 0.28, overrides),
                     "word-component-array": [_component(1, "ww_vrf_scores", "alignments")],
                     "keyword-feature-window-length": 100, "lock-period": 100}}]
    write_json("BDPGeneratedFiles/kw.cfg.json", {"keyword-orchestra": {"buffer-size": 200,
                                                                       "keyword-spotters": spotters}})
    write("BDPGeneratedFiles/words.shrunk.txt", "<eps> 0\nALEXA 1\nHEY_DISNEY 2\nSTOP 3\n")
    write_json("BDPGeneratedFiles/op.cfg.json", {"initial": "sleep"})
    return dest
