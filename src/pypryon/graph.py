"""Make the shipped Pryon nemort ONNX graph run on stock onnxruntime.

Two layers:

* mechanical fixes for exporter artifacts ORT rejects: int32 constants in
  int64 shape positions, a spurious ``dtype`` attribute on ``Shape``,
  initializers re-declared as graph inputs, and empty placeholder
  initializers shadowing the streaming conv-buffer inputs;
* rewrites of the custom ``HybridConv`` / ``HybridGemm`` / ``HybridLSTM``
  kernels into standard ops implementing the measured CPUNK quantization
  contract (vendor LEDGER §CPUNK, ``pryon_quant_model.py``).
"""
from __future__ import annotations

import io

import numpy as np
import onnx
from onnx import TensorProto as T
from onnx import helper
from onnx import numpy_helper as _nh

from ._constants import LANES
from .modelset import ModelError

# every unregistered-op error from ORT names the Hybrid kernel family
ORT_LOAD_ERROR_HYBRID = "Hybrid"


# Ops whose int32 constant inputs participate in the shape pipeline.
# Values are input positions (None = all inputs).
_SHAPE_CONST_CONSUMERS: dict[str, tuple[int, ...] | None] = {
    "Concat": None,       # members must share a dtype
    "Reshape": (1,),      # shape input: int64 only
    "Tile": (1,),         # repeats: int64 only
    "Slice": (1, 2, 3),   # starts/ends/axes
    "Expand": (1,),
    "Pad": (1,),
}


def _has_data(init: onnx.TensorProto) -> bool:
    return bool(init.raw_data or init.float_data or init.int32_data
                or init.int64_data or init.double_data or init.uint64_data
                or init.string_data)


def _shape_cluster_casts(graph) -> dict[str, str]:
    """Map int32 initializer name -> cast output name for every constant
    consumed at a shape-pipeline position (Concat member, Reshape/Tile
    shape, Slice starts/ends/axes).  Computed tensors are already int64 —
    Shape nodes emit int64 and the exporter's Cast-to-int32 nodes are
    flipped to int64 — so casting these members makes every shape cluster
    uniformly int64.  Gather indices are int32-legal and stay untouched."""
    init_int32 = {
        i.name for i in graph.initializer
        if i.data_type == onnx.TensorProto.INT32 and _has_data(i)
    }
    casts: dict[str, str] = {}
    for node in graph.node:
        positions = _SHAPE_CONST_CONSUMERS.get(node.op_type)
        if positions is None and node.op_type in _SHAPE_CONST_CONSUMERS:
            positions = tuple(range(len(node.input)))  # Concat: every member
        for pos in positions or ():
            if pos < len(node.input) and node.input[pos] in init_int32:
                casts[node.input[pos]] = node.input[pos] + "__pypryon_i64"
    return casts


def make_portable(onnx_bytes: bytes, *, rewrite_hybrid: bool = True) -> bytes:
    """Mechanically portability-fix a Pryon nemort ONNX graph.  Idempotent.

    With ``rewrite_hybrid`` (default) the Hybrid* kernels are additionally
    rewritten into standard ops implementing the measured CPUNK contract
    (see ``_rewrite_hybrid_ops``); the result runs in stock onnxruntime.
    """
    model = onnx.load_from_string(onnx_bytes)
    graph = model.graph

    # 1. Drop empty placeholder initializers (TF-exporter artifacts for the
    #    externalized streaming conv buffers); their runtime-input
    #    declarations in graph.input stay.  Fixes onnx.checker.
    kept_inits = [i for i in graph.initializer if _has_data(i)]
    del graph.initializer[:]
    graph.initializer.extend(kept_inits)

    # 1b. Strip exporter-spurious attributes from Shape nodes (tf2onnx's
    #     `dtype` hint): opset-11 Shape has no attributes and outputs int64.
    for node in graph.node:
        if node.op_type == "Shape" and node.attribute:
            del node.attribute[:]

    # 1c. The exporter's whole shape pipeline runs int32 (Cast-to-int32
    #     nodes feeding Reshape/Slice/Concat shape positions).  Flip those
    #     Casts to int64 and drop the stale int32 value_info declarations;
    #     ORT re-infers the (now int64) shape tensors.
    for node in graph.node:
        if node.op_type == "Cast":
            to = None
            for at in node.attribute:
                if at.name == "to":
                    to = onnx.helper.get_attribute_value(at)
            if to == onnx.TensorProto.INT32:
                for at in node.attribute:
                    if at.name == "to":
                        at.i = onnx.TensorProto.INT64
    kept_vi = [v for v in graph.value_info
               if v.type.tensor_type.elem_type != onnx.TensorProto.INT32]
    del graph.value_info[:]
    graph.value_info.extend(kept_vi)

    # 2. Cast int32 shape-cluster constants to int64, rewiring consumers.
    casts = _shape_cluster_casts(graph)
    if casts:
        def rewire_positions(node: onnx.NodeProto) -> tuple[int, ...]:
            """Positions of this node that must see the int64 cast."""
            positions = _SHAPE_CONST_CONSUMERS.get(node.op_type)
            if positions is None and node.op_type in _SHAPE_CONST_CONSUMERS:
                return tuple(range(len(node.input)))  # Concat: every member
            return positions or ()

        new_nodes: list[onnx.NodeProto] = []
        emitted: set[str] = set()
        for node in graph.node:
            # Emit each Cast once, immediately before its first consumer —
            # constants have no dependencies, so this is always topologically
            # valid and keeps node order otherwise untouched.
            positions = rewire_positions(node)
            for pos in positions:
                if pos < len(node.input) and node.input[pos] in casts \
                        and node.input[pos] not in emitted:
                    new_nodes.append(
                        helper.make_node(
                            "Cast", [node.input[pos]], [casts[node.input[pos]]],
                            to=onnx.TensorProto.INT64,
                            name=casts[node.input[pos]],
                        )
                    )
                    emitted.add(node.input[pos])
            for pos in positions:
                if pos < len(node.input) and node.input[pos] in casts:
                    node.input[pos] = casts[node.input[pos]]
            new_nodes.append(node)
        del graph.node[:]
        graph.node.extend(new_nodes)

    # 3. Remove data-carrying initializer names from graph.input; keep the
    #    true runtime/state inputs.
    data_inits = {i.name for i in graph.initializer if _has_data(i)}
    kept = [i for i in graph.input if i.name not in data_inits]
    del graph.input[:]
    graph.input.extend(kept)

    # 4. Rewrite the Hybrid* kernels into standard ops (CPUNK contract).
    if rewrite_hybrid:
        _rewrite_hybrid_ops(model)
        _drop_unused_initializers(graph)

    out = io.BytesIO()
    onnx.save(model, out)
    return out.getvalue()




# ---------------------------------------------------------------------------
# Hybrid* rewrites. Each mirrors vendor pryon_quant_model.py op for op:
#   * activation rows quantize as s=(max-min)/127, inv=1/s, z=rint(-min*inv),
#     q=clip(rint(x*inv)+z, max=127)-z in float32 (do NOT simplify), the
#     accumulator q @ (w-zw)^T is exact in float64, output float32(acc)*float32(s*sw);
#   * constant rows (max == min) take the exact float product instead;
#   * HybridConv adds its bias in-kernel, HybridGemm does not;
#   * HybridLSTM: gates i,o,f,c, B=[Wb,Rb] both added, cell math in float64,
#     unrolled as a Loop over T; lane zeroing stays host-side (runner.py);
#   * BatchNormalization becomes the emulation's fused float64 epilogue.



class _Site:
    """Name-unique node/initializer emitter for one rewrite site.

    Initializers are buffered in ``inits`` so a Loop body's constants can be
    attached to the body graph (subgraphs do not reliably see outer-scope
    initializers) while outer ones go to the main graph.
    """

    def __init__(self, prefix: str):
        self.prefix = prefix
        self.n = 0
        self.inits: list = []

    def t(self, tag: str) -> str:
        self.n += 1
        return f"{self.prefix}__x{self.n:03d}_{tag}"

    def node(self, op: str, inputs, outputs, **attrs):
        return onnx.helper.make_node(op, inputs, outputs,
                                     name=self.t(op.lower()), **attrs)

    def init(self, arr: np.ndarray, tag: str) -> str:
        name = self.t("i_" + tag)
        self.inits.append(_nh.from_array(np.asarray(arr), name))
        return name


def _quant_matmul(site, x: str, w: np.ndarray, sw: np.ndarray,
                  zw: np.ndarray, nodes: list) -> str:
    """Emit the CPUNK quantized matmul of rows x [M,K] against int8 w [N,K].

    Appends nodes to ``nodes``; returns the output tensor name (float32
    [M,N]).  Mirrors pryon_quant_model.quant_matmul exactly.
    """
    w2 = w.reshape(w.shape[0], -1) if w.ndim > 2 else w
    sw = np.asarray(sw).ravel().astype(np.float32)
    zw = np.asarray(zw).ravel()
    n_out, k = w2.shape
    # (w - zw)^T and ((w - zw) * sw)^T, float64 — exact, build-time consts.
    wd_t = (w2.astype(np.float64) - zw[:, None].astype(np.float64)).T.copy()
    wde_t = wd_t * sw.astype(np.float64)[None, :]

    i_wdt = site.init(wd_t, "wdt")
    i_wde = site.init(wde_t, "wde")
    i_sw32 = site.init(sw.reshape(1, n_out), "sw32")
    i_127 = site.init(np.array(127.0, np.float32), "c127")
    i_1 = site.init(np.array(1.0, np.float32), "c1f")
    i_0 = site.init(np.zeros(1, np.float32), "c0f")

    lo, hi = site.t("lo"), site.t("hi")
    nodes.append(site.node("ReduceMin", [x], [lo], axes=[1], keepdims=0))
    nodes.append(site.node("ReduceMax", [x], [hi], axes=[1], keepdims=0))
    rng = site.t("rng")
    nodes.append(site.node("Sub", [hi, lo], [rng]))            # hi-lo f32
    s_raw = site.t("sraw")
    nodes.append(site.node("Div", [rng, i_127], [s_raw]))      # (hi-lo)/127
    mask = site.t("cmask")
    nodes.append(site.node("Equal", [rng, i_0], [mask]))
    s = site.t("s")
    nodes.append(site.node("Where", [mask, i_1, s_raw], [s]))
    inv = site.t("inv")
    nodes.append(site.node("Reciprocal", [s], [inv]))          # 1/s f32
    s_col, inv_col = site.t("scol"), site.t("invcol")
    nodes.append(site.node("Unsqueeze", [s], [s_col], axes=[1]))
    nodes.append(site.node("Unsqueeze", [inv], [inv_col], axes=[1]))
    lo_col = site.t("locol")
    nodes.append(site.node("Unsqueeze", [lo], [lo_col], axes=[1]))
    neg_lo = site.t("neglo")
    nodes.append(site.node("Neg", [lo_col], [neg_lo]))
    z_pre = site.t("zpre")
    nodes.append(site.node("Mul", [neg_lo, inv_col], [z_pre]))  # -lo*inv f32
    z = site.t("z")
    nodes.append(site.node("Round", [z_pre], [z]))              # rint → [M,1]
    xs = site.t("xs")
    nodes.append(site.node("Mul", [x, inv_col], [xs]))          # x*inv
    qr = site.t("qr")
    nodes.append(site.node("Round", [xs], [qr]))
    qz = site.t("qz")
    nodes.append(site.node("Add", [qr, z], [qz]))
    qc = site.t("qc")
    nodes.append(site.node("Clip", [qz, "", i_127], [qc]))      # max=127 only (min omitted)
    q = site.t("q")
    nodes.append(site.node("Sub", [qc, z], [q]))                # int-valued
    q64 = site.t("q64")
    nodes.append(site.node("Cast", [q], [q64], to=T.DOUBLE))
    acc = site.t("acc")
    nodes.append(site.node("MatMul", [q64, i_wdt], [acc]))      # f64 exact
    acc32 = site.t("acc32")
    nodes.append(site.node("Cast", [acc], [acc32], to=T.FLOAT))
    scale = site.t("scale")
    nodes.append(site.node("Mul", [s_col, i_sw32], [scale]))    # s*sw f32
    y_q = site.t("yq")
    nodes.append(site.node("Mul", [acc32, scale], [y_q]))
    x64 = site.t("x64")
    nodes.append(site.node("Cast", [x], [x64], to=T.DOUBLE))
    ex64 = site.t("ex64")
    nodes.append(site.node("MatMul", [x64, i_wde], [ex64]))     # fallback
    exact = site.t("exact")
    nodes.append(site.node("Cast", [ex64], [exact], to=T.FLOAT))
    mask_col = site.t("cmaskcol")
    nodes.append(site.node("Unsqueeze", [mask], [mask_col], axes=[1]))
    y = site.t("y")
    nodes.append(site.node("Where", [mask_col, exact, y_q], [y]))
    return y


def _attr(node, name, default=None):
    for a in node.attribute:
        if a.name == name:
            return onnx.helper.get_attribute_value(a)
    return default


def _init_arrays(graph) -> dict:
    return {i.name: _nh.to_array(i) for i in graph.initializer if _has_data(i)}


def _rewrite_hybrid_conv(graph, node, arrays) -> list:
    """HybridConv → dynamic-T im2col + CPUNK matmul + in-kernel bias."""
    a = {at.name: onnx.helper.get_attribute_value(at) for at in node.attribute}
    kh, kw = a["kernel_shape"][-2:]
    sh, sf = a.get("strides", [1, 1])[-2:]
    if list(a.get("dilations", [1, 1])) != [1, 1]:
        raise ModelError(f"{node.name}: HybridConv dilations {list(a['dilations'])} unsupported")
    if a.get("data_format", b"NCHW") != b"NCHW":
        raise ModelError(f"{node.name}: HybridConv data_format {a['data_format']!r} unsupported")
    site = _Site(node.name + "__port")
    x = node.input[0]
    w, sw, zw, bias = (arrays[node.input[i]] for i in (1, 2, 3, 5))
    w2 = w.reshape(w.shape[0], -1)
    n_out = w2.shape[0]
    k_dim = w2.shape[1]                      # C*kh*kw
    c_in = k_dim // (kh * kw)
    f_dim = arrays[node.input[0]].shape[3] if node.input[0] in arrays else None
    f_dim = _static_dim(graph, x, 3, f_dim)
    fo = (f_dim - kw) // sf + 1

    nd = []
    # Dynamic row count along T: E = sh*((T-kh)//sh)+1 = last needed index+1.
    shape4 = site.t("shape")
    nd.append(site.node("Shape", [x], [shape4]))
    t_dim = site.t("tdim")
    nd.append(site.node("Slice", [shape4, site.init(np.array([2], np.int64), "a2"),
                                  site.init(np.array([3], np.int64), "a3"),
                                  site.init(np.array([0], np.int64), "a0")], [t_dim]))
    sub = site.t("tsub")
    nd.append(site.node("Sub", [t_dim, site.init(np.array([kh], np.int64), "kh")], [sub]))
    div = site.t("tdiv")
    nd.append(site.node("Div", [div0 := sub, site.init(np.array([sh], np.int64), "sh")], [div]))
    mul = site.t("tmul")
    nd.append(site.node("Mul", [div, site.init(np.array([sh], np.int64), "sh2")], [mul]))
    e_end = site.t("eend")
    nd.append(site.node("Add", [mul, site.init(np.array([1], np.int64), "one")], [e_end]))

    # T-axis strided slices (one per kernel row) → concat axis 2.
    ax2 = site.init(np.array([2], np.int64), "ax2")
    t_slices = []
    for ki in range(kh):
        ends = site.t(f"endki{ki}")
        nd.append(site.node("Add", [e_end, site.init(np.array([ki], np.int64), f"ki{ki}")], [ends]))
        s = site.t(f"tki{ki}")
        nd.append(site.node("Slice", [x, site.init(np.array([ki], np.int64), f"sk{ki}"),
                                      ends, ax2, site.init(np.array([sh], np.int64), f"st{ki}")], [s]))
        t_slices.append(s)
    cat_t = site.t("catT")
    nd.append(site.node("Concat", t_slices, [cat_t], axis=2))     # [1,C,kh*to,F]

    # F-axis strided slices (static) → concat axis 3.
    ax3 = site.init(np.array([3], np.int64), "ax3")
    f_slices = []
    for kj in range(kw):
        s0, s1 = kj, kj + sf * (fo - 1) + 1
        s = site.t(f"fkj{kj}")
        nd.append(site.node("Slice", [cat_t,
                                      site.init(np.array([s0], np.int64), f"fs{kj}"),
                                      site.init(np.array([s1], np.int64), f"fe{kj}"),
                                      ax3, site.init(np.array([sf], np.int64), f"fst{kj}")], [s]))
        f_slices.append(s)
    cat_f = site.t("catF")
    nd.append(site.node("Concat", f_slices, [cat_f], axis=3))     # [1,C,kh*to,kw*fo]

    # [1,C,kh,to,kw,fo] → [1,to,fo,C,kh,kw] → [to*fo, K]
    shape6 = site.init(np.array([1, c_in, kh, -1, kw, fo], np.int64), "shape6")
    r6 = site.t("r6")
    nd.append(site.node("Reshape", [cat_f, shape6], [r6]))
    tr6 = site.t("tr6")
    nd.append(site.node("Transpose", [r6], [tr6], perm=[0, 3, 5, 1, 2, 4]))
    shape2 = site.init(np.array([-1, k_dim], np.int64), "shape2")
    rows = site.t("rows")
    nd.append(site.node("Reshape", [tr6, shape2], [rows]))

    y = _quant_matmul(site, rows, w, sw, zw, nd)          # [to*fo, N]
    yb = site.t("yb")
    nd.append(site.node("Add", [y, node.input[5]], [yb]))        # in-kernel bias
    shape3 = site.init(np.array([-1, fo, n_out], np.int64), "shape3")
    y3 = site.t("y3")
    nd.append(site.node("Reshape", [yb, shape3], [y3]))          # [to,fo,N]
    tr3 = site.t("tr3")
    nd.append(site.node("Transpose", [y3], [tr3], perm=[2, 0, 1]))   # [N,to,fo]
    nd.append(site.node("Unsqueeze", [tr3], [node.output[0]], axes=[0]))
    return nd, site.inits


def _rewrite_hybrid_gemm(graph, node, arrays) -> list:
    """HybridGemm → CPUNK matmul (no bias; downstream Add carries it)."""
    site = _Site(node.name + "__port")
    nd = []
    y = _quant_matmul(site, node.input[0],
                      arrays[node.input[1]], arrays[node.input[2]],
                      arrays[node.input[3]], nd)
    nd.append(site.node("Identity", [y], [node.output[0]]))
    return nd, site.inits


def _rewrite_hybrid_lstm(graph, node, arrays) -> list:
    """HybridLSTM → Loop over T carrying h/c, one CPUNK cell per step."""
    site = _Site(node.name + "__port")
    nd = []
    x = node.input[0]                     # [T, 1 or lanes, n_in]; 1 broadcasts
    w, sw_w, zw_w = (arrays[node.input[i]] for i in (1, 2, 3))
    r, sw_r, zw_r = (arrays[node.input[i]] for i in (5, 6, 7))
    b = arrays[node.input[9]].ravel()     # [800] = [Wb, Rb]
    hid = r.shape[-1]
    n_in = w.shape[-1]

    # max_trip_count = Shape(x)[0] as scalar
    shp = site.t("shp")
    nd.append(site.node("Shape", [x], [shp]))
    seq = site.t("seq")
    nd.append(site.node("Slice", [shp, site.init(np.array([0], np.int64), "z0"),
                                  site.init(np.array([1], np.int64), "o1"),
                                  site.init(np.array([0], np.int64), "a0")], [seq]))
    seqs = site.t("seqs")
    nd.append(site.node("Squeeze", [seq], [seqs], axes=[0]))
    true_c = site.init(np.array(True), "true")
    h0 = site.t("h0")
    nd.append(site.node("Squeeze", [node.input[11]], [h0], axes=[0]))   # [lanes, hid]
    c0 = site.t("c0")
    nd.append(site.node("Squeeze", [node.input[12]], [c0], axes=[0]))

    # ---- Loop body: one LSTM step, cell math in float64 (Model.lstm_cell).
    body = _Site(node.name + "__body")
    bn: list = []
    vi = lambda n, t_, s: onnx.helper.make_tensor_value_info(n, t_, s)
    b_iter = vi("_iter", T.INT64, [])
    b_cond = vi("_cond", T.BOOL, [])
    b_h = vi("_h", T.FLOAT, [LANES, hid])
    b_c = vi("_c", T.FLOAT, [LANES, hid])
    b_x = vi("_x", T.FLOAT, [None, None, n_in])
    b_ocond = vi("_ocond", T.BOOL, [])
    b_oh = vi("_oh", T.FLOAT, [LANES, hid])
    b_oc = vi("_oc", T.FLOAT, [LANES, hid])
    b_ox = vi("_ox", T.FLOAT, [None, None, n_in])
    b_oy = vi("_oy", T.FLOAT, [1, LANES, hid])

    it_c = body.t("iter")
    bn.append(body.node("Unsqueeze", ["_iter"], [it_c], axes=[0]))
    it1 = body.t("iter1")
    bn.append(body.node("Add", [it_c, body.init(np.array([1], np.int64), "one")], [it1]))
    xt1 = body.t("xt1")
    bn.append(body.node("Slice", ["_x", it_c, it1,
                                  body.init(np.array([0], np.int64), "a0")], [xt1]))
    xt = body.t("xt")
    bn.append(body.node("Squeeze", [xt1], [xt], axes=[0]))            # [lanes, hid]
    gx = _quant_matmul(body, xt, w[0], sw_w[0], zw_w[0], bn)          # [lanes, 4*hid]
    gh = _quant_matmul(body, "_h", r[0], sw_r[0], zw_r[0], bn)
    g32 = body.t("g32")
    bn.append(body.node("Add", [gx, gh], [g32]))
    g64 = body.t("g64")
    bn.append(body.node("Cast", [g32], [g64], to=T.DOUBLE))
    i_wb = body.init(b[:4 * hid].astype(np.float64), "wb")
    i_rb = body.init(b[4 * hid:].astype(np.float64), "rb")
    gw = body.t("gw")
    bn.append(body.node("Add", [g64, i_wb], [gw]))
    g = body.t("g")
    bn.append(body.node("Add", [gw, i_rb], [g]))                      # f64 [lanes, 4*hid]
    sl = lambda tag, s0: (body.init(np.array([s0], np.int64), f"s{tag}0"),
                          body.init(np.array([s0 + hid], np.int64), f"s{tag}1"),
                          body.init(np.array([1], np.int64), f"a1{tag}"))
    gi = body.t("gi"); a, e, ax = sl("i", 0)
    bn.append(body.node("Slice", [g, a, e, ax], [gi]))
    go = body.t("go"); a, e, ax = sl("o", hid)
    bn.append(body.node("Slice", [g, a, e, ax], [go]))
    gf = body.t("gf"); a, e, ax = sl("f", 2 * hid)
    bn.append(body.node("Slice", [g, a, e, ax], [gf]))
    gc = body.t("gc"); a, e, ax = sl("c", 3 * hid)
    bn.append(body.node("Slice", [g, a, e, ax], [gc]))
    sig_i, sig_o, sig_f = body.t("si"), body.t("so"), body.t("sf")
    bn.append(body.node("Sigmoid", [gi], [sig_i]))
    bn.append(body.node("Sigmoid", [go], [sig_o]))
    bn.append(body.node("Sigmoid", [gf], [sig_f]))
    tanh_gc = body.t("tc")
    bn.append(body.node("Tanh", [gc], [tanh_gc]))
    c64 = body.t("cin64")
    bn.append(body.node("Cast", ["_c"], [c64], to=T.DOUBLE))
    fc = body.t("fc")
    bn.append(body.node("Mul", [sig_f, c64], [fc]))
    ic = body.t("ic")
    bn.append(body.node("Mul", [sig_i, tanh_gc], [ic]))
    c2 = body.t("c2")
    bn.append(body.node("Add", [fc, ic], [c2]))                       # f64
    tanh_c2 = body.t("tc2")
    bn.append(body.node("Tanh", [c2], [tanh_c2]))
    h2_64 = body.t("h2d")
    bn.append(body.node("Mul", [sig_o, tanh_c2], [h2_64]))
    b_oh_actual = body.t("hout")
    bn.append(body.node("Cast", [h2_64], [b_oh_actual], to=T.FLOAT))
    b_oc_actual = body.t("cout")
    bn.append(body.node("Cast", [c2], [b_oc_actual], to=T.FLOAT))
    y_scan = body.t("yscan")
    bn.append(body.node("Unsqueeze", [b_oh_actual], [y_scan], axes=[0]))
    b_x_pass = body.t("xpass")
    bn.append(body.node("Identity", ["_x"], [b_x_pass]))

    body_graph = onnx.helper.make_graph(
        bn, node.name + "__bodygraph",
        [b_iter, b_cond, b_h, b_c, b_x],
        [onnx.helper.make_tensor_value_info("_ocond_o", T.BOOL, []),
         onnx.helper.make_tensor_value_info(b_oh_actual, T.FLOAT, [LANES, hid]),
         onnx.helper.make_tensor_value_info(b_oc_actual, T.FLOAT, [LANES, hid]),
         onnx.helper.make_tensor_value_info(b_x_pass, T.FLOAT, [None, None, n_in]),
         onnx.helper.make_tensor_value_info(y_scan, T.FLOAT, [1, LANES, hid])],
        [])
    body_graph.initializer.extend(body.inits)
    body_graph.node.insert(0, onnx.helper.make_node(
        "Identity", ["_cond"], ["_ocond_o"], name="_ocond_pass"))

    h_f, c_f, x_f, y_all = site.t("hf"), site.t("cf"), site.t("xf"), site.t("yall")
    nd.append(site.node("Loop", [seqs, true_c, h0, c0, x],
                        [h_f, c_f, x_f, y_all], body=body_graph))
    nd.append(site.node("Unsqueeze", [h_f], [node.output[1]], axes=[0]))
    nd.append(site.node("Unsqueeze", [c_f], [node.output[2]], axes=[0]))
    # Y: Loop scanned output stacks body outputs along a new leading axis:
    # stacked body outputs [T,1,lanes,hid] are exactly the ONNX LSTM Y layout
    nd.append(site.node("Identity", [y_all], [node.output[0]]))
    return nd, site.inits


def _rewrite_batchnorm(graph, node, arrays) -> list:
    """BatchNormalization → Model.bn's fused epilogue (f64 mul+add, one
    rounding): scale = g/sqrt(var+eps), shift = b - mu*scale, both float32
    as in the emulation, applied as f32(f64(x)*scale + shift)."""
    g_, b_, mu, var = (arrays[node.input[i]].astype(np.float32) for i in (1, 2, 3, 4))
    eps = np.float32(_attr(node, "epsilon"))
    scale = (g_ / np.sqrt(var + eps)).astype(np.float32)          # float32 math
    shift = (b_ - mu * scale).astype(np.float32)
    c = scale.size
    site = _Site(node.name + "__bn")
    i_sc = site.init(scale.reshape(1, c, 1, 1).astype(np.float64), "sc")
    i_sh = site.init(shift.reshape(1, c, 1, 1).astype(np.float64), "sh")
    nd = []
    x64 = site.t("x64")
    nd.append(site.node("Cast", [node.input[0]], [x64], to=T.DOUBLE))
    m = site.t("m")
    nd.append(site.node("Mul", [x64, i_sc], [m]))
    a = site.t("a")
    nd.append(site.node("Add", [m, i_sh], [a]))
    nd.append(site.node("Cast", [a], [node.output[0]], to=T.FLOAT))
    return nd, site.inits


def _static_dim(graph, tensor: str, axis: int, fallback=None) -> int:
    for v in list(graph.value_info) + list(graph.input):
        if v.name == tensor:
            d = v.type.tensor_type.shape.dim[axis]
            if d.HasField("dim_value"):
                return d.dim_value
    if fallback is not None:
        return fallback
    raise ValueError(f"no static dim {axis} for {tensor}")


def _rewrite_hybrid_ops(model) -> None:
    graph = model.graph
    arrays = _init_arrays(graph)
    new_nodes: list = []
    for node in graph.node:
        if node.op_type == "HybridConv":
            nodes_, inits_ = _rewrite_hybrid_conv(graph, node, arrays)
        elif node.op_type == "HybridGemm":
            nodes_, inits_ = _rewrite_hybrid_gemm(graph, node, arrays)
        elif node.op_type == "HybridLSTM":
            nodes_, inits_ = _rewrite_hybrid_lstm(graph, node, arrays)
        elif node.op_type == "BatchNormalization":
            nodes_, inits_ = _rewrite_batchnorm(graph, node, arrays)
        else:
            new_nodes.append(node)
            continue
        new_nodes.extend(nodes_)
        graph.initializer.extend(inits_)
    del graph.node[:]
    graph.node.extend(new_nodes)


def _drop_unused_initializers(graph) -> None:
    """The rewritten kernels' original weights would make ORT log a warning per tensor."""
    used = {i for n in graph.node for i in n.input}
    used |= {o.name for o in graph.output}
    kept = [i for i in graph.initializer if i.name in used]
    del graph.initializer[:]
    graph.initializer.extend(kept)
