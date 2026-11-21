"""Mechanical portability surgery on the Pryon nemort ONNX graph.

The shipped graph (``runtime_stateless_standard.streaming.fixed.int16.onnx``,
opset 11 / IR 6) is rejected by stock onnxruntime for mechanical reasons
only, ahead of its unknown Hybrid* kernels:

* int32 constants feed shape positions that opset-11 requires to be int64
  (``Reshape[1]``, ``Slice[1..3]``, ``Tile[1]``, and the ``Concat`` members
  that build shape tensors from them) — ORT: "Type Error: Type
  'tensor(int32)' of input parameter (...) of operator (Reshape) ... is
  invalid";
* the ``Shape`` nodes carry a spurious ``dtype`` attribute (tf2onnx
  artifact hinting int32 output) that ``onnx.checker`` rejects — standard
  ``Shape`` at opset 11 is attribute-free and always emits int64, which is
  exactly what the shape-cluster casts produce;
* 99 data-carrying initializers are duplicated into ``graph.input``
  (TF-exporter style), and the 5 streaming conv buffer states appear BOTH as
  runtime inputs (with dynamic dims) and as *empty* placeholder
  initializers, which ``onnx.checker`` rejects ("should contain one and only
  one value field").

``make_portable`` fixes exactly these, touching nothing else: int32
shape-cluster constants get a ``Cast(to=int64)`` node inserted and the
consumer positions rewired; initializer names are dropped from
``graph.input`` (the conv-buffer placeholders keep their runtime-input
declarations and lose their empty initializer stubs); every op — including
the Hybrid* nodes and their int32 zp/sums constants — is left byte-for-byte
as exported.

ORT_LOAD_ERROR_HYBRID is the sentinel for "the graph is mechanically clean;
the only remaining load blocker is the unknown Hybrid* ops" — Task 7's
rewrites make it load.
"""
from __future__ import annotations

import io

import onnx
from onnx import helper

# ORT's unknown-op failure names each unregistered node like
# "Fatal error: HybridGemm(...) is not a registered function/op" — every
# such name starts with the vendor kernel-family prefix.  A load error that
# mentions no Hybrid op is a mechanical/typing failure this module (or a
# later rewrite) must still fix.
ORT_LOAD_ERROR_HYBRID = "Hybrid"

# Input positions that the ONNX spec types as tensor(int64) only.
_SEED_POSITIONS: dict[str, tuple[int, ...]] = {
    "Reshape": (1,),
    "Slice": (1, 2, 3),
    "Tile": (1,),
    "Expand": (1,),
    "Pad": (1,),
}

# Ops whose output dtype follows an input's dtype: when such an output feeds
# an int64-required position, the int32 constants on that path must be cast
# too (e.g. Concat building a Reshape shape tensor from const members).
# Values are the input positions whose dtype flows to the output (None =
# all inputs, i.e. Concat's same-type rule).
_PROPAGATING = {
    "Concat": None,
    "Gather": (0,),      # data only (indices may stay int32)
    "Slice": (0,),       # data only
    "Squeeze": (0,),
    "Unsqueeze": (0,),
    "Identity": (0,),
}


def _has_data(init: onnx.TensorProto) -> bool:
    return bool(init.raw_data or init.float_data or init.int32_data
                or init.int64_data or init.double_data or init.uint64_data
                or init.string_data)


def _shape_cluster_casts(graph) -> dict[str, str]:
    """Map int32 initializer name -> cast output name for every constant on
    a path that reaches an int64-only shape position."""
    init_int32 = {
        i.name for i in graph.initializer
        if i.data_type == onnx.TensorProto.INT32 and _has_data(i)
    }
    # Backward dataflow: start from every int64-required position, walk the
    # producers of the tensors feeding it, and mark int32 initializers.
    producer: dict[str, onnx.NodeProto] = {}
    for node in graph.node:
        for out in node.output:
            producer[out] = node

    casts: dict[str, str] = {}

    def need_int64(tensor: str, seen: set[str]) -> None:
        if tensor in init_int32:
            casts[tensor] = tensor + "__pypryon_i64"
            return
        node = producer.get(tensor)
        if node is None or tensor in seen or node.op_type not in _PROPAGATING:
            return
        seen.add(tensor)
        positions = _PROPAGATING[node.op_type]
        inputs = (node.input if positions is None
                  else [node.input[p] for p in positions if p < len(node.input)])
        for inp in inputs:
            need_int64(inp, seen)

    for node in graph.node:
        for pos in _SEED_POSITIONS.get(node.op_type, ()):
            if pos < len(node.input):
                need_int64(node.input[pos], set())

    return casts


def make_portable(onnx_bytes: bytes) -> bytes:
    """Mechanically portability-fix a Pryon nemort ONNX graph.  Idempotent."""
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

    # 2. Cast int32 shape-cluster constants to int64, rewiring consumers.
    casts = _shape_cluster_casts(graph)
    if casts:
        def rewire_positions(node: onnx.NodeProto) -> tuple[int, ...]:
            """Positions of this node that must see the int64 cast."""
            if node.op_type in _SEED_POSITIONS:
                return _SEED_POSITIONS[node.op_type]
            prop = _PROPAGATING.get(node.op_type)
            if prop is None:                # Concat: every member, same dtype
                return tuple(range(len(node.input)))
            return prop

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

    out = io.BytesIO()
    onnx.save(model, out)
    return out.getvalue()
