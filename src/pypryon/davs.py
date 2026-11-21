"""Load a DAVS model set: a shared encoder plus per-word decoders and a merged kw.cfg.

Layout (alexa-en-US): ``pryon.config`` is a list of ``INCLUDE "<dir>/pryon.config"``
lines; the merged ``BDPGeneratedFiles`` include names ``decoders.cfg.json`` (decoders),
``kw.cfg.json`` (spotters) and ``words.shrunk.txt``; ``encoder/pryon.config`` names the
encoder through ``scorer.single.config_filepath``. Paths are relative to the file that
sets them. Decoders no spotter references (DVAD, speaker ID) are never opened.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .modelset import FeConfig, ModelError, Override, _load_fe, _load_words

_INCLUDE = "INCLUDE"
# '#' starts a comment only after whitespace: keys like `enable#disable-upsampling` exist
_COMMENT = re.compile(r"\s+#.*$")


@dataclass(frozen=True)
class Setting:
    value: Any
    base: Path      # directory of the file that set it

    @property
    def path(self) -> Path:
        return self.base / str(self.value)


def pryon_config(path: Path) -> dict[str, Setting]:
    """Kaldi-style ``key = value`` lines with ``INCLUDE "<rel>"`` expanded in place."""
    out: dict[str, Setting] = {}
    _read_config(Path(path), out, ())
    return out


def _read_config(path: Path, out: dict[str, Setting], stack: tuple[Path, ...]) -> None:
    path = path.resolve()
    if path in stack:
        raise ModelError(f"{path.name}: INCLUDE cycle via {' -> '.join(p.name for p in stack)}")
    if not path.is_file():
        ref = f" (included from {stack[-1]})" if stack else ""
        raise ModelError(f"{path}: file not found{ref}")
    for lineno, raw in enumerate(path.read_text().splitlines(), 1):
        line = _COMMENT.sub("", raw).strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(_INCLUDE + " "):
            _read_config(path.parent / _unquote(line[len(_INCLUDE):].strip()), out, (*stack, path))
            continue
        key, sep, rhs = line.partition("=")
        if not sep or not key.strip():
            raise ModelError(f"{path}: line {lineno} is not `key = value`: {raw!r}")
        out[key.strip()] = Setting(_scalar(rhs.strip()), path.parent)


def _unquote(token: str) -> str:
    return token[1:-1] if len(token) >= 2 and token[0] == token[-1] == '"' else token


def _scalar(token: str) -> Any:
    if len(token) >= 2 and token[0] == token[-1] == '"':
        return token[1:-1]
    for convert in (int, float):
        try:
            return convert(token)
        except ValueError:
            pass
    return token


def _require(cfg: dict[str, Setting], key: str, where: Path) -> Setting:
    if key not in cfg:
        raise ModelError(f"{where}: missing required key {key!r}")
    return cfg[key]


def _json(path: Path) -> Any:
    if not path.is_file():
        raise ModelError(f"{path}: file not found")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ModelError(f"{path}: invalid JSON: {exc}") from exc


def _get(obj: Any, key: str, path: Path, where: str = "") -> Any:
    if not isinstance(obj, dict) or key not in obj:
        raise ModelError(f"{path}: missing {where}{key!r}")
    return obj[key]


@dataclass(frozen=True)
class GraphInput:
    name: str
    source: str            # "audio" or "tensor_artifact"
    tensor: str | None     # artifact name for tensor_artifact inputs


@dataclass(frozen=True)
class GraphSpec:
    name: str
    path: Path
    inputs: list[GraphInput]
    outputs: dict[str, str | None]   # component name -> published artifact name
    state_inputs: list[str]          # LSTM h/c first, then conv buffers (positional with outputs)
    state_outputs: list[str]
    lstm_states: int
    frames_per_step: int
    feat_dim: int


@dataclass(frozen=True)
class WordComponent:
    class_index: int
    component: str
    start_end: str | None
    start_index: int
    end_index: int
    history: int | None = None


@dataclass(frozen=True)
class SpotterSpec:
    name: str
    auxiliary: bool
    scorer: str
    components: list[WordComponent]
    classifier: dict
    accept: float
    notify: float | None = None
    min_accept: float | None = None
    max_accept: float | None = None
    overrides: list[Override] = field(default_factory=list)
    smoothers: list[dict] = field(default_factory=list)
    kw_window: int = 100
    lock: int = 100


@dataclass(frozen=True)
class DavsModelSet:
    dir: Path
    fe: FeConfig
    encoder: GraphSpec
    decoders: dict[str, GraphSpec]
    spotters: list[SpotterSpec]
    words: dict[int, str]
    client_prop_defaults: dict[str, int]


def is_davs(cfg: dict[str, Setting]) -> bool:
    return "scorer.multi.config_filepath" in cfg or "scorer.single.config_filepath" in cfg


def _graph(name: str, scorer: dict, scorer_path: Path) -> GraphSpec:
    nemort_path = scorer_path.parent / str(_get(scorer, "scorer.model_filepath", scorer_path, f"{name}."))
    nemort = _json(nemort_path)
    model = nemort_path.parent / str(_get(nemort, "model_filepath", nemort_path))
    if "inputs" in nemort:
        inputs = [GraphInput(str(_get(i, "input_name", nemort_path, "inputs[].")),
                             str(i.get("input_source", "audio")), i.get("tensor_name"))
                  for i in _get(nemort, "inputs", nemort_path)]
        outputs = {str(_get(o, "component_name", nemort_path, "outputs[].")): o.get("tensor_name")
                   for o in _get(nemort, "outputs", nemort_path)}
    else:  # the older single-graph schema
        mapping = _get(nemort, "model_component_mapping", nemort_path)
        inputs = [GraphInput(str(_get(i, "component_name", nemort_path)), "audio", None)
                  for i in _get(mapping, "input_map", nemort_path)]
        outputs = {str(_get(o, "component_name", nemort_path)): None
                   for o in _get(mapping, "output_map", nemort_path)}
    lstm = nemort.get("lstm_components", [])
    state_in = [str(_get(p, "input_component", nemort_path)) for p in lstm]
    state_out = [str(_get(p, "output_component", nemort_path)) for p in lstm]
    state_in += [str(n) for n in nemort.get("recurrent_components_input", [])]
    state_out += [str(n) for n in nemort.get("recurrent_components_output", [])]
    _check_graph(model, nemort_path, [i.name for i in inputs] + state_in, list(outputs) + state_out)
    return GraphSpec(name, model, inputs, outputs, state_in, state_out, len(lstm),
                     int(scorer.get("scorer.initialize.frame_count", 1)),
                     int(_get(scorer, "scorer.feat.dimensions", scorer_path, f"{name}.")))


def _check_graph(model: Path, ref: Path, inputs: list[str], outputs: list[str]) -> None:
    import onnx

    if not model.is_file():
        raise ModelError(f"{model}: file not found (referenced from {ref.name})")
    try:
        graph = onnx.load(model, load_external_data=False).graph
    except Exception as exc:  # onnx raises a variety of decode errors
        raise ModelError(f"{model.name}: cannot load ONNX graph: {exc}") from exc
    have_in, have_out = {i.name for i in graph.input}, {o.name for o in graph.output}
    for name in inputs:
        if name not in have_in:
            raise ModelError(f"{model.name}: declared input {name!r} (from {ref.name}) is not a graph input")
    for name in outputs:
        if name not in have_out:
            raise ModelError(f"{model.name}: declared output {name!r} (from {ref.name}) is not a graph output")


def _scorers(path: Path) -> dict[str, tuple[dict, Path]]:
    data = _json(path)
    return {str(_get(s, "name", path, "scorers[].")): (s, path) for s in _get(data, "scorers", path)}


def _thresholds(raw: dict, path: Path, where: str) -> dict[str, Any]:
    def num(key: str) -> float | None:
        v = raw.get(key)
        if v is None:
            return None
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ModelError(f"{path}: {where}{key!r} must be a number, got {v!r}")
        return float(v)

    if num("accept-threshold") is None:
        raise ModelError(f"{path}: missing {where}'accept-threshold'")
    overrides: list[Override] = []
    for o in raw.get("overrides", []):
        props = _get(o, "clientProperties", path, where + "overrides[].")
        if not isinstance(props, list) or len(props) != 1:
            raise ModelError(f"{path}: {where}overrides[] must name exactly one client property")
        overrides.append((str(_get(props[0], "name", path)), int(_get(props[0], "equals", path)),
                          float(_get(o, "accept-threshold", path))))
    return {"accept": num("accept-threshold"), "notify": num("notify-threshold"),
            "min_accept": num("min-accept-threshold"), "max_accept": num("max-accept-threshold"),
            "overrides": overrides}


def _spotter(raw: dict, path: Path, i: int) -> SpotterSpec:
    where = f"keyword-spotters[{i}]."
    spot = _get(raw, "spotter", path, where)
    scorers = _get(raw, "scorers", path, where)
    if not isinstance(scorers, list) or len(scorers) != 1:
        raise ModelError(f"{path}: {where}'scorers' must list exactly one scorer")
    components = [WordComponent(int(_get(c, "scorer-index", path, where)),
                                str(_get(_get(c, "scorer-output-component", path, where), "name", path)),
                                (c.get("keyword-start-end-component") or {}).get("name"),
                                int(c.get("keyword-start-scorer-index", 0)),
                                int(c.get("keyword-end-scorer-index", 1)),
                                c.get("word-score-history-size"))
                  for c in _get(spot, "word-component-array", path, where)]
    return SpotterSpec(
        name=str(_get(raw, "name", path, where)),
        auxiliary=spot.get("spotter-type") == "DETECTION_AUXILIARY",
        scorer=str(scorers[0]), components=components,
        classifier=dict(_get(raw, "classifier", path, where)),
        smoothers=list(spot.get("posterior-smoothers", [])),
        kw_window=int(_get(spot, "keyword-feature-window-length", path, where)),
        lock=int(_get(spot, "lock-period", path, where)),
        **_thresholds(_get(spot, "classification-thresholds", path, where), path, where))


def load_davs(model_dir: Path, cfg: dict[str, Setting]) -> DavsModelSet:
    config_path = model_dir / "pryon.config"
    fe = _load_fe({k: s.value for k, s in cfg.items() if k.startswith("fe.")}, config_path)

    enc_name, (enc_scorer, enc_path) = next(iter(
        _scorers(_require(cfg, "scorer.single.config_filepath", config_path).path).items()))
    encoder = _graph(enc_name, enc_scorer, enc_path)

    kw_path = _require(cfg, "keyword_spotter.config_filepath", config_path).path
    raw_spotters = _get(_get(_json(kw_path), "keyword-orchestra", kw_path), "keyword-spotters", kw_path)
    spotters = [_spotter(s, kw_path, i) for i, s in enumerate(raw_spotters)]

    available = _scorers(_require(cfg, "scorer.multi.config_filepath", config_path).path)
    decoders: dict[str, GraphSpec] = {}
    for spot in spotters:
        if spot.scorer not in available:
            raise ModelError(f"{kw_path}: spotter {spot.name} uses unknown scorer {spot.scorer!r}")
        if spot.scorer not in decoders:
            decoders[spot.scorer] = _graph(spot.scorer, *available[spot.scorer])

    words_setting = _require(cfg, "id2word.mapping_filepath", config_path)
    words = _load_words(words_setting.path, config_path)
    props = {name: 0 for s in spotters for name, _, _ in s.overrides}
    return DavsModelSet(model_dir, fe, encoder, decoders, spotters, words, props)
