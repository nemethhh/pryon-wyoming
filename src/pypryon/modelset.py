"""Load a Pryon model directory (model set) into typed structures.

``pryon.manifest`` is the entry point (``default_model_name`` ->
``models[name].config_file_path``); from there:

- ``pryon.config`` — Kaldi-style ``key = value`` lines: the LFBE front end
  (``fe.*``) and pointers to the files below;
- ``nemort_scorer_config.json`` — the ONNX file, its feature input and the
  externalized recurrent state tensors (LSTM h/c pairs, conv buffers); names
  are validated against the graph;
- ``kw.cfg.json`` — per-keyword thresholds, client-property overrides,
  classifier windowing and lock period;
- ``words.shrunk.txt`` — ``word index`` lines; index 0 ``<eps>`` is the
  softmax background class.

Not parsed: ``watermark.cfg.json`` and the fingerprint front end (Amazon leak
tracking), ``cmvn.6khz.mat``, and ``op.cfg.json`` (wake/stop UX policy).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# A threshold override keyed on one client property: (name, equals, accept).
# kw.cfg.json expresses each override as one clientProperties entry; several
# entries per override would be an AND the flat type cannot represent, so the
# loader rejects that loudly instead of guessing semantics.
Override = tuple[str, int, float]


class ModelError(RuntimeError):
    """A model-set file could not be parsed; the message names the offending file."""


@dataclass(frozen=True)
class FeConfig:
    """LFBE frontend parameters, all parsed from pryon.config (fe.* keys).

    fft_pow2 is derived, not hardcoded: frame_length_milli * samplerate / 1000
    samples (25 ms @ 16 kHz = 400) rounded up to a power of two when
    fe.audio_analysis.round_to_power_of_two is set — the Kaldi behavior the
    vendor LEDGER records as "FFT rounded to 512" for this model.
    """

    num_mel: int        # fe.mel_fbank.num_bins
    frame_ms: float     # fe.audio_analysis.frame_length_milli
    shift_ms: float     # fe.audio_analysis.frame_shift_milli
    preemph: float      # fe.audio_analysis.preemphasis_coeff
    low_hz: float       # fe.mel_fbank.low_freq
    high_hz: float      # fe.mel_fbank.high_freq
    fft_pow2: int       # derived: window samples, rounded to pow2 (see above)
    remove_dc: bool     # fe.audio_analysis.remove_dc_offset (flag)
    window: str         # fe.audio_analysis.window_type
    use_energy: int     # fe.cep.use_energy (flag)


@dataclass(frozen=True)
class KeywordSpec:
    """One keyword spotter of kw.cfg.json (keyword-orchestra.keyword-spotters[]).

    The defaults are the ALEXA kw.cfg.json values; the loader always parses
    every field, so they never stand in for missing data.
    """

    name: str                     # spotter "name"
    class_index: int              # word-component-array[0] "scorer-index" (ww softmax class)
    scorer_index_start: int       # word-component-array[0] "keyword-start-scorer-index" (se)
    scorer_index_end: int         # word-component-array[0] "keyword-end-scorer-index" (se)
    accept: float                 # spotter.classification-thresholds."accept-threshold"
    notify: float | None = None   # ... ."notify-threshold" (near-miss; STOP has none)
    overrides: list[Override] = field(default_factory=list)  # ... ."overrides"
    window: int = 3               # classifier "window-size"
    trim_tail: int = 8            # classifier "trim-tail"
    kw_window: int = 100          # spotter "keyword-feature-window-length"
    lock: int = 100               # spotter "lock-period"


@dataclass(frozen=True)
class ModelSet:
    """A parsed model directory: everything the front end, runner and decoder need."""

    graph_path: Path            # nemort_scorer_config.json "model_filepath", resolved
    input_name: str             # its input_map[0] "component_name" (input_feats)
    state_inputs: list[str]     # lstm_components inputs + recurrent_components_input
    state_outputs: list[str]    # lstm_components outputs + recurrent_components_output
    fe: FeConfig
    keywords: list[KeywordSpec]
    words: dict[int, str]       # id2word mapping; 0 "<eps>" = background class
    # client properties named by kw.cfg overrides; no file declares defaults, and
    # 0 (unset) matches no override since overrides test equals == 1
    client_prop_defaults: dict[str, int]


def _parse_pryon_config(text: str, path: Path) -> dict[str, Any]:
    """Parse the Kaldi-style `key = value` lines of pryon.config into a flat dict.

    Values are quoted strings (quotes stripped), ints, floats, or bare tokens;
    blank lines and #-comments are skipped.
    """
    values: dict[str, Any] = {}
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, rhs = line.partition("=")
        if not sep or not key.strip():
            raise ModelError(f"{path.name}: line {lineno} is not `key = value`: {raw!r}")
        token = rhs.strip()
        if len(token) >= 2 and token[0] == '"' and token[-1] == '"':
            values[key.strip()] = token[1:-1]
            continue
        for convert in (int, float):
            try:
                values[key.strip()] = convert(token)
                break
            except ValueError:
                pass
        else:
            values[key.strip()] = token
    return values


def _require(values: dict[str, Any], key: str, path: Path) -> Any:
    if key not in values:
        raise ModelError(f"{path.name}: missing required key {key!r}")
    return values[key]


def _number(values: dict[str, Any], key: str, path: Path) -> float:
    v = _require(values, key, path)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ModelError(f"{path.name}: key {key!r} must be a number, got {v!r}")
    return float(v)


def _int(values: dict[str, Any], key: str, path: Path) -> int:
    v = _require(values, key, path)
    if isinstance(v, bool) or not isinstance(v, int):
        raise ModelError(f"{path.name}: key {key!r} must be an integer, got {v!r}")
    return v


def _load_json(path: Path, referrer: Path | None = None) -> Any:
    if not path.is_file():
        ref = f" (referenced from {referrer.name})" if referrer else ""
        raise ModelError(f"{path.name}: file not found{ref}")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ModelError(f"{path.name}: invalid JSON: {exc}") from exc


def _dig(data: Any, dotted: str, path: Path) -> Any:
    """Fetch a nested JSON value by dotted path, with list indices as segments."""
    cur = data
    for key in dotted.split("."):
        if isinstance(cur, list) and key.isdigit():
            if int(key) >= len(cur):
                raise ModelError(f"{path.name}: missing entry {dotted!r}")
            cur = cur[int(key)]
        elif isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            raise ModelError(f"{path.name}: missing key {dotted!r}")
    return cur


def _key(obj: Any, key: str, where: str, path: Path) -> Any:
    """Fetch one key of a JSON object already located at ``where``, else ModelError."""
    if not isinstance(obj, dict) or key not in obj:
        raise ModelError(f"{path.name}: missing {where}{key!r}")
    return obj[key]


def _num(value: Any, where: str, path: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelError(f"{path.name}: {where} must be a number, got {value!r}")
    return float(value)


def _num_int(value: Any, where: str, path: Path) -> int:
    n = _num(value, where, path)
    if not n.is_integer():
        raise ModelError(f"{path.name}: {where} must be an integer, got {value!r}")
    return int(n)


def _load_fe(cfg: dict[str, Any], path: Path) -> FeConfig:
    fe_type = _require(cfg, "fe.type", path)
    if fe_type != "LFBE":
        raise ModelError(
            f"{path.name}: fe.type {fe_type!r} is not LFBE — only the LFBE frontend is supported"
        )
    frame_ms = _number(cfg, "fe.audio_analysis.frame_length_milli", path)
    shift_ms = _number(cfg, "fe.audio_analysis.frame_shift_milli", path)
    samplerate = _int(cfg, "fe.audio_analysis.samplerate", path)
    window_samples = int(round(frame_ms * samplerate / 1000))
    if _int(cfg, "fe.audio_analysis.round_to_power_of_two", path):
        fft_pow2 = 1 << (window_samples - 1).bit_length()  # 400 -> 512
    else:
        fft_pow2 = window_samples
    return FeConfig(
        num_mel=_int(cfg, "fe.mel_fbank.num_bins", path),
        frame_ms=frame_ms,
        shift_ms=shift_ms,
        preemph=_number(cfg, "fe.audio_analysis.preemphasis_coeff", path),
        low_hz=_number(cfg, "fe.mel_fbank.low_freq", path),
        high_hz=_number(cfg, "fe.mel_fbank.high_freq", path),
        fft_pow2=fft_pow2,
        remove_dc=bool(_int(cfg, "fe.audio_analysis.remove_dc_offset", path)),
        window=str(_require(cfg, "fe.audio_analysis.window_type", path)),
        use_energy=_int(cfg, "fe.cep.use_energy", path),
    )


def _load_words(path: Path, referrer: Path) -> dict[int, str]:
    if not path.is_file():
        raise ModelError(f"{path.name}: file not found (referenced from {referrer.name})")
    words: dict[int, str] = {}
    for lineno, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        try:
            index = int(parts[-1])
        except ValueError:
            index = None
        if index is None or len(parts) < 2:
            raise ModelError(f"{path.name}: line {lineno} is not `word index`: {raw!r}")
        if index in words:
            raise ModelError(f"{path.name}: duplicate index {index} (line {lineno})")
        words[index] = " ".join(parts[:-1])
    return words


def _load_keywords(
    data: Any, path: Path
) -> tuple[list[KeywordSpec], dict[str, int]]:
    spotters = _dig(data, "keyword-orchestra.keyword-spotters", path)
    if not isinstance(spotters, list) or not spotters:
        raise ModelError(f"{path.name}: 'keyword-orchestra.keyword-spotters' must be a non-empty list")
    keywords: list[KeywordSpec] = []
    prop_names: list[str] = []
    for i in range(len(spotters)):
        where = f"keyword-spotters[{i}]."
        spotter = _dig(data, f"keyword-orchestra.keyword-spotters.{i}.spotter", path)
        thresholds = _key(spotter, "classification-thresholds", where + "spotter.", path)
        where_t = where + "spotter.classification-thresholds."
        accept = _num(_key(thresholds, "accept-threshold", where_t, path), where_t + "'accept-threshold'", path)
        notify = _num(_key(thresholds, "notify-threshold", where_t, path), where_t + "'notify-threshold'", path) \
            if "notify-threshold" in thresholds else None
        overrides: list[Override] = []
        for j, o in enumerate(thresholds.get("overrides", [])):
            where_o = f"{where_t}overrides[{j}]."
            props = _key(o, "clientProperties", where_o, path)
            if not isinstance(props, list) or len(props) != 1:
                raise ModelError(
                    f"{path.name}: {where_o}has {len(props) if isinstance(props, list) else 'non-list'} "
                    "clientProperties; exactly one property per override is supported"
                )
            where_p = where_o + "clientProperties[0]."
            prop = _key(props[0], "name", where_p, path)
            equals = _key(props[0], "equals", where_p, path)
            if isinstance(equals, bool) or not isinstance(equals, int):
                raise ModelError(f"{path.name}: {where_p}'equals' must be an integer, got {equals!r}")
            ovr_accept = _num(_key(o, "accept-threshold", where_o, path), where_o + "'accept-threshold'", path)
            overrides.append((prop, equals, ovr_accept))
            prop_names.append(prop)
        classifier = _dig(data, f"keyword-orchestra.keyword-spotters.{i}.classifier", path)
        where_c = where + "classifier."
        word_components = _key(spotter, "word-component-array", where + "spotter.", path)
        if not isinstance(word_components, list) or len(word_components) != 1:
            raise ModelError(
                f"{path.name}: {where}spotter.word-component-array has "
                f"{len(word_components) if isinstance(word_components, list) else 'non-list'} entries; "
                "exactly one per keyword is supported"
            )
        where_w = where + "spotter.word-component-array[0]."
        wc = word_components[0]
        keywords.append(KeywordSpec(
            name=_dig(data, f"keyword-orchestra.keyword-spotters.{i}.name", path),
            class_index=_num_int(_key(wc, "scorer-index", where_w, path), where_w + "'scorer-index'", path),
            scorer_index_start=_num_int(_key(wc, "keyword-start-scorer-index", where_w, path),
                                        where_w + "'keyword-start-scorer-index'", path),
            scorer_index_end=_num_int(_key(wc, "keyword-end-scorer-index", where_w, path),
                                      where_w + "'keyword-end-scorer-index'", path),
            accept=accept,
            notify=notify,
            overrides=overrides,
            window=_num_int(_key(classifier, "window-size", where_c, path), where_c + "'window-size'", path),
            trim_tail=_num_int(_key(classifier, "trim-tail", where_c, path), where_c + "'trim-tail'", path),
            kw_window=_num_int(_key(spotter, "keyword-feature-window-length", where + "spotter.", path),
                               where + "spotter.'keyword-feature-window-length'", path),
            lock=_num_int(_key(spotter, "lock-period", where + "spotter.", path),
                          where + "spotter.'lock-period'", path),
        ))
    return keywords, {name: 0 for name in prop_names}


def load_model_set(model_dir: Path) -> ModelSet:
    """Parse a Pryon model directory (see module docstring for file roles)."""
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise ModelError(f"{model_dir}: not a directory")

    # pryon.manifest: entry point of the set (default model name -> config path).
    manifest = _load_json(model_dir / "pryon.manifest")
    if not isinstance(manifest, dict):
        raise ModelError("pryon.manifest: top level must be an object")
    model_name = _dig(manifest, "default_model_name", Path("pryon.manifest"))
    models = _dig(manifest, "models", Path("pryon.manifest"))
    if not isinstance(models, dict) or model_name not in models:
        raise ModelError(f"pryon.manifest: no model {model_name!r} in 'models'")
    config_name = _key(models[model_name], "config_file_path", f"models.{model_name!r}.",
                       Path("pryon.manifest"))
    config_path = model_dir / str(config_name)

    # pryon.config: Kaldi-style lines; frontend + pointers to the other files.
    if not config_path.is_file():
        raise ModelError(f"{config_path.name}: file not found (referenced from pryon.manifest)")
    cfg = _parse_pryon_config(config_path.read_text(), config_path)
    fe = _load_fe(cfg, config_path)

    # nemort_scorer_config.json: ONNX path, feature input, externalized states.
    nemort_path = model_dir / str(_require(cfg, "scorer.nemort_config_json_filepath", config_path))
    nemort = _load_json(nemort_path, config_path)
    if not isinstance(nemort, dict):
        raise ModelError(f"{nemort_path.name}: top level must be an object")
    input_map = _dig(nemort, "model_component_mapping.input_map", nemort_path)
    if not isinstance(input_map, list) or len(input_map) != 1:
        raise ModelError(
            f"{nemort_path.name}: input_map has "
            f"{len(input_map) if isinstance(input_map, list) else 'non-list'} entries; "
            "exactly one feature input is supported"
        )
    input_name = _key(input_map[0], "component_name", "model_component_mapping.input_map[0].", nemort_path)
    lstm = _dig(nemort, "lstm_components", nemort_path)
    if not isinstance(lstm, list):
        raise ModelError(f"{nemort_path.name}: 'lstm_components' must be a list")
    state_inputs = [_key(pair, "input_component", f"lstm_components[{i}].", nemort_path)
                    for i, pair in enumerate(lstm)]
    state_outputs = [_key(pair, "output_component", f"lstm_components[{i}].", nemort_path)
                     for i, pair in enumerate(lstm)]
    conv_in = _dig(nemort, "recurrent_components_input", nemort_path)
    conv_out = _dig(nemort, "recurrent_components_output", nemort_path)
    if not isinstance(conv_in, list) or not isinstance(conv_out, list):
        raise ModelError(
            f"{nemort_path.name}: 'recurrent_components_input/output' must be lists"
        )
    state_inputs += conv_in
    state_outputs += conv_out

    # The graph itself: names are taken from nemort_scorer_config.json but must
    # exist in the ONNX (inspect, don't guess).
    graph_path = model_dir / str(_dig(nemort, "model_filepath", nemort_path))
    if not graph_path.is_file():
        raise ModelError(f"{graph_path.name}: file not found (referenced from {nemort_path.name})")
    import onnx

    try:
        graph = onnx.load(graph_path, load_external_data=False).graph
    except Exception as exc:  # onnx raises a variety of load/decode errors
        raise ModelError(f"{graph_path.name}: cannot load ONNX graph: {exc}") from exc
    graph_inputs = {i.name for i in graph.input}
    graph_outputs = {o.name for o in graph.output}
    for name in [input_name, *state_inputs]:
        if name not in graph_inputs:
            raise ModelError(
                f"{graph_path.name}: declared input {name!r} "
                f"(from {nemort_path.name}) is not a graph input"
            )
    for name in state_outputs:
        if name not in graph_outputs:
            raise ModelError(
                f"{graph_path.name}: declared state output {name!r} "
                f"(from {nemort_path.name}) is not a graph output"
            )

    # kw.cfg.json: keyword spotters, thresholds and lock config.
    kw_path = model_dir / str(_require(cfg, "keyword_spotter.config_filepath", config_path))
    kw = _load_json(kw_path, config_path)
    keywords, client_prop_defaults = _load_keywords(kw, kw_path)

    words = _load_words(
        model_dir / str(_require(cfg, "id2word.mapping_filepath", config_path)), config_path
    )

    return ModelSet(
        graph_path=graph_path,
        input_name=input_name,
        state_inputs=state_inputs,
        state_outputs=state_outputs,
        fe=fe,
        keywords=keywords,
        words=words,
        client_prop_defaults=client_prop_defaults,
    )
