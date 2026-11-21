"""Model-set loader: the real ALEXA directory (``PYPRYON_MODEL_DIR``) and every
error path on a minimal hand-built set.
"""
import json
import os
from pathlib import Path

import pytest

from pypryon.modelset import FeConfig, KeywordSpec, ModelError, load_model_set

from conftest import MODEL_DIR, requires_model

# The exact externalized-state tensor names of the shipped ALEXA graph, as
# declared by its nemort_scorer_config.json (4 LSTM h/c pairs, in
# lstm_components order, then the 5 streaming conv buffers) and confirmed
# present in the ONNX itself.
EXPECTED_STATE_INPUTS = [
    "input_h_ww", "input_c_ww", "input_h_se", "input_c_se",
    "streaming_conv_buffer_input_7",
    "streaming_conv_buffer_input_15",
    "streaming_conv_buffer_input_23",
    "streaming_conv_buffer_input_30",
    "streaming_conv_buffer_input_37",
]
EXPECTED_STATE_OUTPUTS = [
    "lstm_ww_1", "lstm_ww_1_1", "lstm_se_1", "lstm_se_1_1",
    "streaming_conv_buffer_output_7",
    "streaming_conv_buffer_output_15",
    "streaming_conv_buffer_output_23",
    "streaming_conv_buffer_output_30",
    "streaming_conv_buffer_output_37",
]




def _tiny_onnx_bytes() -> bytes:
    """A minimal ONNX with the input/state/output names the synthetic set declares."""
    import onnx
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        [
            helper.make_node("Identity", ["feats"], ["probs"]),
            helper.make_node("Identity", ["h_in"], ["h_out"]),
        ],
        "tiny",
        [
            helper.make_tensor_value_info("feats", TensorProto.FLOAT, [1, "N", 64, 1]),
            helper.make_tensor_value_info("h_in", TensorProto.FLOAT, [25, 100]),
        ],
        [
            helper.make_tensor_value_info("probs", TensorProto.FLOAT, [25, "N", 3]),
            helper.make_tensor_value_info("h_out", TensorProto.FLOAT, [25, 100]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    return model.SerializeToString()


def _write_min_model(root: Path, *, pryon_config: str | None = None) -> Path:
    """Write a complete minimal model set (config text overridable per test)."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "pryon.manifest").write_text(json.dumps({
        "object_type_name": "ModelSetManifest",
        "object_type_version": 1,
        "default_model_name": "pryon",
        "models": {"pryon": {"config_file_path": "pryon.config"}},
    }))
    (root / "pryon.config").write_text(pryon_config if pryon_config is not None else "\n".join([
        'fe.type = "LFBE"',
        "fe.audio_analysis.frame_length_milli = 25",
        "fe.audio_analysis.frame_shift_milli = 10",
        "fe.audio_analysis.preemphasis_coeff = 0.96875",
        "fe.audio_analysis.remove_dc_offset = 1",
        "fe.audio_analysis.round_to_power_of_two = 1",
        "fe.audio_analysis.samplerate = 16000",
        'fe.audio_analysis.window_type = "povey"',
        "fe.cep.use_energy = 0",
        "fe.mel_fbank.num_bins = 64",
        "fe.mel_fbank.low_freq = 80",
        "fe.mel_fbank.high_freq = 7200",
        'scorer.nemort_config_json_filepath = "nemort.json"',
        'keyword_spotter.config_filepath = "kw.cfg.json"',
        'id2word.mapping_filepath = "words.txt"',
    ]) + "\n")
    (root / "nemort.json").write_text(json.dumps({
        "object_type_name": "nemort_scorer_config",
        "model_filepath": "model.onnx",
        "model_component_mapping": {"input_map": [{"component_name": "feats"}]},
        "lstm_components": [{"input_component": "h_in", "output_component": "h_out"}],
        "recurrent_components_input": [],
        "recurrent_components_output": [],
    }))
    (root / "kw.cfg.json").write_text(json.dumps({
        "object_type_name": "keyword-spotter-orchestra-config",
        "keyword-orchestra": {"keyword-spotters": [{
            "name": "TESTKW",
            "classifier": {"type": "moving-average-max-pool", "window-size": 3, "trim-tail": 8},
            "spotter": {
                "classification-thresholds": {
                    "accept-threshold": 0.5,
                    "overrides": [{
                        "accept-threshold": 0.4,
                        "clientProperties": [{"name": "TestProp", "equals": 1}],
                    }],
                },
                "word-component-array": [{
                    "scorer-index": 1,
                    "keyword-start-scorer-index": 0,
                    "keyword-end-scorer-index": 1,
                }],
                "keyword-feature-window-length": 100,
                "lock-period": 100,
            },
        }]},
    }))
    (root / "words.txt").write_text("<eps> 0\nTESTKW 1\n")
    (root / "model.onnx").write_bytes(_tiny_onnx_bytes())
    return root


@requires_model
def test_load_real_alexa_model_set() -> None:
    ms = load_model_set(MODEL_DIR)

    assert ms.graph_path == MODEL_DIR / "runtime_stateless_standard.streaming.fixed.int16.onnx"
    assert ms.input_name == "input_feats"
    assert ms.state_inputs == EXPECTED_STATE_INPUTS
    assert ms.state_outputs == EXPECTED_STATE_OUTPUTS

    # Frontend, per pryon.config fe.* (values cross-checked with vendor LEDGER
    # §"Audio frontend configuration and path": 25/10 ms, preemph 0.96875,
    # Povey window, FFT rounded to 512, 64 mel bands 80-7200 Hz, DC removal,
    # use_energy 0).
    assert ms.fe == FeConfig(
        num_mel=64, frame_ms=25, shift_ms=10, preemph=0.96875,
        low_hz=80, high_hz=7200, fft_pow2=512, remove_dc=True,
        window="povey", use_energy=0,
    )

    # Both spotters of kw.cfg.json, in file order, with every classifier and
    # lock field parsed (nothing vendor-configurable left hardcoded).
    assert len(ms.keywords) == 2
    assert ms.keywords[0] == KeywordSpec(
        name="ALEXA", class_index=1, scorer_index_start=0, scorer_index_end=1,
        accept=0.923, notify=0.69,
        overrides=[
            ("AlarmState", 1, 0.834),
            ("AudioPlayerState", 1, 0.864),
            ("audio_playback", 1, 0.864),
        ],
        window=3, trim_tail=8, kw_window=100, lock=100,
    )
    assert ms.keywords[1] == KeywordSpec(
        name="STOP", class_index=2, scorer_index_start=0, scorer_index_end=1,
        accept=0.55, notify=None, overrides=[],
        window=3, trim_tail=8, kw_window=100, lock=100,
    )

    # words.shrunk.txt: softmax_output_ww class table, <eps> = background.
    assert ms.words == {0: "<eps>", 1: "ALEXA", 2: "STOP"}

    # Client properties named by kw.cfg overrides default to 0 (unset — no
    # override active until the client pushes a value).
    assert ms.client_prop_defaults == {"AlarmState": 0, "AudioPlayerState": 0, "audio_playback": 0}


@requires_model
def test_real_graph_input_shape_and_states_exist() -> None:
    """Cross-check the loader's names against the actual ONNX graph."""
    import onnx

    ms = load_model_set(MODEL_DIR)
    graph = onnx.load(ms.graph_path, load_external_data=False).graph
    input_names = {i.name for i in graph.input}
    output_names = {o.name for o in graph.output}

    feats = next(i for i in graph.input if i.name == ms.input_name)
    dims = [
        d.dim_value if d.HasField("dim_value") else d.dim_param
        for d in feats.type.tensor_type.shape.dim
    ]
    assert dims[0] == 1 and dims[2] == 64 and dims[3] == 1  # [1, N, 64, 1]
    assert isinstance(dims[1], str) and dims[1]  # N axis is dynamic

    assert set(ms.state_inputs) <= input_names
    assert set(ms.state_outputs) <= output_names
    assert len(ms.state_inputs) == 9 and len(ms.state_outputs) == 9


def test_load_synthetic_minimal_model_set(tmp_path: Path) -> None:
    ms = load_model_set(_write_min_model(tmp_path / "model"))
    assert ms.input_name == "feats"
    assert ms.state_inputs == ["h_in"]
    assert ms.state_outputs == ["h_out"]
    assert ms.fe.fft_pow2 == 512  # 25 ms @ 16 kHz = 400 samples -> next pow2
    assert ms.fe.window == "povey"
    assert [kw.name for kw in ms.keywords] == ["TESTKW"]
    assert ms.keywords[0].overrides == [("TestProp", 1, 0.4)]
    assert ms.words == {0: "<eps>", 1: "TESTKW"}
    assert ms.client_prop_defaults == {"TestProp": 0}


def test_fft_size_without_round_to_power_of_two(tmp_path: Path) -> None:
    config = "\n".join([
        'fe.type = "LFBE"',
        "fe.audio_analysis.frame_length_milli = 25",
        "fe.audio_analysis.frame_shift_milli = 10",
        "fe.audio_analysis.preemphasis_coeff = 0.96875",
        "fe.audio_analysis.remove_dc_offset = 1",
        "fe.audio_analysis.round_to_power_of_two = 0",
        "fe.audio_analysis.samplerate = 16000",
        'fe.audio_analysis.window_type = "povey"',
        "fe.cep.use_energy = 0",
        "fe.mel_fbank.num_bins = 64",
        "fe.mel_fbank.low_freq = 80",
        "fe.mel_fbank.high_freq = 7200",
        'scorer.nemort_config_json_filepath = "nemort.json"',
        'keyword_spotter.config_filepath = "kw.cfg.json"',
        'id2word.mapping_filepath = "words.txt"',
    ]) + "\n"
    ms = load_model_set(_write_min_model(tmp_path / "model", pryon_config=config))
    assert ms.fe.fft_pow2 == 400  # 25 ms @ 16 kHz, no rounding


def test_missing_manifest_names_offending_file(tmp_path: Path) -> None:
    root = _write_min_model(tmp_path / "model")
    (root / "pryon.manifest").unlink()
    with pytest.raises(ModelError, match="pryon.manifest"):
        load_model_set(root)


def test_missing_required_config_key_names_config(tmp_path: Path) -> None:
    root = _write_min_model(tmp_path / "model")
    text = (root / "pryon.config").read_text()
    (root / "pryon.config").write_text(text.replace("fe.mel_fbank.num_bins = 64\n", ""))
    with pytest.raises(ModelError, match=r"pryon\.config.*num_bins"):
        load_model_set(root)


def test_malformed_config_line_names_config(tmp_path: Path) -> None:
    root = _write_min_model(tmp_path / "model")
    with (root / "pryon.config").open("a") as fh:
        fh.write("fe.mel_fbank.num_bins 64\n")  # missing '='
    with pytest.raises(ModelError, match="pryon.config"):
        load_model_set(root)


def test_non_lfbe_frontend_rejected(tmp_path: Path) -> None:
    root = _write_min_model(tmp_path / "model")
    text = (root / "pryon.config").read_text()
    (root / "pryon.config").write_text(text.replace('fe.type = "LFBE"', 'fe.type = "MFCC"'))
    with pytest.raises(ModelError, match=r"pryon\.config.*LFBE"):
        load_model_set(root)


def test_malformed_kw_cfg_names_kw_cfg(tmp_path: Path) -> None:
    root = _write_min_model(tmp_path / "model")
    (root / "kw.cfg.json").write_text("{ not json")
    with pytest.raises(ModelError, match="kw.cfg.json"):
        load_model_set(root)


def test_kw_cfg_missing_accept_threshold_names_kw_cfg(tmp_path: Path) -> None:
    root = _write_min_model(tmp_path / "model")
    data = json.loads((root / "kw.cfg.json").read_text())
    spotter = data["keyword-orchestra"]["keyword-spotters"][0]["spotter"]
    del spotter["classification-thresholds"]["accept-threshold"]
    (root / "kw.cfg.json").write_text(json.dumps(data))
    with pytest.raises(ModelError, match=r"kw\.cfg\.json.*accept-threshold"):
        load_model_set(root)


def test_kw_cfg_multi_property_override_rejected(tmp_path: Path) -> None:
    # KeywordSpec.overrides is a flat (prop, equals, accept) list; an override
    # keyed on several client properties at once has no representation.
    root = _write_min_model(tmp_path / "model")
    data = json.loads((root / "kw.cfg.json").read_text())
    thresholds = data["keyword-orchestra"]["keyword-spotters"][0]["spotter"]["classification-thresholds"]
    thresholds["overrides"][0]["clientProperties"].append({"name": "Other", "equals": 1})
    (root / "kw.cfg.json").write_text(json.dumps(data))
    with pytest.raises(ModelError, match=r"kw\.cfg\.json.*clientPropert"):
        load_model_set(root)


def test_kw_cfg_missing_classifier_window_names_kw_cfg(tmp_path: Path) -> None:
    root = _write_min_model(tmp_path / "model")
    data = json.loads((root / "kw.cfg.json").read_text())
    classifier = data["keyword-orchestra"]["keyword-spotters"][0]["classifier"]
    del classifier["window-size"]
    (root / "kw.cfg.json").write_text(json.dumps(data))
    with pytest.raises(ModelError, match=r"kw\.cfg\.json.*window-size"):
        load_model_set(root)


def test_kw_cfg_non_number_accept_threshold_names_kw_cfg(tmp_path: Path) -> None:
    root = _write_min_model(tmp_path / "model")
    data = json.loads((root / "kw.cfg.json").read_text())
    thresholds = data["keyword-orchestra"]["keyword-spotters"][0]["spotter"]["classification-thresholds"]
    thresholds["accept-threshold"] = "high"
    (root / "kw.cfg.json").write_text(json.dumps(data))
    with pytest.raises(ModelError, match=r"kw\.cfg\.json.*must be a number"):
        load_model_set(root)


def test_nemort_missing_lstm_pair_key_names_nemort_config(tmp_path: Path) -> None:
    root = _write_min_model(tmp_path / "model")
    data = json.loads((root / "nemort.json").read_text())
    del data["lstm_components"][0]["output_component"]
    (root / "nemort.json").write_text(json.dumps(data))
    with pytest.raises(ModelError, match=r"nemort\.json.*output_component"):
        load_model_set(root)


def test_missing_words_file_names_words_file(tmp_path: Path) -> None:
    root = _write_min_model(tmp_path / "model")
    (root / "words.txt").unlink()
    with pytest.raises(ModelError, match=r"words\.txt.*pryon\.config"):
        load_model_set(root)


def test_malformed_words_file_names_words_file(tmp_path: Path) -> None:
    root = _write_min_model(tmp_path / "model")
    (root / "words.txt").write_text("<eps> 0\nTESTKW notanumber\n")
    with pytest.raises(ModelError, match="words.txt"):
        load_model_set(root)


def test_malformed_nemort_config_names_nemort_config(tmp_path: Path) -> None:
    root = _write_min_model(tmp_path / "model")
    (root / "nemort.json").write_text("[1, 2,")
    with pytest.raises(ModelError, match="nemort.json"):
        load_model_set(root)


def test_missing_graph_file_names_graph(tmp_path: Path) -> None:
    root = _write_min_model(tmp_path / "model")
    (root / "model.onnx").unlink()
    with pytest.raises(ModelError, match="model.onnx"):
        load_model_set(root)


def test_graph_missing_declared_input_names_graph(tmp_path: Path) -> None:
    root = _write_min_model(tmp_path / "model")
    data = json.loads((root / "nemort.json").read_text())
    data["model_component_mapping"]["input_map"][0]["component_name"] = "wrong_name"
    (root / "nemort.json").write_text(json.dumps(data))
    with pytest.raises(ModelError, match=r"model\.onnx.*wrong_name"):
        load_model_set(root)


def test_graph_missing_declared_state_names_graph(tmp_path: Path) -> None:
    root = _write_min_model(tmp_path / "model")
    data = json.loads((root / "nemort.json").read_text())
    data["lstm_components"].append({"input_component": "h_extra", "output_component": "h_out2"})
    (root / "nemort.json").write_text(json.dumps(data))
    with pytest.raises(ModelError, match=r"model\.onnx.*h_extra"):
        load_model_set(root)


def test_not_a_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(ModelError, match="not a directory"):
        load_model_set(tmp_path / "nope")
