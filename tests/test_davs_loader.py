"""DAVS model-set loader: INCLUDE chains, multi-scorer configs, new nemort schema, merged kw.cfg."""
from __future__ import annotations

from pathlib import Path

import pytest

from pypryon.davs import DavsModelSet, pryon_config
from pypryon.modelset import ModelError, ModelSet, load_model_set

from conftest import DAVS_MODEL_DIR, requires_davs
from synthmodel import build_davs_model_dir, build_model_dir


@pytest.fixture(scope="module")
def davs_dir(tmp_path_factory) -> Path:
    return build_davs_model_dir(tmp_path_factory.mktemp("models") / "DAVS")


def test_include_chain_and_override(tmp_path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "main.config").write_text('x = 1\nINCLUDE "a/inner.config"\nz = "top"\n')
    (tmp_path / "a/inner.config").write_text('x = 2\ny = "rel.json"  # trailing\n')
    cfg = pryon_config(tmp_path / "main.config")
    assert cfg["x"].value == 2 and cfg["z"].value == "top"
    assert cfg["y"].value == "rel.json" and cfg["y"].base == tmp_path / "a"
    assert cfg["y"].path == tmp_path / "a/rel.json"


def test_include_cycle_rejected(tmp_path) -> None:
    (tmp_path / "a.config").write_text('INCLUDE "b.config"\n')
    (tmp_path / "b.config").write_text('INCLUDE "a.config"\n')
    with pytest.raises(ModelError, match="cycle"):
        pryon_config(tmp_path / "a.config")


def test_davs_missing_include_names_file(davs_dir, tmp_path) -> None:
    import shutil
    broken = tmp_path / "BROKEN"
    shutil.copytree(davs_dir, broken)
    (broken / "encoder/pryon.config").unlink()
    with pytest.raises(ModelError, match=r"encoder/pryon\.config"):
        load_model_set(broken)


def test_load_synthetic_davs(davs_dir) -> None:
    ms = load_model_set(davs_dir)
    assert isinstance(ms, DavsModelSet)
    assert ms.fe.num_mel == 64
    assert ms.encoder.frames_per_step == 8 and ms.encoder.inputs[0].source == "audio"
    assert ms.encoder.outputs == {"encoder_embeddings": "encoder_output"}
    assert ms.encoder.state_inputs == ["conv_buffer_input/Concat:2"]
    assert sorted(ms.decoders) == ["ww_alexa", "ww_stop"]          # dvad unreferenced, never opened
    alexa = ms.decoders["ww_alexa"]
    assert alexa.inputs[0].tensor == "encoder_output" and alexa.lstm_states == 4
    names = [(s.name, s.auxiliary) for s in ms.spotters]
    assert names == [("STOP", False), ("HEY_DISNEY", False), ("ALEXA", False), ("ALEXA", True)]
    primary = ms.spotters[2]
    assert (primary.accept, primary.min_accept, primary.max_accept, primary.notify) == (0.921, 0.845, 0.929, 0.278)
    assert primary.lock == 132 and primary.scorer == "ww_alexa"
    assert primary.components[0].component == "ww_det_scores" and primary.components[0].class_index == 1
    assert primary.overrides[0] == ("AlarmState", 1, 0.67)
    assert ms.spotters[3].components[0].component == "ww_vrf_scores"
    assert ms.spotters[0].classifier == {"type": "exponential-decay", "decay-rate": 0.6}
    assert ms.words == {0: "<eps>", 1: "ALEXA", 2: "HEY_DISNEY", 3: "STOP"}
    assert ms.client_prop_defaults == {"AlarmState": 0, "AudioPlayerState": 0}


def test_old_model_still_loads_as_modelset(tmp_path) -> None:
    assert isinstance(load_model_set(build_model_dir(tmp_path / "OLD")), ModelSet)


@requires_davs
def test_load_real_davs() -> None:
    ms = load_model_set(DAVS_MODEL_DIR)
    assert ms.encoder.frames_per_step == 8 and ms.encoder.feat_dim == 64
    assert sorted(ms.decoders) == ["ww_alexa", "ww_stop"]
    assert ms.decoders["ww_alexa"].lstm_states == 4
    assert len(ms.decoders["ww_stop"].state_inputs) == 4
    primary, aux = (s for s in ms.spotters if s.name == "ALEXA")
    assert (primary.accept, primary.min_accept, primary.max_accept, primary.lock) == (0.921, 0.845, 0.929, 132)
    assert (aux.accept, aux.auxiliary, aux.lock) == (0.898, True, 100)


def _mutate_kw(src: Path, dst: Path, edit) -> Path:
    import json
    import shutil
    shutil.copytree(src, dst)
    kw = dst / "BDPGeneratedFiles/kw.cfg.json"
    data = json.loads(kw.read_text())
    edit(data["keyword-orchestra"]["keyword-spotters"])
    kw.write_text(json.dumps(data))
    return dst


def _alexa(spotters) -> dict:
    return next(s for s in spotters if s["name"] == "ALEXA" and "spotter-type" not in s["spotter"])


@pytest.mark.parametrize("edit", [
    lambda sp: _alexa(sp)["spotter"]["word-component-array"][0]["scorer-output-component"].update(name="nope"),
    lambda sp: _alexa(sp)["spotter"]["word-component-array"][0].update({"scorer-index": 9}),
    lambda sp: _alexa(sp)["classifier"].update({"window-size": 4}),
])
def test_unservable_keyword_not_served_never_crashes(davs_dir, tmp_path, edit, caplog) -> None:
    import logging

    import numpy as np

    from pypryon import Spotter
    d = _mutate_kw(davs_dir, tmp_path / "BAD", edit)
    with caplog.at_level(logging.WARNING):
        sp = Spotter(d)
    assert sp.keywords == ["STOP"]
    sp.feed(np.random.default_rng(0).normal(0, 3000, 16000).astype(np.int16))


def test_lanes_first_decoder_not_served(tmp_path) -> None:
    from pypryon import Spotter

    from synthmodel import build_davs_model_dir
    d = build_davs_model_dir(tmp_path / "LF", lanes_first=True)
    assert Spotter(d).keywords == ["STOP"]


@pytest.mark.parametrize(("field", "value"), [("lock-period", "abc"), ("keyword-feature-window-length", None)])
def test_malformed_spotter_number_is_model_error(davs_dir, tmp_path, field, value) -> None:
    d = _mutate_kw(davs_dir, tmp_path / "NUM", lambda sp: _alexa(sp)["spotter"].update({field: value}))
    with pytest.raises(ModelError, match=field):
        load_model_set(d)


def test_decoder_reading_unpublished_tensor_is_model_error(davs_dir, tmp_path) -> None:
    import json
    import shutil
    d = tmp_path / "ART"
    shutil.copytree(davs_dir, d)
    cfg = d / "multi-wakeword-wakeworddecoder/nemort_scorer_config_decoder_stop.json"
    data = json.loads(cfg.read_text())
    data["inputs"][0]["tensor_name"] = "not_published"
    cfg.write_text(json.dumps(data))
    with pytest.raises(ModelError, match="not_published"):
        load_model_set(d)
