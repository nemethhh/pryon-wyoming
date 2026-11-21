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
