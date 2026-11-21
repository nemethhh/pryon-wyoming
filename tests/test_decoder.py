"""Key-phrase decoder tests (Task 9).

Semantics per vendor pryon_decoder.py + kw.cfg.json: per cycle t the score
is the max over age-windows (trim-tail 8 youngest ages dropped, width-3
moving average) of the 25-lane softmax ring; accept at/above the (possibly
client-property-overridden) active threshold outside the 100-frame lock;
near-miss = peak of a non-accepting run at/above notify, emitted once the
keyword feature window has elapsed past the peak; detection_type 2=Accept,
0=NearMiss.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from pypryon.decoder import Detection, Orchestra
from pypryon.modelset import KeywordSpec, load_model_set

from conftest import ESPEAK_WAV, MODEL_DIR, REPO_ROOT, load_wav16, requires_model

ALEXA = KeywordSpec(
    name="ALEXA", class_index=1, scorer_index_start=0, scorer_index_end=1,
    accept=0.923, notify=0.69,
    overrides=[("AlarmState", 1, 0.834)],
    window=3, trim_tail=8, kw_window=100, lock=100,
)
STOP = KeywordSpec(
    name="STOP", class_index=2, scorer_index_start=0, scorer_index_end=1,
    accept=0.55, notify=None,
    overrides=[], window=3, trim_tail=8, kw_window=100, lock=100,
)


def lanes_from(class_scores: dict[int, np.ndarray], t: int = 40) -> np.ndarray:
    """Build a [t, 25, 3] posterior ring from per-class per-cycle scores.

    class_scores maps class index -> (t,) array broadcast to every lane
    (uniform lanes make the age-window max equal the raw score).
    """
    p = np.full((t, 25, 3), 0.02, np.float32)
    for k, v in class_scores.items():
        p[:, :, k] = np.asarray(v, np.float32)[:, None]
    return p


def posteriors(lanes: np.ndarray):
    from pypryon.runner import Posteriors
    return Posteriors(ww=lanes.max(1), se=np.zeros((lanes.shape[0], 2), np.float32),
                      lanes=lanes, se_lanes=np.zeros(lanes.shape[:2] + (2,), np.float32))


def test_accept_crossing_threshold() -> None:
    scores = np.full(40, 0.05)
    scores[5:12] = 0.95
    orch = Orchestra([ALEXA])
    events = orch.push(posteriors(lanes_from({1: scores})), {})
    accepts = [e for e in events if e.detection_type == 2]
    assert len(accepts) == 1
    e = accepts[0]
    assert e.keyword == "ALEXA"
    assert e.score >= 0.923
    assert e.peak_frame == 5
    assert e.end_frame == 6 * 6          # 960 * (t+1) in samples / 160


def test_near_miss_delayed_by_feature_window() -> None:
    scores = np.full(40, 0.05)
    scores[5:11] = 0.80                  # peak run, then drops below notify
    orch = Orchestra([ALEXA])
    events = orch.push(posteriors(lanes_from({1: scores})), {})
    near = [e for e in events if e.detection_type == 0]
    assert len(near) == 1
    e = near[0]
    assert e.peak_frame == 5
    assert e.score == pytest.approx(0.80, abs=1e-6)
    # emitted once 6*(peak+1) + 100 frames elapse: cycle 22 (end 138 ≥ 136)
    assert e.end_frame == 6 * 6 + 100


def test_lock_suppresses_second_accept() -> None:
    scores = np.full(40, 0.95)           # above threshold the whole time
    orch = Orchestra([ALEXA])
    events = orch.push(posteriors(lanes_from({1: scores})), {})
    accepts = [e for e in events if e.detection_type == 2]
    # accept at 5 (lock until frame 136), 22 (end 138 > 136; lock to 238),
    # 39 (end 240 > 238) — one accept per lock period
    assert [e.peak_frame for e in accepts] == [5, 22, 39]


def test_client_property_override_lowers_threshold() -> None:
    scores = np.full(40, 0.05)
    scores[5:12] = 0.85                  # between 0.834 and 0.923
    lanes = lanes_from({1: scores})
    base = Orchestra([ALEXA]).push(posteriors(lanes), {})
    assert [e.detection_type for e in base] == [0]      # near-miss at 0.923
    alarmed = Orchestra([ALEXA]).push(posteriors(lanes), {"AlarmState": 1})
    assert [e.detection_type for e in alarmed] == [2]   # accept at 0.834
    assert alarmed[0].score == pytest.approx(0.85, abs=1e-6)


def test_two_keywords_independent() -> None:
    alexa = np.full(40, 0.05)
    alexa[5:9] = 0.95
    stop = np.full(40, 0.02)
    stop[6:10] = 0.96
    orch = Orchestra([ALEXA, STOP])
    events = orch.push(posteriors(lanes_from({1: alexa, 2: stop})), {})
    accepts = {(e.keyword, e.peak_frame) for e in events if e.detection_type == 2}
    assert ("ALEXA", 5) in accepts
    assert ("STOP", 6) in accepts


def test_stop_has_no_near_miss() -> None:
    scores = np.full(40, 0.05)
    scores[5:11] = 0.545                 # above any notify, below STOP accept
    events = Orchestra([STOP]).push(posteriors(lanes_from({2: scores})), {})
    assert events == []                  # notify is None: never a near-miss


def test_constant_mid_stream_never_emits() -> None:
    events = Orchestra([ALEXA]).push(posteriors(lanes_from({1: np.full(40, 0.5)})), {})
    assert events == []


def test_streaming_pushes_match_batch() -> None:
    scores = np.full(60, 0.05)
    scores[5:12] = 0.95
    scores[30:40] = 0.9                  # second accept outside the lock
    lanes = lanes_from({1: scores}, t=60)
    whole = Orchestra([ALEXA]).push(posteriors(lanes), {})
    parts = []
    orch = Orchestra([ALEXA])
    for i in range(0, 60, 7):
        parts += orch.push(posteriors(lanes[i:i + 7]), {})
    assert parts == whole


@requires_model
def test_parity_vs_vendor_decoder() -> None:
    os.environ["PRYON_MODEL"] = str(MODEL_DIR / "runtime_stateless_standard.streaming.fixed.int16.onnx")
    sys.path.insert(0, str(REPO_ROOT / "vendor/pryon-native"))
    from pryon_decoder import decode as vendor_decode
    from pryon_quant_model import Model

    from pypryon.fe import lfbe_frames
    from pypryon.runner import GraphRunner

    pcm = load_wav16(ESPEAK_WAV)
    frames = lfbe_frames(pcm, load_model_set(MODEL_DIR).fe)
    got = GraphRunner(load_model_set(MODEL_DIR)).push_frames(frames)
    lanes = got.lanes

    want = list(vendor_decode(lanes, k=1))
    specs = load_model_set(MODEL_DIR).keywords
    mine = Orchestra(specs).push(got, {})

    alexa = [e for e in mine if e.keyword == "ALEXA"]
    assert len(alexa) == len(want)
    for evt, (vtype, vscore, _vsample, vcycle) in zip(alexa, want):
        assert evt.detection_type == vtype
        assert evt.score == pytest.approx(vscore, abs=1e-12)
        assert evt.peak_frame == vcycle
