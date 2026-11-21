"""DAVS keyword rules on hand-made score sequences (rules fitted against the libpryon oracle)."""
from __future__ import annotations

import numpy as np
import pytest

from pypryon.davs import SpotterSpec, WordComponent
from pypryon.rules import ACCEPT, ACCEPT_AUXILIARY, NEAR_MISS, DavsOrchestra, ExponentialDecay, MaxPool

STEP = 1280


def _spot(name="ALEXA", aux=False, comp="det", accept=0.921, notify=0.278, lock=132,
          classifier=None, overrides=()):
    return SpotterSpec(name=name, auxiliary=aux, scorer="ww", components=[WordComponent(1, comp, None, 0, 1)],
                       classifier=classifier or {"type": "moving-average-max-pool", "window-size": 1},
                       accept=accept, notify=None if aux else notify, lock=lock, overrides=list(overrides))


def _rows(det, vrf=None, lanes=4):
    """Per-step component arrays [steps, lanes, 3] with the keyword score in lane 1, class 1."""
    def build(seq):
        arr = np.zeros((len(seq), lanes, 3), np.float32)
        arr[:, 1, 1] = seq
        return arr
    out = {"ww/det": build(det)}
    out["ww/vrf"] = build(vrf if vrf is not None else np.zeros(len(det)))
    return out


def test_maxpool_window1_is_max_over_lanes() -> None:
    row = np.array([[0.1, 0.2, 0.0], [0.0, 0.7, 0.0], [0.0, 0.4, 0.0]], np.float32)
    assert MaxPool(window=1).score(row, class_index=1) == pytest.approx(0.7)


def test_exponential_decay_rate_is_memory() -> None:
    d = ExponentialDecay(rate=0.6)
    assert d.update(1.0) == pytest.approx(0.4)
    assert d.update(1.0) == pytest.approx(0.64)
    d.reset()
    assert d.update(0.5) == pytest.approx(0.2)


def test_accept_at_first_crossing_with_end_sample() -> None:
    orch = DavsOrchestra([_spot(), _spot(aux=True, comp="vrf", accept=0.898, lock=100)])
    det = [0.0] * 17 + [0.920934, 0.973194, 0.98, 0.99]
    events = orch.push(_rows(det))
    assert [(e.keyword, e.detection_type, round(e.score, 6), e.end_sample) for e in events] == [
        ("ALEXA", ACCEPT, 0.973194, 19 * STEP)]


def test_verifier_is_an_alternative_accept() -> None:
    orch = DavsOrchestra([_spot(), _spot(aux=True, comp="vrf", accept=0.898, lock=100)])
    det = [0.0] * 20 + [0.916] * 9
    vrf = [0.0] * 28 + [0.911658]
    events = orch.push(_rows(det, vrf))
    assert [(e.detection_type, round(e.score, 6), e.end_sample) for e in events] == [
        (ACCEPT_AUXILIARY, 0.911658, 29 * STEP)]


def test_accept_locks_the_verifier_too() -> None:
    orch = DavsOrchestra([_spot(), _spot(aux=True, comp="vrf", accept=0.898, lock=100)])
    det = [0.0] * 10 + [0.95] + [0.0] * 5
    vrf = [0.0] * 11 + [0.95] * 5
    assert [e.detection_type for e in orch.push(_rows(det, vrf))] == [ACCEPT]


def test_near_miss_window_and_emission() -> None:
    # run starts at the first notify crossing (t=17); peak over steps 17..24 only
    det = [0.0] * 17 + [0.282, 0.30, 0.45, 0.43, 0.36, 0.51, 0.46, 0.679309, 0.740785, 0.69] + [0.0] * 30
    orch = DavsOrchestra([_spot()])
    events = orch.push(_rows(det))
    assert [(e.detection_type, round(e.score, 6), e.end_sample) for e in events] == [
        (NEAR_MISS, 0.679309, 25 * STEP + 132 * 160)]


def test_accept_cancels_pending_near_miss() -> None:
    det = [0.0] * 5 + [0.5] * 8 + [0.95] + [0.0] * 30
    events = DavsOrchestra([_spot()]).push(_rows(det))
    assert [e.detection_type for e in events] == [ACCEPT]


def test_chunked_push_matches_whole() -> None:
    rng = np.random.default_rng(3)
    det = rng.uniform(0, 1, 200)
    vrf = rng.uniform(0, 1, 200)
    spots = [_spot(), _spot(aux=True, comp="vrf", accept=0.898, lock=100)]
    whole = DavsOrchestra(spots).push(_rows(det, vrf))
    orch = DavsOrchestra(spots)
    parts = []
    rows = _rows(det, vrf)
    for i in range(0, 200, 7):
        parts += orch.push({k: v[i:i + 7] for k, v in rows.items()})
    assert parts == whole


def test_override_lowers_threshold() -> None:
    det = [0.0] * 5 + [0.7] + [0.0] * 3
    spot = _spot(overrides=[("AlarmState", 1, 0.67)])
    assert all(e.detection_type != ACCEPT for e in DavsOrchestra([spot]).push(_rows(det), {"AlarmState": 0}))
    events = DavsOrchestra([spot]).push(_rows(det), {"AlarmState": 1})
    assert [e.detection_type for e in events] == [ACCEPT]


def test_exponential_decay_spotter_accepts() -> None:
    stop = _spot(name="STOP", comp="det", accept=0.6, notify=None, lock=100,
                 classifier={"type": "exponential-decay", "decay-rate": 0.6})
    x = [0.0] * 30 + [0.003, 0.0139, 0.1218, 0.4705, 0.988, 1.0, 0.9999]
    det = np.zeros((len(x), 1, 3), np.float32)
    det[:, 0, 1] = x
    events = DavsOrchestra([stop]).push({"ww/det": det})
    assert [(e.keyword, e.end_sample) for e in events] == [("STOP", 36 * STEP)]
    expected = 0.0
    for v in x[:36]:
        expected = 0.6 * expected + 0.4 * v
    assert events[0].score == pytest.approx(expected, abs=1e-6)   # inputs stored as float32


def test_unsupported_classifier_keyword_skipped(caplog) -> None:
    spots = [_spot(), _spot(name="HEY_DISNEY", classifier={"type": "weighted-geometric-average-aggregator"})]
    orch = DavsOrchestra(spots)
    assert orch.keywords == ["ALEXA"]
    assert "HEY_DISNEY" in caplog.text
