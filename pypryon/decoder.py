"""Key-phrase decoder: kw.cfg.json orchestra semantics.

Mirrors vendor pryon_decoder.py (measured against stock RESULT metadata —
LEDGER §"Decoder: score and accept/near-miss rule") on the streaming side:

* per engine cycle t, score(t) = max over age-windows of the 25-lane
  softmax ring: ages 9..25 (trim-tail 8 youngest dropped), width-3 moving
  average, lane of age a read from ring row (t - a + 1) % 25;
* accept on the first cycle outside the 100-frame lock whose score is ≥ the
  active threshold (client-property overrides may lower it); the accept's
  end frame is 6·(t+1) LFBE frames (= 960·(t+1) samples);
* near-miss: the peak of a non-accepting run ≥ notify, emitted at the first
  later cycle whose end frame passes 6·(peak+1) + keyword-feature-window;
* detection_type 2 = Accept, 0 = NearMiss (types 1/3 are unknown to the
  vendor rule — never synthesized here).

Detection.begin_frame derives from the fc_output_se head (the vendor's
underived kwSampleStartIndex): at the accept/peak cycle t, over the same
age window the score rule uses (trim-tail+1..25), take the age whose se
posterior is maximal (column 1 of fc_output_se — empirically the utterance
start column; kw.cfg's scorer-index naming is swapped against observed
behavior) and set begin_frame = 6*(t - a* + 1) + 1.  Fitted on the two
live-oracle espeak events (begin 22240 = 960*23+160, 61760 = 960*64+160)
and re-validated by Task 11's parity suite; end_frame stays the vendor
rule (6*(t+1) for accepts, +keyword-feature-window for near-misses).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .modelset import KeywordSpec
from .runner import Posteriors

# Cycles before the first decoded one (vendor decode default start=5: the
# young ring needs settled lanes — measured, not from kw.cfg).
START_CYCLE = 5


@dataclass(frozen=True)
class Detection:
    keyword: str
    detection_type: int        # 2 = Accept, 0 = NearMiss
    score: float
    peak_frame: int            # engine cycle of the peak
    begin_frame: int | None    # LFBE frame; None until the se derivation
    end_frame: int             # LFBE frame (see module docstring)


SE_START_COLUMN = 1   # fc_output_se column of the utterance-start posterior


def keyword_begin_frame(se_lanes_row: np.ndarray, t: int, trim_tail: int) -> int:
    """kwSampleStartIndex rule: age-argmax of the start posterior over the
    score rule's age window, mapped to the age's absolute cycle (+1 frame,
    the engine's frame phase — see module docstring provenance)."""
    ages = np.arange(trim_tail + 1, 26)
    priors = se_lanes_row[(t - ages + 1) % 25, SE_START_COLUMN]
    a = ages[int(np.argmax(priors))]
    return 6 * (t - a + 1) + 1


def cycle_score(lanes_row: np.ndarray, t: int, class_index: int,
                trim_tail: int, window: int) -> float:
    """Score of one cycle's 25-lane ring row (vendor pryon_decoder.cycle_score).

    Ages trim_tail+1..25, lane of age a at ring row (t - a + 1) % 25;
    width-`window` moving average over consecutive ages, then max.
    """
    ages = np.arange(trim_tail + 1, 26)
    ring = np.arange(25)
    a = lanes_row[(t - ages + 1) % 25, class_index].astype(np.float64)
    n = a.size
    ma = sum(a[i:n - window + 1 + i] for i in range(window)) / window
    return float(ma.max())


class _KeywordState:
    __slots__ = ("spec", "lock_until", "run")

    def __init__(self, spec: KeywordSpec):
        self.spec = spec
        self.lock_until = -1
        self.run: tuple[float, int] | None = None   # (peak score, peak cycle)


class Orchestra:
    """Streaming kw.cfg decision layer over runner posteriors."""

    def __init__(self, keywords: list[KeywordSpec], buffer: int = 200):
        # `buffer` is kw.cfg's orchestra buffer-size; this rule reads only
        # the current cycle's ring (history lives in the lanes), so it is
        # accepted for interface fidelity and reserved for the se head.
        self._buffer = buffer
        self._states = [_KeywordState(spec) for spec in keywords]
        self._cycle = 0

    def _active_threshold(self, spec: KeywordSpec,
                          client_props: dict[str, int]) -> float:
        for prop, equals, accept in spec.overrides:
            if client_props.get(prop, 0) == equals:
                return accept
        return spec.accept

    def push(self, posteriors: Posteriors,
             client_props: dict[str, int] | None = None) -> list[Detection]:
        """Decode one push's posteriors; returns events in cycle order."""
        client_props = client_props or {}
        events: list[Detection] = []
        lanes = posteriors.lanes
        se_lanes = posteriors.se_lanes
        if se_lanes is None:
            se_lanes = np.zeros(lanes.shape[:2] + (2,), np.float32)
        for row in range(lanes.shape[0]):
            t = self._cycle + row
            for st in self._states:
                self._step(st, t, lanes[row], se_lanes[row], client_props, events)
        self._cycle += lanes.shape[0]
        return events

    def _step(self, st: _KeywordState, t: int, lanes_row: np.ndarray,
              se_lanes_row: np.ndarray, client_props: dict[str, int],
              events: list[Detection]) -> None:
        spec = st.spec
        if t < START_CYCLE:
            return
        s = cycle_score(lanes_row, t, spec.class_index, spec.trim_tail,
                        spec.window)
        end_frame = 6 * (t + 1)
        if end_frame > st.lock_until and s >= self._active_threshold(spec, client_props):
            events.append(Detection(
                spec.name, 2, s, t,
                keyword_begin_frame(se_lanes_row, t, spec.trim_tail),
                end_frame))
            st.lock_until = end_frame + spec.lock
            st.run = None
            return
        notify = spec.notify
        if notify is not None and s >= notify and end_frame > st.lock_until:
            if st.run is None or s > st.run[0]:
                begin = keyword_begin_frame(se_lanes_row, t, spec.trim_tail)
                st.run = (s, t, begin)
        elif st.run is not None and 6 * (st.run[1] + 1) + spec.kw_window <= end_frame:
            events.append(Detection(
                spec.name, 0, st.run[0], st.run[1], st.run[2],
                6 * (st.run[1] + 1) + spec.kw_window))
            st.run = None
