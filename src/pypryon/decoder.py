"""kw.cfg.json keyword orchestra: per-cycle scores to Accept / NearMiss events.

Rules (vendor LEDGER §Decoder, measured against libpryon RESULT metadata):

* score(t) = max over lane ages trim_tail+1..25 of a width-``window`` moving
  average of the keyword's softmax; the lane of age a is ring row (t-a+1) % 25;
* Accept on the first cycle past the lock whose score reaches the active
  threshold (client-property overrides may lower it); end frame 6*(t+1);
* NearMiss: the peak of a non-accepting run at or above notify, emitted once
  the stream passes 6*(peak+1) + keyword-feature-window frames;
* begin frame: over the same age window, the age a* whose se-head start
  posterior is largest; begin = 6*(t-a*+1)+1 (fitted to oracle events,
  re-validated by the parity suite).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._constants import CYCLE_FRAMES, LANES
from .modelset import KeywordSpec
from .runner import Posteriors

# first decoded cycle: the young ring needs settled lanes (measured)
START_CYCLE = 5
# fc_output_se column of the utterance-start posterior (kw.cfg's naming is swapped)
SE_START_COLUMN = 1


@dataclass(frozen=True)
class Detection:
    keyword: str
    detection_type: int        # 2 = Accept, 0 = NearMiss
    score: float
    peak_frame: int            # engine cycle of the peak
    begin_frame: int           # LFBE frame
    end_frame: int             # LFBE frame


def _ages(trim_tail: int) -> np.ndarray:
    return np.arange(trim_tail + 1, LANES + 1)


def keyword_begin_frame(se_lanes_row: np.ndarray, t: int, trim_tail: int) -> int:
    ages = _ages(trim_tail)
    priors = se_lanes_row[(t - ages + 1) % LANES, SE_START_COLUMN]
    a = int(ages[int(np.argmax(priors))])
    return CYCLE_FRAMES * (t - a + 1) + 1


def cycle_score(lanes_row: np.ndarray, t: int, class_index: int,
                trim_tail: int, window: int) -> float:
    """Classifier score of one cycle's 25-lane softmax ring (vendor pryon_decoder.cycle_score)."""
    a = lanes_row[(t - _ages(trim_tail) + 1) % LANES, class_index].astype(np.float64)
    n = a.size
    ma = sum(a[i:n - window + 1 + i] for i in range(window)) / window
    return float(ma.max())


class _KeywordState:
    __slots__ = ("spec", "lock_until", "run")

    def __init__(self, spec: KeywordSpec):
        self.spec = spec
        self.lock_until = -1
        self.run: tuple[float, int, int] | None = None   # (peak score, peak cycle, begin frame)


class Orchestra:
    """Streaming decision layer over runner posteriors."""

    def __init__(self, keywords: list[KeywordSpec]):
        self._states = [_KeywordState(spec) for spec in keywords]
        self._cycle = 0

    @staticmethod
    def _active_threshold(spec: KeywordSpec, client_props: dict[str, int]) -> float:
        for prop, equals, accept in spec.overrides:
            if client_props.get(prop, 0) == equals:
                return accept
        return spec.accept

    def push(self, posteriors: Posteriors,
             client_props: dict[str, int] | None = None) -> list[Detection]:
        """Decode one push's posteriors; returns events in cycle order."""
        client_props = client_props or {}
        events: list[Detection] = []
        for row in range(posteriors.lanes.shape[0]):
            t = self._cycle + row
            for st in self._states:
                self._step(st, t, posteriors.lanes[row], posteriors.se_lanes[row],
                           client_props, events)
        self._cycle += posteriors.lanes.shape[0]
        return events

    def _step(self, st: _KeywordState, t: int, lanes_row: np.ndarray,
              se_lanes_row: np.ndarray, client_props: dict[str, int],
              events: list[Detection]) -> None:
        spec = st.spec
        if t < START_CYCLE:
            return
        s = cycle_score(lanes_row, t, spec.class_index, spec.trim_tail, spec.window)
        end_frame = CYCLE_FRAMES * (t + 1)
        if end_frame > st.lock_until and s >= self._active_threshold(spec, client_props):
            events.append(Detection(spec.name, 2, s, t,
                                    keyword_begin_frame(se_lanes_row, t, spec.trim_tail),
                                    end_frame))
            st.lock_until = end_frame + spec.lock
            st.run = None
            return
        if spec.notify is not None and s >= spec.notify and end_frame > st.lock_until:
            if st.run is None or s > st.run[0]:
                st.run = (s, t, keyword_begin_frame(se_lanes_row, t, spec.trim_tail))
        elif st.run is not None and CYCLE_FRAMES * (st.run[1] + 1) + spec.kw_window <= end_frame:
            peak, peak_t, begin = st.run
            events.append(Detection(spec.name, 0, peak, peak_t, begin,
                                    CYCLE_FRAMES * (peak_t + 1) + spec.kw_window))
            st.run = None
