"""Wyoming event handler: audio in, wake-word detections out."""
from __future__ import annotations

import asyncio
import logging

import numpy as np
from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Attribution, Describe, Info, WakeModel, WakeProgram
from wyoming.server import AsyncEventHandler
from wyoming.wake import Detect, Detection, NotDetected

from .. import __version__
from .._constants import SAMPLE_RATE
from ..spotter import Spotter
from .models import LoadedModel

_LOGGER = logging.getLogger(__name__)
_ATTRIBUTION = Attribution(name="pypryon", url="https://github.com/nemethhh/pryon-wyoming")


def make_info(models: list[LoadedModel]) -> Info:
    wake_models = [
        WakeModel(name=kw.lower(), attribution=_ATTRIBUTION, installed=True,
                  description=f"{kw.title()} (Pryon)", version=None, languages=[],
                  phrase=kw.title())
        for model in models for kw in model.keywords]
    return Info(wake=[WakeProgram(
        name="pypryon", attribution=_ATTRIBUTION, installed=True,
        description="Amazon Pryon wake words, from models you extract yourself",
        version=__version__, models=wake_models)])


class PryonWakeHandler(AsyncEventHandler):
    """One client connection: per-connection Spotters over shared model sessions."""

    def __init__(self, info: Info, models: list[LoadedModel], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._info = info
        self._models = models
        self._all_names = {kw.lower() for m in models for kw in m.keywords}
        # with no names requested, only each model's wake word (e.g. not ALEXA's STOP)
        self._default_names = {m.default.lower() for m in models if m.default}
        self._converter = AudioChunkConverter(rate=SAMPLE_RATE, width=2, channels=1)
        self._names: set[str] = set(self._default_names)
        self._spotters: list[Spotter] | None = None
        self._detected = False
        self._samples = 0

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self._info.event())
            return True
        if Detect.is_type(event.type):
            self._select(Detect.from_event(event).names)
        elif AudioStart.is_type(event.type):
            self._start()
        elif AudioChunk.is_type(event.type):
            if self._spotters is None:
                self._start()
            await self._feed(AudioChunk.from_event(event))
        elif AudioStop.is_type(event.type):
            if not self._detected:
                await self.write_event(NotDetected().event())
            return False
        else:
            _LOGGER.debug("Unexpected event: %s", event.type)
        return True

    def _select(self, names: list[str] | None) -> None:
        if not names:
            self._names = set(self._default_names)
            return
        wanted = {n.lower() for n in names}
        unknown = wanted - self._all_names
        if unknown:
            _LOGGER.warning("Unknown wake word(s) requested: %s", ", ".join(sorted(unknown)))
        self._names = wanted & self._all_names

    def _start(self) -> None:
        self._spotters = [Spotter(m.model_set, thresholds=m.thresholds, session=m.session)
                          for m in self._models]
        self._detected = False
        self._samples = 0

    async def _feed(self, chunk: AudioChunk) -> None:
        chunk = self._converter.convert(chunk)
        pcm = np.frombuffer(chunk.audio, "<i2").astype(np.int16)
        if pcm.size == 0:
            return
        timestamp = chunk.timestamp
        if timestamp is None:
            timestamp = self._samples * 1000 // SAMPLE_RATE
        self._samples += pcm.size
        for model, spotter in zip(self._models, self._spotters, strict=True):
            for det in await asyncio.to_thread(spotter.feed, pcm):
                name = det.keyword.lower()
                if not det.is_accept or det.keyword not in model.keywords \
                        or name not in self._names:
                    continue
                self._detected = True
                _LOGGER.info("Detected %s (score %.3f)", name, det.score)
                await self.write_event(Detection(name=name, timestamp=timestamp).event())
