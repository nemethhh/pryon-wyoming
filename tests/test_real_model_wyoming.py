"""Wyoming server on the real ALEXA model: the espeak clip wakes it exactly once."""
from __future__ import annotations

import asyncio

import pytest
from wyoming.wake import Detection

from conftest import CLIPS_DIR, DAVS_MODEL_DIR, ESPEAK_WAV, MODEL_DIR, load_wav16, requires_davs, requires_model
from test_wyoming import _serve, _stream


@requires_model
def test_espeak_clip_detects_alexa(tmp_path) -> None:
    (tmp_path / "ALEXA").symlink_to(MODEL_DIR, target_is_directory=True)

    async def main():
        server, port = await _serve(tmp_path, thresholds={})
        async with server:
            return await _stream(port, load_wav16(ESPEAK_WAV), names=["alexa"])

    events = asyncio.run(main())
    detections = [Detection.from_event(e) for e in events if Detection.is_type(e.type)]
    assert [d.name for d in detections] == ["alexa"]


@requires_davs
def test_davs_server_wakes_on_tts_alexa_not_on_espeak(tmp_path) -> None:
    clip = (CLIPS_DIR / "01-lessac-medium.wav") if CLIPS_DIR else None
    if clip is None or not clip.exists():
        pytest.skip("TTS clip 01 absent (tools/make_tts_clips.py)")
    (tmp_path / "alexa-en-US").symlink_to(DAVS_MODEL_DIR, target_is_directory=True)

    async def stream(pcm):
        server, port = await _serve(tmp_path, thresholds={})
        async with server:
            return await _stream(port, pcm, names=["alexa"])

    woke = asyncio.run(stream(load_wav16(clip)))
    assert [Detection.from_event(e).name for e in woke if Detection.is_type(e.type)] == ["alexa"]
    quiet = asyncio.run(stream(load_wav16(ESPEAK_WAV)))
    assert [e.type for e in quiet] == ["not-detected"]
