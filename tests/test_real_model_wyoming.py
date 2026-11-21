"""Wyoming server on the real ALEXA model: the espeak clip wakes it exactly once."""
from __future__ import annotations

import asyncio

from wyoming.wake import Detection

from conftest import ESPEAK_WAV, MODEL_DIR, load_wav16, requires_model
from test_wyoming import _serve, _stream

pytestmark = requires_model


def test_espeak_clip_detects_alexa(tmp_path) -> None:
    (tmp_path / "ALEXA").symlink_to(MODEL_DIR, target_is_directory=True)

    async def main():
        server, port = await _serve(tmp_path, thresholds={})
        async with server:
            return await _stream(port, load_wav16(ESPEAK_WAV), names=["alexa"])

    events = asyncio.run(main())
    detections = [Detection.from_event(e) for e in events if Detection.is_type(e.type)]
    assert [d.name for d in detections] == ["alexa"]
