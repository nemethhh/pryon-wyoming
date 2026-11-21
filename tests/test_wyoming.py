"""Wyoming wake-word server, in-process, on the synthetic model."""
from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pytest
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.event import async_write_event
from wyoming.info import Describe, Info
from wyoming.wake import Detect, Detection, NotDetected

from pypryon import Spotter, __version__
from pypryon.wyoming.models import load_all, parse_thresholds
from pypryon.wyoming.server import PryonWakeHandler, make_info

from conftest import REPO_ROOT
from synthmodel import build_model_dir

NOISE = (np.random.default_rng(5).normal(0, 3000, 16000 * 6)).astype(np.int16)
SILENCE = np.zeros(16000 * 3, np.int16)
LOW = {"HEY": 1e-4}  # below every synthetic HEY score: accepts whenever the lock allows
CHUNK = 1280


@pytest.fixture(scope="module")
def models_dir(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("models")
    build_model_dir(root / "SYNTH")
    return root


async def _serve(models_dir: Path, thresholds=LOW):
    models = load_all(models_dir, thresholds)
    factory = partial(PryonWakeHandler, make_info(models), models)

    async def on_client(reader, writer):
        await factory(reader, writer).run()

    server = await asyncio.start_server(on_client, "127.0.0.1", 0)
    return server, server.sockets[0].getsockname()[1]


def _chunks(pcm: np.ndarray, rate=16000, width=2, channels=1, size=CHUNK):
    for i in range(0, pcm.size, size):
        part = pcm[i:i + size]
        yield AudioChunk(rate=rate, width=width, channels=channels,
                         audio=part.tobytes(), timestamp=(i // channels) * 1000 // rate)


async def _stream(port, pcm, names=None, rate=16000, channels=1, size=CHUNK):
    events = []
    async with AsyncTcpClient("127.0.0.1", port) as client:
        await client.write_event(Detect(names=names).event())
        await client.write_event(AudioStart(rate=rate, width=2, channels=channels).event())
        for chunk in _chunks(pcm, rate=rate, channels=channels, size=size):
            await client.write_event(chunk.event())
        await client.write_event(AudioStop().event())
        while (event := await client.read_event()) is not None:
            events.append(event)
    return events


def _run(coro_fn, models_dir, thresholds=LOW):
    async def main():
        server, port = await _serve(models_dir, thresholds)
        async with server:
            return await coro_fn(port)
    return asyncio.run(main())


def _expected_accept_samples(models_dir: Path) -> list[int]:
    sp = Spotter(models_dir / "SYNTH", thresholds=LOW)
    out = []
    for i in range(0, NOISE.size, CHUNK):
        out += [i for d in sp.feed(NOISE[i:i + CHUNK]) if d.keyword == "HEY" and d.detection_type == 2]
    return out


def test_describe_lists_keywords(models_dir) -> None:
    async def go(port):
        async with AsyncTcpClient("127.0.0.1", port) as client:
            await client.write_event(Describe().event())
            return Info.from_event(await client.read_event())
    info = _run(go, models_dir)
    assert [m.name for m in info.wake[0].models] == ["hey", "stop"]
    assert info.wake[0].models[0].phrase == "Hey"


def test_detection_on_stream(models_dir) -> None:
    events = _run(lambda port: _stream(port, NOISE, names=["hey"]), models_dir)
    detections = [Detection.from_event(e) for e in events if Detection.is_type(e.type)]
    want = _expected_accept_samples(models_dir)
    assert want
    assert [d.name for d in detections] == ["hey"] * len(want)
    assert [d.timestamp for d in detections] == [i * 1000 // 16000 for i in want]
    assert not any(NotDetected.is_type(e.type) for e in events)


def test_not_detected_on_silence(models_dir) -> None:
    events = _run(lambda port: _stream(port, SILENCE, names=["hey"]), models_dir, thresholds={})
    assert [e.type for e in events] == ["not-detected"]


def test_detect_names_filter(models_dir) -> None:
    events = _run(lambda port: _stream(port, NOISE, names=["stop"]), models_dir)
    assert [e.type for e in events] == ["not-detected"]


def test_converts_48k_stereo_audio(models_dir) -> None:
    stereo48 = np.repeat(np.repeat(NOISE, 3)[:, None], 2, axis=1).ravel()
    events = _run(lambda port: _stream(port, stereo48, names=["hey"], rate=48000, channels=2,
                                       size=CHUNK * 6), models_dir)
    got = [Detection.from_event(e).timestamp for e in events if Detection.is_type(e.type)]
    want = [i * 1000 // 16000 for i in _expected_accept_samples(models_dir)]
    assert len(got) == len(want)
    assert all(abs(g - w) <= 80 for g, w in zip(got, want, strict=True))


def test_two_clients_independent_and_disconnect(models_dir) -> None:
    async def go(port):
        async def dropper():
            _, writer = await asyncio.open_connection("127.0.0.1", port)
            await async_write_event(Detect(names=["hey"]).event(), writer)
            await async_write_event(AudioStart(rate=16000, width=2, channels=1).event(), writer)
            for chunk in list(_chunks(NOISE))[:20]:
                await async_write_event(chunk.event(), writer)
            writer.close()
        results = await asyncio.gather(_stream(port, NOISE, names=["hey"]), dropper(),
                                       _stream(port, NOISE, names=["hey"]))
        return results[0], results[2]
    first, second = _run(go, models_dir)
    names = lambda evs: [Detection.from_event(e).timestamp for e in evs if Detection.is_type(e.type)]  # noqa: E731
    assert names(first) == names(second) == \
        [i * 1000 // 16000 for i in _expected_accept_samples(models_dir)]


def test_models_dir_errors(tmp_path, caplog) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(RuntimeError, match=str(empty)):
        load_all(empty, {})
    mixed = tmp_path / "mixed"
    build_model_dir(mixed / "GOOD")
    (mixed / "BROKEN").mkdir(parents=True)
    (mixed / "BROKEN" / "pryon.manifest").write_text("{not json")
    with caplog.at_level(logging.WARNING):
        models = load_all(mixed, {})
    assert [m.dir.name for m in models] == ["GOOD"]
    assert "BROKEN" in caplog.text


def test_threshold_option_parsing(models_dir) -> None:
    assert parse_thresholds(["HEY=0.5", " STOP = 0.7 "]) == {"HEY": 0.5, "STOP": 0.7}
    for bad in ("HEY", "HEY=", "HEY=abc", "HEY=1.5", "HEY=0", "=0.5"):
        with pytest.raises(ValueError, match="threshold"):
            parse_thresholds([bad])
    with pytest.raises(ValueError, match=r"'HAY'.*HEY, STOP"):
        load_all(models_dir, {"HAY": 0.5})


def test_duplicate_keyword_served_once(tmp_path, caplog) -> None:
    build_model_dir(tmp_path / "A")
    build_model_dir(tmp_path / "B", keywords=("HO", "STOP"))
    with caplog.at_level(logging.WARNING):
        models = load_all(tmp_path, {})
    assert [m.keywords for m in models] == [["HEY", "STOP"], ["HO"]]
    assert "STOP" in caplog.text and "B" in caplog.text
    assert [m.name for m in make_info(models).wake[0].models] == ["hey", "stop", "ho"]


def test_default_names_are_each_models_wake_word(models_dir) -> None:
    # STOP scores ~0.92 on silence in the synthetic model, above its 0.55 threshold
    default = _run(lambda port: _stream(port, SILENCE, names=None), models_dir, thresholds={})
    assert [e.type for e in default] == ["not-detected"]
    asked = _run(lambda port: _stream(port, SILENCE, names=["stop"]), models_dir, thresholds={})
    assert {Detection.from_event(e).name for e in asked if Detection.is_type(e.type)} == {"stop"}


def test_cli_version() -> None:
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src"))
    proc = subprocess.run([sys.executable, "-m", "pypryon.wyoming", "--version"],
                          capture_output=True, text=True, env=env)
    assert proc.returncode == 0
    assert proc.stdout.strip() == __version__


def test_davs_dir_served(tmp_path, caplog) -> None:
    from synthmodel import build_davs_model_dir

    build_davs_model_dir(tmp_path / "models" / "DAVS")
    with caplog.at_level(logging.WARNING):
        models = load_all(tmp_path / "models", {})
    assert [m.keywords for m in models] == [["ALEXA", "STOP"]]
    assert "HEY_DISNEY" in caplog.text
    assert [m.name for m in make_info(models).wake[0].models] == ["alexa", "stop"]


def test_davs_wins_shared_keyword(tmp_path, caplog) -> None:
    from synthmodel import build_davs_model_dir

    build_model_dir(tmp_path / "A_OLD", keywords=("ALEXA", "STOP"))
    build_davs_model_dir(tmp_path / "Z_DAVS")
    with caplog.at_level(logging.WARNING):
        models = load_all(tmp_path, {})
    assert [(m.dir.name, m.keywords) for m in models] == [("Z_DAVS", ["ALEXA", "STOP"])]
    assert "A_OLD" in caplog.text


def test_davs_detection_through_server(tmp_path) -> None:
    from synthmodel import build_davs_model_dir

    build_davs_model_dir(tmp_path / "DAVS")
    events = _run(lambda port: _stream(port, NOISE, names=["alexa"]), tmp_path, thresholds={"ALEXA": 0.01})
    names = [Detection.from_event(e).name for e in events if Detection.is_type(e.type)]
    assert names and set(names) == {"alexa"}
