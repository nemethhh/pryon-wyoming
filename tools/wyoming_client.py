"""Stream a WAV to a Wyoming wake-word server: wyoming_client.py HOST PORT WAV [--name NAME]

Prints the server's wake words, then `detection <name> <ms>` per detection or `not-detected`.
"""

import argparse
import asyncio
import wave

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.info import Describe, Info
from wyoming.wake import Detect, Detection, NotDetected


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("host")
    p.add_argument("port", type=int)
    p.add_argument("wav")
    p.add_argument("--name", action="append", help="wake word to detect (default: all)")
    a = p.parse_args()

    async with AsyncTcpClient(a.host, a.port) as client:
        await client.write_event(Describe().event())
        info = Info.from_event(await client.read_event())
        print("wake words:", ", ".join(m.name for prog in info.wake for m in prog.models))

    with wave.open(a.wav, "rb") as w:
        rate, width, channels = w.getframerate(), w.getsampwidth(), w.getnchannels()
        audio = w.readframes(w.getnframes())
    step = 1280 * width * channels
    async with AsyncTcpClient(a.host, a.port) as client:
        await client.write_event(Detect(names=a.name).event())
        await client.write_event(AudioStart(rate=rate, width=width, channels=channels).event())
        for i in range(0, len(audio), step):
            ms = i // (width * channels) * 1000 // rate
            await client.write_event(AudioChunk(rate=rate, width=width, channels=channels,
                                                audio=audio[i:i + step], timestamp=ms).event())
        await client.write_event(AudioStop().event())
        while (event := await client.read_event()) is not None:
            if Detection.is_type(event.type):
                d = Detection.from_event(event)
                print(f"detection {d.name} {d.timestamp}")
            elif NotDetected.is_type(event.type):
                print("not-detected")


if __name__ == "__main__":
    asyncio.run(main())
