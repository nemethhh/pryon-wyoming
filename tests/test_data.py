import hashlib
import wave

from conftest import ESPEAK_WAV, REPO_ROOT


def test_espeak_wav_checksum() -> None:
    sums = dict(
        reversed(line.split(maxsplit=1))
        for line in (REPO_ROOT / "tests/data/SHA256SUMS").read_text().splitlines()
    )
    assert hashlib.sha256(ESPEAK_WAV.read_bytes()).hexdigest() == sums["alexa_espeak.wav"]
    with wave.open(str(ESPEAK_WAV)) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2)
        assert w.getnframes() == 134_407
