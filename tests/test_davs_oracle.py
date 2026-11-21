"""The live oracle on the DAVS alexa-en-US set still reproduces the committed fixtures."""
from __future__ import annotations

import json
import subprocess
import sys
import wave

from conftest import DAVS_MODEL_DIR, ESPEAK_WAV, REPO_ROOT, requires_davs_oracle

FIXTURES = REPO_ROOT / "tests/fixtures/davs/alexa-en-US"
CLIPS = REPO_ROOT / "tests/clips"


@requires_davs_oracle
def test_live_oracle_reproduces_davs_fixtures(tmp_path) -> None:
    from pryon_oracle import run_oracle

    subprocess.run([sys.executable, str(REPO_ROOT / "tools/make_clips.py"), "--out", str(CLIPS)],
                   check=True, capture_output=True)
    espeak = tmp_path / "alexa_espeak.raw"
    with wave.open(str(ESPEAK_WAV)) as w:
        espeak.write_bytes(w.readframes(w.getnframes()))
    clips = {"alexa_espeak": espeak,
             **{n: CLIPS / f"{n}.raw" for n in ("silence_30s", "sine_30s", "noise_30s", "hum_30s")}}
    for name, raw in clips.items():
        want = json.loads((FIXTURES / f"{name}_oracle.json").read_text())
        assert run_oracle(raw, DAVS_MODEL_DIR) == want, name
