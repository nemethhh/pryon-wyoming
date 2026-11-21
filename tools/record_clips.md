# Recording parity clips

Real recordings are the final check that pypryon decides like libpryon on a DAVS set. They stay on your
machine: never commit them.

**Format:** 16 kHz, mono, 16-bit WAV, e.g. `arecord -f S16_LE -r 16000 -c 1 alexa-01.wav`, or convert with
`ffmpeg -i in.wav -ar 16000 -ac 1 -sample_fmt s16 out.wav`.

**What to record** (one folder, e.g. `~/pryon-clips/`):

- about 10 clips of 5–10 s: "Alexa" once, then a request ("Alexa, what time is it?"); vary distance to the
  microphone, loudness, and a couple with music or TV in the background;
- 2–3 clips with "Alexa, stop" (checks STOP);
- 2–3 clips of 30 s without the word: talk, TV or music, room silence.

**Run:**

    PYPRYON_CLIPS_DIR=~/pryon-clips PYPRYON_DAVS_MODEL_DIR=~/pryon-models/alexa-en-US \
    PYPRYON_ORACLE_ROOTFS=/path/to/rootfs uv run pytest tests/test_davs_parity.py -s

The oracle's result for each clip is cached in `~/pryon-clips/oracle/`; later runs need no oracle.
