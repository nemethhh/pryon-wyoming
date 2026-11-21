# pryon-wyoming

Amazon's own Echo wake-word engine — the **Pryon** keyword models that ship in Echo firmware — running on any
machine with Python and stock [onnxruntime](https://onnxruntime.ai), and served to Home Assistant's Assist as a
[Wyoming](https://github.com/OHF-Voice/wyoming) wake word service.

No Amazon code and no Amazon models are included: you extract the model from firmware of a device you own. The
custom `Hybrid*` kernels of the shipped ONNX graph are rewritten into standard ONNX ops implementing the measured
quantization arithmetic, and the rest of the engine (front end, LSTM lane ring, keyword decoder) is reimplemented
in NumPy. Detections are checked against the real `libpryon.so` (see [How it is validated](#how-it-is-validated)).

## Home Assistant add-on

[![Add repository to Home Assistant](https://my.home-assistant.io/badges/supervisor_add_addon_repository.svg)](https://my.home-assistant.io/redirect/supervisor_add_addon_repository/?repository_url=https%3A%2F%2Fgithub.com%2Fnemethhh%2Fpryon-wyoming)

Or: Settings → Add-ons → Add-on Store → ⋮ → Repositories → `https://github.com/nemethhh/pryon-wyoming`.
Copy your model directory to `/share/pypryon/ALEXA/`, then install **Pryon wake word**; see its
[documentation](pypryon/DOCS.md).

## Other installs

Any machine Home Assistant can reach (Python 3.10+):

```sh
uv tool install "pypryon[zeroconf] @ git+https://github.com/nemethhh/pryon-wyoming"
pypryon-wyoming --models-dir ~/pryon-models --zeroconf      # serves every <dir>/<NAME>/pryon.manifest
```

Home Assistant discovers it with `--zeroconf`; otherwise add the **Wyoming Protocol** integration with this host
and port 10400. Every flag has an environment variable (`PYPRYON_URI`, `PYPRYON_MODELS_DIR`,
`PYPRYON_THRESHOLDS`, `PYPRYON_ZEROCONF`, `PYPRYON_DEBUG`); `--threshold ALEXA=0.88` overrides a keyword's
accept threshold. The add-on image also runs standalone:

```sh
docker run -d -p 10400:10400 -v ~/pryon-models:/models:ro \
  --entrypoint /opt/venv/bin/pypryon-wyoming ghcr.io/nemethhh/amd64-addon-pypryon --models-dir /models
```

## Getting a model directory

Two model generations are supported:

- **Firmware keyword models**: what Echo firmware keeps under
  `/system/local/models/keyword/<locale>/<NAME>/`: `pryon.manifest`, `pryon.config`,
  `nemort_scorer_config.json`, `kw.cfg.json`, `words.shrunk.txt` and
  `runtime_stateless_standard.streaming.fixed.int16.onnx` (other files are ignored). Get it from an update
  image of a device you own: download the OTA package, unpack its payload into partition images (e.g. with
  `payload-dumper-go`), copy the directory out of the system image (`debugfs`, or `mount -o loop,ro`).
  Example: Echo Dot 3rd gen, `ALEXA` (keywords ALEXA, STOP).
- **DAVS sets**: the wake words an Echo downloads (`alexa-en-US`, …): one shared encoder plus per-word
  decoders, laid out as `encoder/`, `alexa-wakeworddecoder/`, `multi-wakeword-wakeworddecoder/`,
  `BDPGeneratedFiles/`, … under a top-level `pryon.manifest`. A rooted Echo keeps them under
  `/data/.../speech/wakeword_models/davs/resources/`. Supported and validated: **alexa-en-US** (ALEXA, STOP).
  The other en-US sets load and use the same rules, but are not validated yet.

Not supported: DAVS keywords with aggregated classifiers (HEY_DISNEY, logged and skipped), TVM-compiled sets,
the CNN-only de-DE sets, and the older Kaldi-FST generation.

## Python library

```python
from pypryon import Spotter

spotter = Spotter("~/pryon-models/ALEXA")          # thresholds={"ALEXA": 0.88} to override
for chunk in mic_chunks:                           # int16, mono, 16 kHz, any chunk size
    for det in spotter.feed(chunk):
        print(det.keyword, det.type_name, round(det.score, 3), det.begin_sample, det.end_sample)
spotter.last_scores                                # per-cycle classifier scores of the last feed
spotter.set_client_property("AlarmState", 1)       # kw.cfg threshold overrides
```

Input is **int16 mono 16 kHz**; a sample rate cannot be detected from raw PCM, so anything else silently shifts
every index and score. Detection types are Accept and NearMiss (a run that peaked above the notify threshold);
scores are kw.cfg classifier scores, so compare them with the model's thresholds (`spotter.thresholds`, 0.923
for ALEXA), not with openWakeWord's habitual 0.5.

Command line:

```
pypryon run <model_dir> <audio.wav|audio.raw> [--scores] [--chunk N]
pypryon convert <model_dir> [-o model.onnx]     # the stock-onnxruntime graph
```

## How it is validated

- **Oracle parity.** `tools/oracle/` runs Amazon's `libpryon.so` itself under `qemu-arm`.
  - Firmware ALEXA: on an espeak "Alexa" clip and synthetic negatives, pypryon produces the same events
    (keyword, type, begin/end sample) with scores within 1e-3 of the library's (events within 1e-2 of a
    threshold are reported, not failed). The parsed oracle runs are committed in `tests/fixtures/`.
  - DAVS alexa-en-US: on 12 Piper TTS "Alexa" clips (`tools/make_tts_clips.py`), espeak and synthetic
    negatives, every decision matches (keyword, Accept / AcceptAuxiliary / NearMiss, end sample, client-property
    overrides). Scores match within 1e-4, except two events that differ by 0.0016 and 0.022; the cause is not
    yet identified, and near a threshold such a difference can flip a decision.
  - The DAVS rules are fitted to the oracle's events, not read out of libpryon. In particular the verifier
    (a second ALEXA spotter that can accept on its own, reported as `AcceptAuxiliary`) is fitted from two
    such events: open corners are both spotters crossing in the same step (pypryon lets the primary win), which
    lock period an auxiliary accept uses (pypryon: the primary's 132 frames), and the verifier's own notify
    threshold (unused).
  - Not reproduced for DAVS sets: exact `begin_sample` (pypryon reports the start of the winning lane's window)
    and the library's 0.1 % "probabilistic accept". Keywords whose rules or decoder layout have not been
    validated are logged at startup and not served (e.g. everything in alexa-de-DE today).
- **Arithmetic parity.** Every rewritten kernel is compared with the vendored NumPy emulation of the device's
  quantized arithmetic (`vendor/pryon-native`), on opsets 11–21.
- **CI without Amazon bytes.** Synthetic models with the same graph structure, exporter quirks and file layout
  (random weights, built at test time) exercise the loader, the graph rewrite, the runner, the CLI and the
  Wyoming server on every push, for both generations.

## Development

```sh
uv sync                                       # dev environment
make test                                     # firmware-free suite
PYPRYON_MODEL_DIR=~/pryon-models/ALEXA PYPRYON_DAVS_MODEL_DIR=~/pryon-models/alexa-en-US make test
make image && PYPRYON_MODEL_DIR=… make smoke  # add-on image, then stream the espeak clip through it
```

`PYPRYON_CLIPS_DIR=~/pryon-clips` points the DAVS parity tests at your own 16 kHz mono recordings instead of
the generated TTS clips (oracle results are cached next to them). The live-oracle tests also need
`PYPRYON_ORACLE_ROOTFS` and a built harness; see
[tools/oracle/README.md](tools/oracle/README.md). `tools/wyoming_client.py HOST PORT clip.wav` streams a WAV to
any Wyoming wake word server.

## Legal and provenance

This repository is code only: no Amazon models, weights, firmware or recordings are part of it, its history or its
images, and the add-on never downloads them. Extract models only from devices you own and do not redistribute
them.

The reverse-engineering record this builds on — front end, quantization rule, LSTM lane contract, decoder rule — is
the pryon-native toolkit by BenStein1 ([echo-dot-assist PR #171](https://github.com/Gamer92000/echo-dot-assist/pull/171),
MIT), vendored unmodified under `vendor/pryon-native/` (see [vendor/PROVENANCE.md](vendor/PROVENANCE.md)).
Original here: the standard-ONNX port of the `Hybrid*` kernels, the NumPy front end, streaming runner and decoder,
the keyword-start derivation, the Wyoming server and the add-on.

"Amazon", "Alexa" and "Echo" are trademarks of Amazon.com, Inc. or its affiliates; this project is not affiliated
with or endorsed by Amazon.

MIT licensed, see [LICENSE](LICENSE).
