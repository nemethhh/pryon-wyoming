# Pryon wake word

Amazon's own Echo wake-word engine (Pryon), as a [Wyoming](https://github.com/OHF-Voice/wyoming)
wake word service for Home Assistant's Assist. It runs the keyword models shipped in Echo firmware,
the same ones the Echo uses, on the Home Assistant machine.

**No model is included, and none is ever downloaded.** You extract the model from firmware of a device you own
and copy it to Home Assistant (see below). Without one, the add-on does not start.

## Setup

1. **Copy a model directory** to `/share/pypryon/<NAME>/`, e.g. `/share/pypryon/ALEXA/`, with the Samba
   or Advanced SSH add-on. It must contain at least:
   `pryon.manifest`, `pryon.config`, `nemort_scorer_config.json`, `kw.cfg.json`, `words.shrunk.txt` and the
   `.onnx` file named in `nemort_scorer_config.json`.
   In Echo firmware they live in `/system/local/models/keyword/<locale>/<NAME>/` (see the
   [README](https://github.com/nemethhh/pryon-wyoming#getting-a-model-directory)).
   Each directory under `/share/pypryon` is loaded; every keyword in it becomes a wake word
   (the ALEXA directory offers `alexa` and `stop`). A keyword found in two directories is served from the
   first one only.
2. **Install and start the add-on.** Home Assistant finds it: accept the **Wyoming Protocol** integration it
   offers (Settings → Devices & services).
3. **Pick the wake word**: Settings → Voice assistants → your assistant → **Wake word**, engine `pypryon`,
   wake word `alexa`. For a voice satellite, choose the wake word in the satellite's device settings
   (only for satellites that stream audio to Home Assistant for wake word detection). A client that names no
   wake word gets each directory's first one (`alexa`, not `stop`).

## Options

| option | default | |
|---|---|---|
| `models_dir` | `/share/pypryon` | Directory holding one subdirectory per model. |
| `thresholds` | none | Per-keyword accept threshold, `KEYWORD=VALUE` with 0 < VALUE ≤ 1, e.g. `ALEXA=0.88`. Lower wakes more easily (and falsely more often). Without it the model's own threshold is used (0.923 for ALEXA). |
| `debug_logging` | false | Debug-level logging, including unexpected protocol events. |

## Troubleshooting

- **The add-on stops at start with `no Pryon model found under /share/pypryon`**: the model directory is missing
  or misplaced; the log names the path it searched. The directory must hold `pryon.manifest` directly.
- **`Skipping /share/pypryon/X: ...`**: that directory is not a model this version can load (for example a DAVS
  multi-model download); the others still load.
- Detections are logged at the default level as `Detected alexa (score 0.937)`.

## Legal

This add-on is code only. Amazon's models are Amazon's: extract them only from devices you own, and do not
share them.
