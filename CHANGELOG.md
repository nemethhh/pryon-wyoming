# Changelog

## 0.2.0

- DAVS model sets (Amazon's current wake-word generation: shared encoder plus per-word decoders):
  alexa-en-US with ALEXA and STOP, including the verifier (`AcceptAuxiliary`) and client-property
  overrides. Decisions match libpryon on the parity clips; see the README for known differences.
- A DAVS set wins over a single-graph set that offers the same keyword.
- `tools/make_tts_clips.py`: Piper "Alexa" clips for local parity checks.

## 0.1.0

- `Spotter`: Pryon keyword models from Echo firmware on stock onnxruntime, validated
  against the real libpryon (oracle parity).
- `pypryon-wyoming`: Wyoming wake-word server; Home Assistant add-on (amd64, aarch64).
- `pypryon` CLI: `run` a clip, `convert` a model's graph to standard ONNX.
