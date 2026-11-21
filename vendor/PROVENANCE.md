# Vendor provenance — pryon-native snapshot

`vendor/pryon-native/` is a pinned snapshot of the upstream `tools/pryon-native/`
reverse-engineering toolkit: a PC-side reimplementation of Amazon's Pryon
wake-word engine (`libpryon.so`), reconstructed from measured runtime behaviour.
No Amazon source code or bytes are included (see its `README.md` and `LEDGER.md`).

Facts recorded per the vendoring decision:

- **Upstream change:** PR #171 of the host repository
  (github.com/BenStein1/echo-dot-assist, the pryon-native research branch).
- **Vendored commit:** `dd56117`.
- **Source tarball:** `backups/pryon-native-dd56117.tar.gz` (layout
  `tools/pryon-native/...`, extracted with `--strip-components=1` into
  `vendor/` so the content sits at `vendor/pryon-native/`).
- **Tarball sha256:** `5220c222704aa0458667754f9da8b984822ed7ddb97ba2dd26ea778f27a27801`
  (verified with `sha256sum` before extraction and recording).
- **License:** the host repository is MIT-licensed; that MIT license applies to
  the vendored code (this project's top-level `LICENSE` is the pypryon package's
  own MIT license). Third-party components inside the snapshot are covered by
  `pryon-native/licenses/` (Kaldi, Ne10) and `pryon-native/THIRD_PARTY_NOTICES.md`.
- **Upstream-fix flow:** fixes to vendored code are made upstream first and
  re-vendored as new pinned snapshots. `vendor/pryon-native/` is not edited in
  place.

## Firmware-free test suite

Runs in this project's venv (numpy / onnx / onnxruntime / pytest installed):

    PYTHONPATH=vendor/pryon-native venv/bin/python -m unittest discover -s vendor/pryon-native/tests -p 'test_l*.py' -v

The `-p 'test_l*.py'` pattern selects exactly the firmware-free modules of this
snapshot (`test_lfbe_frontend.py`, `test_lstm_contract.py`) and skips
`test_pryon_runtime_capture.py`, which needs the unpacked firmware image and
`qemu-arm` (it drives the Pryon engine under a debugger) and is not runnable
from this repo.

## Quantizer note (from upstream maintainer review, carried with the pin)

`pryon_quant_model.quantize_activation_rows` (saturating, ±127 clip) is the
authoritative quantizer; the tie-rejecting `quantize_rows` in
`pryon_runtime_analysis.py` is stale — do not port from it.
