# Vendor provenance — pryon-native snapshot

`vendor/pryon-native/` is a pinned snapshot of the upstream `tools/pryon-native/`
reverse-engineering toolkit: a PC-side reimplementation of Amazon's Pryon
wake-word engine (`libpryon.so`), reconstructed from measured runtime behaviour.
No Amazon source code or bytes are included (see its `README.md` and `LEDGER.md`).

Facts recorded per the vendoring decision:

- **Upstream:** <https://github.com/Gamer92000/echo-dot-assist> (MIT),
  [pull request #171](https://github.com/Gamer92000/echo-dot-assist/pull/171) by BenStein1
  ("Document and reproduce Pryon's native inference pipeline"), directory `tools/pryon-native/`.
- **Vendored commit:** `dd56117`, copied unmodified to `vendor/pryon-native/`.
- **Snapshot tarball sha256:** `5220c222704aa0458667754f9da8b984822ed7ddb97ba2dd26ea778f27a27801`
  (a tarball of that directory at `dd56117`, verified before extraction).
- **License:** the host repository is MIT-licensed; that MIT license applies to
  the vendored code (this project's top-level `LICENSE` is the pypryon package's
  own MIT license). Third-party components inside the snapshot are covered by
  `pryon-native/licenses/` (Kaldi, Ne10) and `pryon-native/THIRD_PARTY_NOTICES.md`.
- **Upstream-fix flow:** fixes to vendored code are made upstream first and
  re-vendored as new pinned snapshots. `vendor/pryon-native/` is not edited in
  place.

## Firmware-free test suite

With the project's dev environment (`uv sync`):

    PYTHONPATH=vendor/pryon-native uv run python -m unittest discover -s vendor/pryon-native/tests -p 'test_l*.py' -v

The `-p 'test_l*.py'` pattern selects exactly the firmware-free modules of this
snapshot (`test_lfbe_frontend.py`, `test_lstm_contract.py`) and skips
`test_pryon_runtime_capture.py`, which needs the unpacked firmware image and
`qemu-arm` (it drives the Pryon engine under a debugger).

## Quantizer note (from upstream maintainer review, carried with the pin)

`pryon_quant_model.quantize_activation_rows` (saturating, ±127 clip) is the
authoritative quantizer; the tie-rejecting `quantize_rows` in
`pryon_runtime_analysis.py` is stale — do not port from it.
