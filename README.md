# pypryon

Run Amazon Pryon wake-word models on stock onnxruntime.

Status: scaffold (Task 1) — the model runner lands in later tasks.

## Vendored substrate

`vendor/pryon-native/` is a pinned snapshot of the pryon-native
reverse-engineering toolkit (upstream PR #171, commit `dd56117`, MIT).
See [vendor/PROVENANCE.md](vendor/PROVENANCE.md) for provenance, licenses,
and the firmware-free test command. Vendored code is not modified locally;
fixes flow back upstream as PRs.

## Development

    python3 -m venv venv
    venv/bin/pip install numpy onnx onnxruntime pytest

Firmware-free vendored suite:

    PYTHONPATH=vendor/pryon-native venv/bin/python -m unittest discover -s vendor/pryon-native/tests -p 'test_l*.py' -v
