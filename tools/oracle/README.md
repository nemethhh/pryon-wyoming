# Oracle (development only)

The reference pypryon is validated against: Amazon's own `libpryon.so`, run
under `qemu-arm` through a small C harness (`pryon_test.c`). Nothing here is
part of the installed package, and nothing here ships any Amazon bytes; you
need your own unpacked Echo system image.

## Build

`build.sh` (with `ANDROID_NDK_ROOT` set to Android NDK r21e) cross-compiles `pryon_test.c` against the
rootfs's `libpryon.so` and writes `build/oracle/pryon_test`
(see the header of `Makefile` for the pinned recipe).

## Run

    export PYPRYON_ORACLE_ROOTFS=/path/to/unpacked/rootfs   # contains system/lib/libpryon.so
    python tools/oracle/pryon_oracle.py clip.raw            # s16le mono 16 kHz

Needs `qemu-arm` and `unshare` on `PATH`.

## Regenerating test fixtures

`tests/fixtures/*_oracle.json` are parsed oracle runs on the espeak clip and
the synthetic clips from `tools/make_clips.py`. With `PYPRYON_MODEL_DIR` and
`PYPRYON_ORACLE_ROOTFS` set, `pytest tests/test_oracle_parity.py` checks that
the live oracle still reproduces them and that pypryon matches it.
