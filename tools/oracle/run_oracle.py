#!/usr/bin/env python3
"""Operator entry point: run the pinned pryon_test oracle and print parsed detections.

Thin wrapper around pypryon.oracle (same behavior as ``python -m pypryon.oracle``),
kept next to build.sh so the oracle tooling has one stable entry:
    tools/oracle/run_oracle.py <clip.raw>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # pypryon repo root

from pypryon.oracle import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
