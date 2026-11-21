"""pypryon command line.

  pypryon run <model_dir> <audio.(raw|wav)> [--scores]
  pypryon convert <model_dir>

`run` feeds 16 kHz mono s16 audio (raw, or a wav that must be 16 kHz mono
16-bit) through a Spotter and prints one line per detection.  `convert`
writes the portable ONNX conversion to .pypryon-cache/ under the current
directory (never next to the model — model dirs are extracted firmware and
often read-only).
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import wave
from pathlib import Path

import numpy as np

from .graph import make_portable
from .spotter import Spotter

CHUNK = 1280  # 80 ms feed cadence


def _load_pcm(path: Path) -> np.ndarray:
    if path.suffix == ".raw":
        return np.fromfile(path, "<i2")
    with wave.open(str(path), "rb") as w:
        if w.getframerate() != 16000:
            print(f"error: {path.name} is {w.getframerate()} Hz — pypryon takes "
                  "16 kHz mono s16 only (resample first)", file=sys.stderr)
            raise SystemExit(2)
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            print(f"error: {path.name} must be mono 16-bit", file=sys.stderr)
            raise SystemExit(2)
        return np.frombuffer(w.readframes(w.getnframes()), "<i2")


def _cmd_run(args: argparse.Namespace) -> int:
    spotter = Spotter(args.model_dir)
    pcm = _load_pcm(Path(args.audio))
    for i in range(0, pcm.size, CHUNK):
        for det in spotter.feed(pcm[i:i + CHUNK]):
            print(f"{det.keyword} {det.type_name} score={det.score:.6f} "
                  f"begin={det.begin_sample} end={det.end_sample}")
    if args.scores:
        for name, score in spotter.scores_tail().items():
            print(f"score {name} {score:.6f}")
    return 0


def _cmd_convert(args: argparse.Namespace) -> int:
    from .modelset import load_model_set

    graph_path = load_model_set(Path(args.model_dir)).graph_path
    data = graph_path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()[:16]
    cache = Path(".pypryon-cache")
    cache.mkdir(exist_ok=True)
    out = cache / f"{graph_path.stem}.{digest}.onnx"
    out.write_bytes(make_portable(data))
    print(out)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pypryon", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="spot keywords in a raw/wav16k clip")
    run.add_argument("model_dir")
    run.add_argument("audio")
    run.add_argument("--scores", action="store_true",
                     help="also print final per-keyword scores")
    run.set_defaults(fn=_cmd_run)
    conv = sub.add_parser("convert", help="write the portable ONNX cache")
    conv.add_argument("model_dir")
    conv.set_defaults(fn=_cmd_convert)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
