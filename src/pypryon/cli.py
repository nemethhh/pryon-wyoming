"""pypryon command line.

  pypryon run <model_dir> <audio> [--scores] [--chunk N]
  pypryon convert <model_dir> [-o PATH]

``run`` feeds 16 kHz mono s16 audio (WAV, or raw s16le) through a Spotter and
prints one line per detection. ``convert`` writes the stock-onnxruntime version
of the model's graph (default ``./.pypryon-cache/``, never next to the model:
model directories are often read-only).
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from . import __version__
from .audio import load_pcm16
from .graph import make_portable
from .modelset import ModelError, load_model_set
from .spotter import Spotter


def _cmd_run(args: argparse.Namespace) -> int:
    spotter = Spotter(args.model_dir)
    pcm = load_pcm16(args.audio)
    for i in range(0, pcm.size, args.chunk):
        for det in spotter.feed(pcm[i:i + args.chunk]):
            print(f"{det.keyword} {det.type_name} score={det.score:.6f} "
                  f"begin={det.begin_sample} end={det.end_sample}")
    if args.scores:
        for name, score in spotter.scores_tail().items():
            print(f"score {name} {score:.6f}")
    return 0


def _cmd_convert(args: argparse.Namespace) -> int:
    graph_path = load_model_set(Path(args.model_dir).expanduser()).graph_path
    data = graph_path.read_bytes()
    if args.output:
        out = Path(args.output)
    else:
        digest = hashlib.sha256(data).hexdigest()[:16]
        out = Path(".pypryon-cache") / f"{graph_path.stem}.{digest}.onnx"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(make_portable(data))
    print(out)
    return 0


def _positive(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return value


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pypryon", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"pypryon {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="spot keywords in a 16 kHz mono s16 WAV or raw file")
    run.add_argument("model_dir")
    run.add_argument("audio")
    run.add_argument("--scores", action="store_true", help="also print final per-keyword scores")
    run.add_argument("--chunk", type=_positive, default=1280,
                     help="samples per feed (default 1280 = 80 ms)")
    run.set_defaults(fn=_cmd_run)
    conv = sub.add_parser("convert", help="write the stock-onnxruntime graph")
    conv.add_argument("model_dir")
    conv.add_argument("-o", "--output", help="output path (default ./.pypryon-cache/...)")
    conv.set_defaults(fn=_cmd_convert)
    args = ap.parse_args(argv)
    try:
        return args.fn(args)
    except (ModelError, ValueError, OSError) as exc:
        print(f"pypryon: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
