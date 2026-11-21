"""pypryon-wyoming: serve Pryon wake words to Home Assistant over Wyoming."""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from functools import partial
from pathlib import Path
from urllib.parse import urlparse

from wyoming.server import AsyncServer

from .. import __version__
from .models import load_all, parse_thresholds
from .server import PryonWakeHandler, make_info

_LOGGER = logging.getLogger("pypryon.wyoming")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    env = os.environ.get
    p = argparse.ArgumentParser(prog="pypryon-wyoming", description=__doc__)
    p.add_argument("--uri", default=env("PYPRYON_URI", "tcp://0.0.0.0:10400"))
    p.add_argument("--models-dir", type=Path,
                   default=Path(env("PYPRYON_MODELS_DIR", "/share/pypryon")),
                   help="directory of model dirs (<dir>/<NAME>/pryon.manifest)")
    p.add_argument("--threshold", action="append", metavar="KEYWORD=VALUE",
                   default=[t for t in env("PYPRYON_THRESHOLDS", "").split(",") if t.strip()],
                   help="override a keyword's accept threshold (repeatable)")
    p.add_argument("--zeroconf", action="store_true", default=env("PYPRYON_ZEROCONF", "") == "1",
                   help="announce the server for Home Assistant discovery (needs pypryon[zeroconf])")
    p.add_argument("--debug", action="store_true", default=env("PYPRYON_DEBUG", "") == "1")
    p.add_argument("--version", action="version", version=__version__)
    return p.parse_args(argv)


async def serve(args: argparse.Namespace) -> None:
    models = load_all(args.models_dir, parse_thresholds(args.threshold))
    keywords = [kw for m in models for kw in m.keywords]
    server = AsyncServer.from_uri(args.uri)
    if args.zeroconf:
        from wyoming.zeroconf import HomeAssistantZeroconf

        port = urlparse(args.uri).port or 10400
        await HomeAssistantZeroconf(port=port, name="pypryon").register_server()
    _LOGGER.info("Ready on %s (wake words: %s)", args.uri, ", ".join(k.lower() for k in keywords))
    await server.run(partial(PryonWakeHandler, make_info(models), models))


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        asyncio.run(serve(args))
    except KeyboardInterrupt:
        pass
    except (RuntimeError, ValueError) as exc:
        _LOGGER.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
