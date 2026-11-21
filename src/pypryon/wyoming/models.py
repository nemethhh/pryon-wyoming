"""Discover and load the Pryon model directories a server offers."""
from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..davs import DavsModelSet
from ..modelset import ModelSet, load_model_set
from ..pipeline import create_sessions
from ..rules import DavsOrchestra
from ..runner import create_session

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoadedModel:
    dir: Path
    model_set: ModelSet | DavsModelSet
    session: Any                        # shared by every connection's Spotter
    thresholds: dict[str, float]        # overrides for this model's keywords only
    keywords: list[str]                 # served keywords, word order (first = wake word)


def discover(models_dir: Path) -> list[Path]:
    """``models_dir`` itself or its immediate subdirectories holding a pryon.manifest."""
    models_dir = Path(models_dir).expanduser()
    if (models_dir / "pryon.manifest").is_file():
        return [models_dir]
    if not models_dir.is_dir():
        return []
    return sorted(p for p in models_dir.iterdir() if (p / "pryon.manifest").is_file())


def parse_thresholds(entries: Iterable[str]) -> dict[str, float]:
    """``["ALEXA=0.88"]`` -> ``{"ALEXA": 0.88}``; values must lie in (0, 1]."""
    out: dict[str, float] = {}
    for entry in entries:
        name, sep, value = entry.partition("=")
        name = name.strip()
        try:
            number = float(value)
        except ValueError:
            number = None
        if not sep or not name or number is None or not 0 < number <= 1:
            raise ValueError(f"bad threshold {entry!r}: expected KEYWORD=VALUE with 0 < VALUE <= 1")
        out[name] = number
    return out


def _open(path: Path) -> tuple[ModelSet | DavsModelSet, Any, list[str]]:
    model_set = load_model_set(path)
    if isinstance(model_set, DavsModelSet):
        keywords = DavsOrchestra(model_set.spotters, words=model_set.words).keywords
        skipped = sorted({s.name for s in model_set.spotters} - set(keywords))
        if skipped:
            _LOGGER.warning("%s: not serving %s (unsupported classifier)", path, ", ".join(skipped))
        return model_set, create_sessions(model_set), keywords
    return model_set, create_session(model_set), [k.name for k in model_set.keywords]


def load_all(models_dir: Path, thresholds: Mapping[str, float]) -> list[LoadedModel]:
    """Load every model under ``models_dir``; broken ones are logged and skipped.

    A keyword offered by several models is served once: DAVS sets (the newer
    generation) first, then directory order.
    """
    opened = []
    for path in discover(models_dir):
        try:
            opened.append((path, *_open(path)))
        except Exception as exc:  # noqa: BLE001 - one bad dir must not take the others down
            _LOGGER.warning("Skipping %s: %s", path, exc)
    opened.sort(key=lambda item: not isinstance(item[1], DavsModelSet))
    models: list[LoadedModel] = []
    seen: dict[str, Path] = {}
    for path, model_set, session, keywords in opened:
        served = []
        for name in keywords:
            if name in seen:
                _LOGGER.warning("%s: keyword %s is already served from %s; skipping it here",
                                path, name, seen[name])
                continue
            seen[name] = path
            served.append(name)
        if not served:
            continue
        models.append(LoadedModel(path, model_set, session,
                                  {k: v for k, v in thresholds.items() if k in served}, served))
    if not models:
        raise RuntimeError(
            f"no Pryon model found under {models_dir} (expected <dir>/<NAME>/pryon.manifest)")
    unknown = [k for k in thresholds if k not in seen]
    if unknown:
        raise ValueError(f"threshold for unknown keyword {unknown[0]!r}; models have: "
                         f"{', '.join(seen)}")
    return models
