"""Discover and load the Pryon model directories a server offers."""
from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import onnxruntime as ort

from ..modelset import ModelSet, load_model_set
from ..runner import create_session

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoadedModel:
    dir: Path
    model_set: ModelSet
    session: ort.InferenceSession       # shared by every connection's Spotter
    thresholds: dict[str, float]        # overrides for this model's keywords only
    keywords: list[str]                 # served keywords, kw.cfg order (first = wake word)


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


def load_all(models_dir: Path, thresholds: Mapping[str, float]) -> list[LoadedModel]:
    """Load every model under ``models_dir``; broken ones are logged and skipped."""
    models: list[LoadedModel] = []
    seen: dict[str, Path] = {}
    for path in discover(models_dir):
        try:
            model_set = load_model_set(path)
            session = create_session(model_set)
        except Exception as exc:  # noqa: BLE001 - one bad dir must not take the others down
            _LOGGER.warning("Skipping %s: %s", path, exc)
            continue
        served = []
        for kw in model_set.keywords:
            if kw.name in seen:
                _LOGGER.warning("%s: keyword %s is already served from %s; skipping it here",
                                path, kw.name, seen[kw.name])
                continue
            seen[kw.name] = path
            served.append(kw.name)
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
