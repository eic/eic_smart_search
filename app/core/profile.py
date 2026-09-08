"""Collaboration profile: the thin layer that makes this deployment EIC's
(or STAR's, or anyone's).

Everything collaboration-specific that is data rather than code lives in
``profiles/<name>/``:

    synonyms.yaml       acronym -> expansion, used for query expansion
    prompt_notes.txt    collaboration-specific rules appended to the LLM
                        system prompt (disambiguations, tool hints)
    golden.yaml         the evaluation benchmark
    services.yaml       curated live-services registry (optional)

Select with the ``PROFILE`` environment variable (default: ``eic``).
The generic pipeline — retrieval, fusion, reranking, generation, eval —
never branches on the profile name; it only reads these files.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

PROFILES_ROOT = Path(__file__).resolve().parent.parent.parent / "profiles"


def profile_name() -> str:
    return os.environ.get("PROFILE", "eic").strip() or "eic"


def profile_dir() -> Path:
    d = PROFILES_ROOT / profile_name()
    if not d.is_dir():
        logger.warning("profile_dir_missing", extra={"profile": profile_name()})
    return d


@lru_cache(maxsize=4)
def load_synonyms(name: str | None = None) -> dict[str, str]:
    path = PROFILES_ROOT / (name or profile_name()) / "synonyms.yaml"
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return {str(k).lower(): str(v) for k, v in data.items()}
    except FileNotFoundError:
        logger.warning("profile_synonyms_missing", extra={"path": str(path)})
        return {}


@lru_cache(maxsize=4)
def load_prompt_notes(name: str | None = None) -> str:
    path = PROFILES_ROOT / (name or profile_name()) / "prompt_notes.txt"
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""


def golden_path(name: str | None = None) -> Path:
    return PROFILES_ROOT / (name or profile_name()) / "golden.yaml"


def services_path(name: str | None = None) -> Path:
    return PROFILES_ROOT / (name or profile_name()) / "services.yaml"
