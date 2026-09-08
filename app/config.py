"""Runtime configuration, read from the environment (.env is loaded if present)."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

load_dotenv(ROOT / ".env")


def _path(env_key: str, default: str) -> Path:
    raw = os.getenv(env_key, default)
    p = Path(raw)
    return p if p.is_absolute() else ROOT / p


def _flag(env_key: str, default: bool) -> bool:
    raw = os.getenv(env_key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# --- llm backend (nvidia nim, openai-compatible) ---------------------------
NIM_API_KEY = os.getenv("NIM_API_KEY", "")
NIM_BASE_URL = os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
EXTRACT_MODEL = os.getenv("NIM_EXTRACT_MODEL", "")
ADJUDICATE_MODEL = os.getenv("NIM_ADJUDICATE_MODEL", "")

# --- storage ---------------------------------------------------------------
DB_PATH = _path("DB_PATH", "data/facts.db")
UPLOAD_DIR = _path("UPLOAD_DIR", "data/uploads")

# --- llm response cache ----------------------------------------------------
# Keyed by content hash so re-running the corpus costs nothing.
LLM_CACHE_DIR = _path("LLM_CACHE_DIR", ".cache/llm")
LLM_CACHE_ENABLED = _flag("LLM_CACHE_ENABLED", True)

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


def ensure_dirs() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
