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


# --- llm backend -----------------------------------------------------------
# Any OpenAI-compatible endpoint. The NIM_* names are still read so an existing
# .env keeps working, but LLM_* is the documented form.
def _either(*keys: str, default: str = "") -> str:
    for key in keys:
        value = os.getenv(key)
        if value:
            return value
    return default


LLM_API_KEY = _either("LLM_API_KEY", "NIM_API_KEY")
LLM_BASE_URL = _either(
    "LLM_BASE_URL", "NIM_BASE_URL", default="https://api.groq.com/openai/v1"
)
EXTRACT_MODEL = _either("LLM_EXTRACT_MODEL", "NIM_EXTRACT_MODEL")
ADJUDICATE_MODEL = _either("LLM_ADJUDICATE_MODEL", "NIM_ADJUDICATE_MODEL")

# Extraction is a copying task; reasoning tokens are spent before any answer
# appears and are charged against max_tokens. Empty means "send nothing".
REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "").strip()

# Backwards-compatible aliases.
NIM_API_KEY = LLM_API_KEY
NIM_BASE_URL = LLM_BASE_URL

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
