"""Config file + LLM provider resolution (OpenAI or OpenRouter).

Both providers expose an OpenAI-compatible API. We pick the right ``base_url`` and
default model based on the key prefix:

  * ``sk-or-...``   → OpenRouter (https://openrouter.ai/api/v1)
  * anything else   → OpenAI      (default openai endpoint)

Resolution order for the API key (highest priority first):
  1. Explicit argument to :func:`resolve_llm_config`
  2. ``OPENROUTER_API_KEY`` / ``OPENAI_API_KEY`` env vars
  3. ``.env`` file in the current working directory (same names)
  4. ``~/.config/bibsync/config.json`` keys ``openrouter_key`` / ``openai_key``

If a key is stored under the "wrong" config slot (e.g., an ``sk-or-`` key under
``openai_key``), we auto-detect by prefix rather than failing — friendlier UX.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from platformdirs import user_config_dir

CONFIG_FILE_NAME = "config.json"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_DEFAULT_MODEL = "openai/gpt-4o-mini"
OPENAI_DEFAULT_MODEL = "gpt-4o-mini"


@dataclass
class LLMConfig:
    api_key: str
    provider: str  # "openai" | "openrouter"
    base_url: Optional[str]  # None = OpenAI default
    model: str
    source: str  # human-readable: where the key came from


def config_path() -> Path:
    return Path(user_config_dir("bibsync", "bibsync")) / CONFIG_FILE_NAME


def load_config() -> dict:
    p = config_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def save_config(data: dict) -> Path:
    p = config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    try:
        os.chmod(p, 0o600)  # api keys are sensitive
    except OSError:
        pass
    return p


def _read_dotenv_key(key: str) -> Optional[str]:
    """Tiny .env reader — handles `KEY=value` and `KEY="value"`. No subshell expansion."""
    dotenv = Path.cwd() / ".env"
    if not dotenv.exists():
        return None
    try:
        for raw_line in dotenv.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() == key:
                v = v.strip()
                if (v.startswith('"') and v.endswith('"')) or (
                    v.startswith("'") and v.endswith("'")
                ):
                    v = v[1:-1]
                return v
    except OSError:
        return None
    return None


def _detect_provider(api_key: str) -> str:
    """OpenRouter keys begin with ``sk-or-``. Everything else is treated as OpenAI."""
    return "openrouter" if api_key.startswith("sk-or-") else "openai"


def _find_key() -> tuple[Optional[str], str]:
    """Return (api_key, source_description) using the resolution order."""
    cfg = load_config()

    # 1. Env vars (OpenRouter first since it's more specific)
    if v := os.environ.get("OPENROUTER_API_KEY"):
        return v, "env:OPENROUTER_API_KEY"
    if v := os.environ.get("OPENAI_API_KEY"):
        return v, "env:OPENAI_API_KEY"

    # 2. .env file
    if v := _read_dotenv_key("OPENROUTER_API_KEY"):
        return v, ".env:OPENROUTER_API_KEY"
    if v := _read_dotenv_key("OPENAI_API_KEY"):
        return v, ".env:OPENAI_API_KEY"

    # 3. Config file
    if v := cfg.get("openrouter_key"):
        return v, "config:openrouter_key"
    if v := cfg.get("openai_key"):
        return v, "config:openai_key"

    return None, "not set"


def resolve_llm_config(explicit_api_key: Optional[str] = None) -> Optional[LLMConfig]:
    """Return a fully-resolved :class:`LLMConfig`, or ``None`` if no key is available."""
    cfg = load_config()
    source = "explicit"
    api_key = explicit_api_key

    if not api_key:
        api_key, source = _find_key()
    if not api_key:
        return None

    provider = cfg.get("provider") or _detect_provider(api_key)

    if provider == "openrouter":
        base_url = cfg.get("llm_base_url") or OPENROUTER_BASE_URL
        default_model = OPENROUTER_DEFAULT_MODEL
    else:
        base_url = cfg.get("llm_base_url")  # None → openai default
        default_model = OPENAI_DEFAULT_MODEL

    model = cfg.get("llm_model") or default_model

    return LLMConfig(
        api_key=api_key,
        provider=provider,
        base_url=base_url,
        model=model,
        source=source,
    )


# Backward-compat shim — older code paths call resolve_openai_key().
def resolve_openai_key(explicit: Optional[str] = None) -> Optional[str]:
    cfg = resolve_llm_config(explicit)
    return cfg.api_key if cfg else None
