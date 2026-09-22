"""Best-effort model catalogs for the locally installed harnesses."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any


_CACHE: dict[str, list[dict[str, Any]]] | None = None
_CACHE_AT = 0.0
_CACHE_LOCK = threading.Lock()


def _option(
    provider: str,
    model_id: str,
    label: str,
    source: str,
    supported_efforts: list[str] | None = None,
    default_effort: str = "",
) -> dict[str, Any]:
    option: dict[str, Any] = {
        "provider": provider,
        "id": model_id,
        "label": label or model_id,
        "source": source,
    }
    if supported_efforts:
        option["supported_efforts"] = supported_efforts
    if default_effort:
        option["default_effort"] = default_effort
    return option


def _reasoning_levels(model: dict[str, Any]) -> list[str]:
    values = model.get("supported_reasoning_levels", [])
    if not isinstance(values, list):
        return []

    result: list[str] = []
    for value in values:
        if isinstance(value, dict):
            value = value.get("effort")
        effort = str(value or "").strip().lower()
        if effort and effort not in result:
            result.append(effort)
    return result


def _json_models(path: Path, provider: str) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return []
    models = payload.get("models", []) if isinstance(payload, dict) else payload if isinstance(payload, list) else []
    source_label = {
        "codex": "Codex installed catalog",
        "codex-fugu": "codex-fugu installed catalog",
    }.get(provider, "installed catalog")
    result = []
    for model in models:
        if not isinstance(model, dict):
            continue
        slug = model.get("slug")
        if not slug or model.get("show_in_picker") is False:
            continue
        visibility = str(model.get("visibility") or "").strip().lower()
        if visibility and visibility not in {"list", "visible"}:
            continue
        result.append(
            _option(
                provider,
                str(slug),
                str(model.get("display_name") or slug),
                source_label,
                _reasoning_levels(model),
                str(model.get("default_reasoning_level") or "").strip().lower(),
            )
        )
    return result


def _codex_home() -> Path:
    return Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")).expanduser()


def _opencode_models() -> list[dict[str, Any]]:
    executable = shutil.which("opencode")
    if not executable:
        return []
    try:
        completed = subprocess.run(
            [executable, "models"],
            capture_output=True,
            text=True,
            timeout=int(os.environ.get("SOT_MODEL_CATALOG_TIMEOUT", "20")),
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    result = []
    for line in completed.stdout.splitlines():
        candidate = line.strip().split()[0] if line.strip() else ""
        if not re.fullmatch(r"[A-Za-z0-9._-]+/[A-Za-z0-9._:-]+", candidate):
            continue
        result.append(_option("opencode", candidate, candidate, "opencode models"))
    return result


def _claude_models() -> list[dict[str, Any]]:
    if not shutil.which("claude"):
        return []
    configured = [item.strip() for item in os.environ.get("SOT_CLAUDE_MODELS", "").split(",") if item.strip()]
    values = configured or ["fable", "opus", "sonnet", "haiku"]
    return [_option("claude", item, item, "Claude CLI alias/configuration") for item in values]


def _dedupe(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    result = []
    for model in models:
        key = (model["provider"], model["id"])
        if key not in seen:
            seen.add(key)
            result.append(model)
    return result


def model_catalog(refresh: bool = False) -> dict[str, list[dict[str, Any]]]:
    """Return currently discoverable model IDs grouped by harness.

    OpenCode is queried through its own model command. Codex and Fugu are read
    from their local model catalogs, while Claude exposes its documented model
    aliases plus optional comma-separated ``SOT_CLAUDE_MODELS`` entries. Each
    catalog model carries its supported reasoning levels when the harness
    publishes them.
    """

    global _CACHE, _CACHE_AT
    with _CACHE_LOCK:
        if _CACHE is not None and not refresh and time.monotonic() - _CACHE_AT < 60:
            return _CACHE

        home = _codex_home()
        result = {
            "claude": _claude_models(),
            "codex": _json_models(home / "models_cache.json", "codex") if shutil.which("codex") else [],
            "codex-fugu": _json_models(home / "fugu.json", "codex-fugu") if shutil.which("codex-fugu") else [],
            "opencode": _opencode_models(),
        }
        _CACHE = {key: _dedupe(value) for key, value in result.items()}
        _CACHE_AT = time.monotonic()
        return _CACHE

