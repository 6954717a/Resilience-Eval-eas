"""Safe serialization helpers for runtime configuration snapshots."""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


REDACTED_VALUE = "<redacted>"

_ALWAYS_SENSITIVE_PARTS = frozenset(
    {
        "auth",
        "authorization",
        "bearer",
        "cookie",
        "cookies",
        "credential",
        "credentials",
        "passwd",
        "password",
        "passwords",
        "pwd",
        "secret",
        "secrets",
        "token",
        "tokens",
    }
)
_KEY_QUALIFIERS = frozenset(
    {
        "access",
        "anthropic",
        "api",
        "azure",
        "client",
        "encryption",
        "hf",
        "openai",
        "private",
        "secret",
        "service",
        "signing",
        "ssh",
    }
)
_SENSITIVE_EXACT_NAMES = frozenset(
    {
        "connection_string",
        "database_dsn",
        "dsn",
        "key",
        "key_file",
        "key_path",
    }
)


def _normalized_field_name(field_name: Any) -> str:
    text = str(field_name)
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", text)
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def is_sensitive_config_field(field_name: Any) -> bool:
    """Return whether a config field can hold authentication material."""

    normalized = _normalized_field_name(field_name)
    if normalized in _SENSITIVE_EXACT_NAMES:
        return True

    parts = normalized.split("_") if normalized else []
    if any(part in _ALWAYS_SENSITIVE_PARTS for part in parts):
        return True

    for index, part in enumerate(parts):
        if part != "key":
            continue
        if index > 0 and parts[index - 1] in _KEY_QUALIFIERS:
            return True
        if index + 1 < len(parts) and parts[index + 1] in {"file", "path"}:
            return True
    return False


def redact_sensitive_fields(
    value: Any, *, replacement: Any = REDACTED_VALUE
) -> Any:
    """Return a recursively redacted copy without mutating ``value``."""

    if isinstance(value, Mapping):
        return {
            key: (
                copy.deepcopy(replacement)
                if is_sensitive_config_field(key)
                else redact_sensitive_fields(child, replacement=replacement)
            )
            for key, child in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [
            redact_sensitive_fields(child, replacement=replacement)
            for child in value
        ]
    return copy.deepcopy(value)


def redacted_config_yaml(
    config: Any,
    *,
    resolve: bool = False,
    replacement: Any = REDACTED_VALUE,
) -> str:
    """Serialize a config to YAML after redacting credential-like fields.

    OmegaConf conversion creates a detached container first, so neither
    redaction nor YAML serialization changes the runtime configuration.
    Use ``replacement=None`` for child-process inputs so they can fall back
    to credentials supplied through their environment.
    """

    if OmegaConf.is_config(config):
        plain_config = OmegaConf.to_container(
            config,
            resolve=False,
            throw_on_missing=False,
        )
    else:
        plain_config = copy.deepcopy(config)
    redacted = redact_sensitive_fields(plain_config, replacement=replacement)
    return OmegaConf.to_yaml(OmegaConf.create(redacted), resolve=resolve)


def write_redacted_config_copy(
    source_path: Any,
    destination_path: Any,
    *,
    resolve: bool = False,
    replacement: Any = REDACTED_VALUE,
) -> Path:
    """Load a YAML config and write only its redacted representation."""

    source_config = OmegaConf.load(source_path)
    destination = Path(destination_path)
    destination.write_text(
        redacted_config_yaml(
            source_config,
            resolve=resolve,
            replacement=replacement,
        ),
        encoding="utf-8",
    )
    return destination


__all__ = [
    "REDACTED_VALUE",
    "is_sensitive_config_field",
    "redact_sensitive_fields",
    "redacted_config_yaml",
    "write_redacted_config_copy",
]
