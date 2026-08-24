#!/usr/bin/env python3

"""Strict, atomic CSV helpers for evaluation artifacts.

The evaluation pipeline treats CSV files as durable evidence.  These helpers
therefore make replacement and empty-input behaviour explicit, validate the
schema instead of silently dropping columns, and rewrite schema-expanding
appends atomically.
"""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union


class TabularSchemaError(ValueError):
    """Raised when a row or an existing CSV violates its declared schema."""


class DuplicateColumnError(TabularSchemaError):
    """Raised when a schema contains the same CSV column more than once."""


def _normalize_fieldnames(fieldnames: Sequence[Any]) -> List[str]:
    normalized: List[str] = []
    seen = set()
    for raw_name in fieldnames:
        name = str(raw_name)
        if name in seen:
            raise DuplicateColumnError(f"Duplicate CSV column: {name!r}")
        seen.add(name)
        normalized.append(name)
    return normalized


def _normalize_rows(rows: Sequence[Mapping[Any, Any]]) -> List[Dict[str, Any]]:
    normalized_rows: List[Dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(
                f"CSV row {row_index} must be a mapping, got {type(row).__name__}"
            )
        normalized: Dict[str, Any] = {}
        for raw_key, value in row.items():
            key = str(raw_key)
            if key in normalized:
                raise DuplicateColumnError(
                    f"CSV row {row_index} contains duplicate normalized key {key!r}"
                )
            normalized[key] = value
        normalized_rows.append(normalized)
    return normalized_rows


def _infer_fieldnames(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    return fieldnames


def _validate_rows(
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    declared = set(fieldnames)
    for row_index, row in enumerate(rows):
        unexpected = [key for key in row if key not in declared]
        if unexpected:
            raise TabularSchemaError(
                "CSV row "
                f"{row_index} has columns absent from the declared schema: "
                f"{unexpected}"
            )


def _read_csv_table(
    path: Union[str, Path],
) -> Tuple[List[str], List[Dict[str, str]]]:
    input_path = Path(path)
    if not input_path.exists():
        return [], []

    with open(input_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            raw_header = next(reader)
        except StopIteration:
            return [], []
        fieldnames = _normalize_fieldnames(raw_header)
        rows: List[Dict[str, str]] = []
        for line_number, values in enumerate(reader, start=2):
            if not values:
                continue
            if len(values) > len(fieldnames):
                raise TabularSchemaError(
                    f"CSV row {line_number} has {len(values)} values for "
                    f"{len(fieldnames)} columns"
                )
            padded = list(values) + [""] * (len(fieldnames) - len(values))
            rows.append(dict(zip(fieldnames, padded)))
    return fieldnames, rows


def read_csv_rows(path: Union[str, Path]) -> List[Dict[str, str]]:
    """Read CSV rows while rejecting duplicate or over-wide schemas.

    A missing or zero-byte file yields an empty list, matching the historical
    collector behaviour.  Duplicate headers are rejected because ``DictReader``
    would otherwise overwrite an earlier column silently.
    """

    _fieldnames, rows = _read_csv_table(path)
    return rows


def _atomic_write(
    output_path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=str(output_path.parent),
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(
            file_descriptor,
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(fieldnames),
                extrasaction="raise",
                restval="",
            )
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def write_csv_rows_exact(
    output_path: Union[str, Path],
    rows: Sequence[Mapping[Any, Any]],
    *,
    fieldnames: Optional[Sequence[Any]] = None,
    overwrite_policy: str = "replace",
    empty_policy: str = "error",
) -> Path:
    """Atomically replace a CSV artifact with exactly ``rows``.

    ``overwrite_policy`` is ``"replace"`` or ``"error"``.  For empty rows,
    callers must choose one of:

    * ``"error"``: fail and leave any existing artifact untouched;
    * ``"write_header"``: atomically replace it with a header-only CSV and
      provide ``fieldnames``;
    * ``"remove"``: remove the exact target so stale evidence cannot survive.

    When ``fieldnames`` is provided, any undeclared row key raises instead of
    being discarded.  Without it, the schema is the first-seen union of all
    row keys.
    """

    path = Path(output_path)
    if overwrite_policy not in {"replace", "error"}:
        raise ValueError(
            "overwrite_policy must be either 'replace' or 'error', got "
            f"{overwrite_policy!r}"
        )
    if empty_policy not in {"error", "write_header", "remove"}:
        raise ValueError(
            "empty_policy must be 'error', 'write_header', or 'remove', got "
            f"{empty_policy!r}"
        )
    if path.exists() and overwrite_policy == "error":
        raise FileExistsError(path)

    normalized_rows = _normalize_rows(rows)
    normalized_fieldnames = (
        _normalize_fieldnames(fieldnames) if fieldnames is not None else None
    )
    if not normalized_rows:
        if empty_policy == "error":
            raise ValueError(
                "Refusing to write empty CSV rows without an explicit "
                "empty_policy"
            )
        if empty_policy == "remove":
            if path.exists():
                path.unlink()
            return path
        if normalized_fieldnames is None:
            raise ValueError(
                "fieldnames are required when empty_policy='write_header'"
            )
        _atomic_write(path, (), normalized_fieldnames)
        return path

    schema = (
        normalized_fieldnames
        if normalized_fieldnames is not None
        else _infer_fieldnames(normalized_rows)
    )
    _validate_rows(normalized_rows, schema)
    if not schema:
        raise TabularSchemaError("A non-empty CSV requires at least one column")
    _atomic_write(path, normalized_rows, schema)
    return path


def append_csv_rows(
    output_path: Union[str, Path],
    rows: Sequence[Mapping[Any, Any]],
    *,
    fieldnames: Optional[Sequence[Any]] = None,
    schema_policy: str = "expand",
    empty_policy: str = "noop",
) -> Path:
    """Append rows atomically, expanding the existing schema when requested.

    ``schema_policy='expand'`` preserves every existing column and appends new
    columns in first-seen order.  Because an in-place append cannot change the
    CSV header safely, the complete table is rewritten with ``os.replace``.
    ``schema_policy='exact'`` rejects every schema change.

    Empty append input is either an explicit ``"noop"`` or an ``"error"``;
    unlike exact replacement, it never implies that an existing artifact is
    stale.
    """

    path = Path(output_path)
    if schema_policy not in {"expand", "exact"}:
        raise ValueError(
            "schema_policy must be either 'expand' or 'exact', got "
            f"{schema_policy!r}"
        )
    if empty_policy not in {"noop", "error"}:
        raise ValueError(
            "empty_policy must be either 'noop' or 'error', got "
            f"{empty_policy!r}"
        )

    normalized_rows = _normalize_rows(rows)
    if not normalized_rows:
        if empty_policy == "error":
            raise ValueError("Refusing an empty CSV append")
        return path

    fieldnames_supplied = fieldnames is not None
    requested_schema = _normalize_fieldnames(fieldnames or ())
    if not path.exists():
        schema = list(requested_schema)
        inferred_schema = _infer_fieldnames(normalized_rows)
        if schema_policy == "expand":
            seen = set(schema)
            schema.extend(name for name in inferred_schema if name not in seen)
        elif not fieldnames_supplied:
            schema = inferred_schema
        return write_csv_rows_exact(
            path,
            normalized_rows,
            fieldnames=schema,
            overwrite_policy="replace",
            empty_policy="error",
        )

    existing_schema, existing_rows = _read_csv_table(path)
    if not existing_schema:
        schema = (
            list(requested_schema)
            if fieldnames_supplied
            else _infer_fieldnames(normalized_rows)
        )
    elif schema_policy == "exact":
        if fieldnames_supplied and requested_schema != existing_schema:
            raise TabularSchemaError(
                "Requested append schema does not exactly match existing CSV "
                f"schema: requested={requested_schema}, existing={existing_schema}"
            )
        schema = list(existing_schema)
    else:
        schema = list(existing_schema)
        seen = set(schema)
        for name in requested_schema:
            if name not in seen:
                seen.add(name)
                schema.append(name)
        for name in _infer_fieldnames(normalized_rows):
            if name not in seen:
                seen.add(name)
                schema.append(name)

    _validate_rows(normalized_rows, schema)
    return write_csv_rows_exact(
        path,
        [*existing_rows, *normalized_rows],
        fieldnames=schema,
        overwrite_policy="replace",
        empty_policy="error",
    )


__all__ = [
    "DuplicateColumnError",
    "TabularSchemaError",
    "append_csv_rows",
    "read_csv_rows",
    "write_csv_rows_exact",
]
