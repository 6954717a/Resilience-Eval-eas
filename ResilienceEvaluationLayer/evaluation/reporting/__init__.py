"""Validity-aware reporting surfaces for resilience evaluation."""

from .resilience_schema import (
    POSTPROCESS_METRIC_KEYS,
    RUNTIME_METRIC_KEYS,
    empty_runtime_metric_summary,
    primary_display_items,
)
from .episode_rows import (
    DEFAULT_EPISODE_DYNAMIC_PREFIXES,
    build_episode_csv_metrics,
    collect_episode_stats_keys,
)
from .tabular import (
    DuplicateColumnError,
    TabularSchemaError,
    append_csv_rows,
    read_csv_rows,
    write_csv_rows_exact,
)

__all__ = [
    "DEFAULT_EPISODE_DYNAMIC_PREFIXES",
    "build_episode_csv_metrics",
    "collect_episode_stats_keys",
    "POSTPROCESS_METRIC_KEYS",
    "RUNTIME_METRIC_KEYS",
    "empty_runtime_metric_summary",
    "primary_display_items",
    "DuplicateColumnError",
    "TabularSchemaError",
    "append_csv_rows",
    "read_csv_rows",
    "write_csv_rows_exact",
]
