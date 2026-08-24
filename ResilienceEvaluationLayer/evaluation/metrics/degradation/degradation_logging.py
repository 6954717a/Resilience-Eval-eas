#!/usr/bin/env python3

"""
Degradation Logging

Handles logging and persistence of Degradation metrics to JSON files.
"""

import json
import logging
import os
import tempfile
from typing import Any, Dict

from .degradation_metrics import DegradationMetrics

logger = logging.getLogger(__name__)


def _json_default(obj: Any) -> Any:
    """JSON serialization helper for numpy types."""
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return obj.item()
        except Exception:
            pass
    if hasattr(obj, "tolist") and callable(obj.tolist):
        try:
            return obj.tolist()
        except Exception:
            pass
    return str(obj)


def _write_json(filepath: str, data: Dict[str, Any]) -> None:
    """Atomically write JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(
        prefix=".tmp_", suffix=".json", dir=os.path.dirname(filepath)
    )
    try:
        with os.fdopen(tmp_fd, "w") as file:
            json.dump(data, file, indent=2, default=_json_default)
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp_path, filepath)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def log_degradation_metrics(
    degradation_metrics: DegradationMetrics,
    output_dir: str,
    episode_filename: str,
    env_interface: Any,
) -> str:
    """
    Log Degradation metrics to JSON file.

    Args:
        degradation_metrics: DegradationMetrics instance
        output_dir: Output directory for analyses
        episode_filename: Episode filename
        env_interface: Environment interface for episode ID

    Returns:
        Path to saved JSON file
    """
    try:
        episode_id = env_interface.env.env.env._env.current_episode.episode_id
    except Exception:
        episode_id = "unknown"

    analyses_dir = os.path.join(output_dir, "analyses", "degradation")
    os.makedirs(analyses_dir, exist_ok=True)

    filename = f"degradation_metrics-episode_{episode_id}_{episode_filename}.json"
    filepath = os.path.join(analyses_dir, filename)

    save_data = {
        "episode_id": str(episode_id),
        "episode_filename": episode_filename,
        "degradation_metrics": degradation_metrics.to_dict(),
        "summary": {
            "auc_loss": degradation_metrics.auc_loss,
            "p_cliff": degradation_metrics.p_cliff,
            "t_rec": degradation_metrics.t_rec,
            "l_bd": degradation_metrics.l_bd,
            "nrr": degradation_metrics.nrr,
            "recovery_event_count": len(degradation_metrics.recovery_events),
        },
        "metadata": {
            "l_bd_scope": degradation_metrics.l_bd_scope,
            "l_bd_available": degradation_metrics.l_bd_available,
        },
    }

    _write_json(filepath, save_data)
    logger.info("Degradation metrics saved: %s", filepath)
    return filepath
