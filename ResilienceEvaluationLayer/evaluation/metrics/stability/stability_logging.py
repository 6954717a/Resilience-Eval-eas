#!/usr/bin/env python3

"""
Stability Logging

Handles logging and persistence of Stability metrics to JSON files.
"""

import json
import logging
import os
import tempfile
from typing import Any, Dict

from .stability_metrics import StabilityMetrics

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


def log_stability_metrics(
    stability_metrics: StabilityMetrics,
    output_dir: str,
    episode_filename: str,
    env_interface: Any,
) -> str:
    """
    Log Stability metrics to JSON file.

    Args:
        stability_metrics: StabilityMetrics instance
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

    analyses_dir = os.path.join(output_dir, "analyses", "stability")
    os.makedirs(analyses_dir, exist_ok=True)

    filename = f"stability_metrics-episode_{episode_id}_{episode_filename}.json"
    filepath = os.path.join(analyses_dir, filename)

    save_data = {
        "episode_id": str(episode_id),
        "episode_filename": episode_filename,
        "stability_metrics": stability_metrics.to_dict(),
        "summary": {
            "beta": stability_metrics.beta,
            "value_variance": stability_metrics.value_variance,
            "value_delta_variance": stability_metrics.value_delta_variance,
            "n_replan": stability_metrics.n_replan,
            "p_cbf": stability_metrics.p_cbf,
        },
        "metadata": {
            "beta_mode": stability_metrics.beta_mode,
            "beta_scope": stability_metrics.beta_scope,
        },
    }

    _write_json(filepath, save_data)
    logger.info("Stability metrics saved: %s", filepath)
    return filepath
