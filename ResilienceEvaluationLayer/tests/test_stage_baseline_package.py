import os
from pathlib import Path
import subprocess
import sys
import types

import pytest


# Keep these package-boundary tests independent of the Habitat simulator. The
# StageBaseline implementation only needs the lightweight evaluation modules.
_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _PACKAGE_ROOT.parent
if "habitat_llm" not in sys.modules:
    root_package = types.ModuleType("habitat_llm")
    root_package.__path__ = [str(_PACKAGE_ROOT)]
    sys.modules["habitat_llm"] = root_package
if "habitat_llm.evaluation" not in sys.modules:
    evaluation_package = types.ModuleType("habitat_llm.evaluation")
    evaluation_package.__path__ = [str(_PACKAGE_ROOT / "evaluation")]
    sys.modules["habitat_llm.evaluation"] = evaluation_package
if "habitat_llm.evaluation.metrics" not in sys.modules:
    metrics_package = types.ModuleType("habitat_llm.evaluation.metrics")
    metrics_package.__path__ = [str(_PACKAGE_ROOT / "evaluation" / "metrics")]
    sys.modules["habitat_llm.evaluation.metrics"] = metrics_package


from habitat_llm.evaluation.metrics.rebound import (
    StageAnchor as ReboundStageAnchor,
    StageBaseline as ReboundStageBaseline,
    StageBaselineDim as ReboundStageBaselineDim,
    StageBaselineEstimator as ReboundStageBaselineEstimator,
    StageResolver as ReboundStageResolver,
)
from habitat_llm.evaluation.metrics.rebound.stage_anchor import (
    StageAnchor as LegacyStageAnchor,
    StageResolver as LegacyStageResolver,
)
from habitat_llm.evaluation.metrics.rebound.stage_baseline import (
    StageBaseline as LegacyStageBaseline,
    StageBaselineDim as LegacyStageBaselineDim,
    StageBaselineEstimator as LegacyStageBaselineEstimator,
)
from habitat_llm.evaluation.stage_baseline import (
    StageAnchor,
    StageBaseline,
    StageBaselineDim,
    StageBaselineEstimator,
    StageResolver,
)


_LIGHTWEIGHT_IMPORT_PREAMBLE = f"""
import pathlib
import sys
import types

package_root = pathlib.Path({str(_PACKAGE_ROOT)!r})
root_package = types.ModuleType("habitat_llm")
root_package.__path__ = [str(package_root)]
sys.modules["habitat_llm"] = root_package
evaluation_package = types.ModuleType("habitat_llm.evaluation")
evaluation_package.__path__ = [str(package_root / "evaluation")]
sys.modules["habitat_llm.evaluation"] = evaluation_package
metrics_package = types.ModuleType("habitat_llm.evaluation.metrics")
metrics_package.__path__ = [str(package_root / "evaluation" / "metrics")]
sys.modules["habitat_llm.evaluation.metrics"] = metrics_package
"""

_REAL_METRICS_IMPORT_PREAMBLE = f"""
import pathlib
import sys
import types

package_root = pathlib.Path({str(_PACKAGE_ROOT)!r})
root_package = types.ModuleType("habitat_llm")
root_package.__path__ = [str(package_root)]
sys.modules["habitat_llm"] = root_package
evaluation_package = types.ModuleType("habitat_llm.evaluation")
evaluation_package.__path__ = [str(package_root / "evaluation")]
sys.modules["habitat_llm.evaluation"] = evaluation_package
"""


def _run_lightweight_python(source: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [sys.executable, "-B", "-c", _LIGHTWEIGHT_IMPORT_PREAMBLE + source],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _run_with_real_metrics_package(source: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [sys.executable, "-B", "-c", _REAL_METRICS_IMPORT_PREAMBLE + source],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_stage_baseline_legacy_exports_preserve_symbol_identity():
    assert StageAnchor is LegacyStageAnchor is ReboundStageAnchor
    assert StageResolver is LegacyStageResolver is ReboundStageResolver
    assert StageBaseline is LegacyStageBaseline is ReboundStageBaseline
    assert StageBaselineDim is LegacyStageBaselineDim is ReboundStageBaselineDim
    assert (
        StageBaselineEstimator
        is LegacyStageBaselineEstimator
        is ReboundStageBaselineEstimator
    )


def test_stage_resolver_uses_compact_recent_actions_and_treats_wait_as_idle():
    resolver = StageResolver()

    assert resolver.current_modality({"recent_action": "0:Navigate[chair]; 1:Wait[]"}) == "navigation"
    assert resolver.current_modality({"recent_action": "Agent_0:Pick[cup]"}) == "manipulation"
    assert resolver.current_modality({"recent_action": "0:Coordinate[agent_1]"}) == "coordination"
    assert resolver.current_modality({"recent_action": "0:Wait[]"}) == "planning"


@pytest.mark.parametrize(
    "ordered_imports",
    [
        """
from habitat_llm.evaluation.stage_baseline import StageBaselineEstimator
from habitat_llm.evaluation.metrics.stability.trajectory_loss import compute_stage_trajectory_loss
""",
        """
from habitat_llm.evaluation.metrics.stability.trajectory_loss import compute_stage_trajectory_loss
from habitat_llm.evaluation.stage_baseline import StageBaselineEstimator
""",
    ],
    ids=("stage-baseline-first", "stability-first"),
)
def test_stage_baseline_and_stability_import_order_is_safe(ordered_imports):
    result = _run_lightweight_python(ordered_imports)
    assert result.returncode == 0, result.stderr


def test_canonical_import_does_not_cycle_through_real_metrics_package():
    result = _run_with_real_metrics_package(
        """
from habitat_llm.evaluation.stage_baseline import StageBaselineEstimator
from habitat_llm.evaluation.metrics.stability.trajectory_loss import extract_stability_vector
"""
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "module_name",
    [
        "habitat_llm.evaluation.stage_baseline",
        "habitat_llm.evaluation.metrics.rebound.stage_baseline",
    ],
    ids=("canonical", "legacy"),
)
def test_stage_baseline_cli_help_smoke(module_name):
    result = _run_lightweight_python(
        f"""
import runpy
import sys

sys.argv = [{module_name!r}, "--help"]
runpy.run_module({module_name!r}, run_name="__main__")
"""
    )
    assert result.returncode == 0, result.stderr
    assert "Estimate clean stage baselines" in result.stdout
