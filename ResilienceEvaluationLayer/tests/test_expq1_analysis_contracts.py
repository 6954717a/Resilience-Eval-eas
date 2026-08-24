import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "evaluation"
    / "analysis"
    / "analyze_expq1_results.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "_expq1_analysis_contracts_target", _MODULE_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
analysis = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = analysis
_SPEC.loader.exec_module(analysis)


def _ge_fixture(tmp_path: Path, stress_lambdas: list[float]) -> analysis.RunSpec:
    root = tmp_path / "run"
    ge_root = root / "results" / "expq1" / analysis.GE_BLOCK
    metrics_dir = (
        ge_root
        / "conditions"
        / "default"
        / "runner"
        / "stress_sweep"
        / "spec_llm_instruction_context_stress_seed3_lambda0.200000"
    )
    metrics_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "episode_id": list(range(len(stress_lambdas))),
            "stress_lambda": stress_lambdas,
            "task_percent_complete": [1.0] * len(stress_lambdas),
        }
    ).to_csv(metrics_dir / "episode_metrics.csv", index=False)

    for path in [
        ge_root / "summary.csv",
        ge_root / "raw_rollouts.csv",
        ge_root / "ge_boundary_cells.csv",
        ge_root / "ge_boundary_curve_statistics.csv",
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("value\n1\n", encoding="utf-8")
    return analysis.RunSpec(model="model", root=root)


def test_ge_two_point_sweep_is_diagnostic_only(tmp_path: Path) -> None:
    _, diagnostic = analysis.load_ge_partial(_ge_fixture(tmp_path, [0.0, 1.0]))

    assert diagnostic["formal_ge_estimable"] is False
    assert diagnostic["stress_grid_complete"] is False
    assert diagnostic["missing_stress_lambdas"] == [0.2, 0.4, 0.6, 0.8]
    assert diagnostic["stress_lambda_coverage"] == pytest.approx(2.0 / 6.0)
    assert "requires_multi_stress_sweep" in diagnostic["incomplete_reason"]


def test_ge_complete_six_point_sweep_is_formally_estimable(tmp_path: Path) -> None:
    _, diagnostic = analysis.load_ge_partial(
        _ge_fixture(tmp_path, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    )

    assert diagnostic["formal_ge_estimable"] is True
    assert diagnostic["stress_grid_complete"] is True
    assert diagnostic["missing_stress_lambdas"] == []
    assert diagnostic["stress_lambda_coverage"] == pytest.approx(1.0)
    assert diagnostic["incomplete_reason"] == ""


def test_rebound_index_is_always_derived_from_raw_cost() -> None:
    prepared = analysis.prepare_rebound(
        pd.DataFrame(
            {
                "rebound_c_rec": [1.0, 9.0, np.nan, -1.0],
                "rebound_c_rec_index_100": [99.0, 90.0, 80.0, -5.0],
                "rebound_c_rec_valid": [1, 1, 1, 1],
            }
        )
    )

    assert prepared.loc[0, "rebound_c_rec_index_100"] == pytest.approx(50.0)
    assert prepared.loc[1, "rebound_c_rec_index_100"] == pytest.approx(90.0)
    assert pd.isna(prepared.loc[2, "rebound_c_rec_index_100"])
    assert pd.isna(prepared.loc[3, "rebound_c_rec_index_100"])
    assert prepared.loc[0, "rebound_c_rec_index_100_artifact"] == 99.0
    assert prepared.loc[0, "rebound_c_rec_index_100_artifact_status"] == "mismatch"
    assert prepared.loc[1, "rebound_c_rec_index_100_artifact_status"] == "matches_raw"
    assert prepared.loc[2, "rebound_c_rec_index_100_artifact_status"] == "raw_unavailable"
    assert prepared.loc[3, "rebound_c_rec_index_100_artifact_status"] == "invalid_range"

    extreme = analysis.prepare_rebound(
        pd.DataFrame(
            {
                "rebound_c_rec": [np.finfo(float).max],
                "rebound_c_rec_valid": [1],
            }
        )
    ).loc[0, "rebound_c_rec_index_100"]
    assert np.isfinite(extreme)
    assert 0.0 <= extreme < 100.0


def test_judge_pairing_uses_intensity_and_reports_missing_keys(tmp_path: Path) -> None:
    formal = pd.DataFrame(
        [
            {
                "model": "A",
                "episode_id": 1,
                "perturbation_seed": 3,
                "perturbation_type": "surface_rewrite",
                "perturbation_intensity": 0.25,
                "c_rec_observed": 1.0,
            },
            {
                "model": "A",
                "episode_id": 1,
                "perturbation_seed": 3,
                "perturbation_type": "surface_rewrite",
                "perturbation_intensity": 0.5,
                "c_rec_observed": 100.0,
            },
            {
                "model": "A",
                "episode_id": 2,
                "perturbation_seed": 3,
                "perturbation_type": "surface_rewrite",
                "perturbation_intensity": 0.25,
                "c_rec_observed": 2.0,
            },
            {
                "model": "B",
                "episode_id": 1,
                "perturbation_seed": 3,
                "perturbation_type": "surface_rewrite",
                "perturbation_intensity": 0.25,
                "c_rec_observed": 2.0,
            },
            {
                "model": "B",
                "episode_id": 2,
                "perturbation_seed": 3,
                "perturbation_type": "surface_rewrite",
                "perturbation_intensity": 0.25,
                "c_rec_observed": 3.0,
            },
        ]
    )

    tests = analysis.build_stat_tests(
        formal,
        pd.DataFrame(),
        ["A", "B"],
        tmp_path,
    )
    row = tests[
        (tests["context"] == "rebound_model_comparison/surface_rewrite")
        & (tests["metric"] == "c_rec_observed")
    ].iloc[0]

    assert json.loads(row["pair_key_columns"]) == list(analysis.REBOUND_PAIR_COLUMNS)
    assert row["n_unique_pair_keys_a"] == 3
    assert row["n_unique_pair_keys_b"] == 2
    assert row["n_pairs"] == 2
    assert row["n_missing_pair_keys_in_b"] == 1
    assert row["pair_key_coverage"] == pytest.approx(2.0 / 3.0)
    assert json.loads(row["missing_episode_ids_in_b"]) == ["1"]
    missing_key = json.loads(row["missing_pair_keys_in_b"])[0]
    assert missing_key["perturbation_intensity"] == pytest.approx(0.5)


def test_pairing_excludes_missing_and_duplicate_cells() -> None:
    rows = pd.DataFrame(
        [
            ["A", "1", 3, "surface", 0.25, 1.0],
            ["A", "1", 3, "surface", 0.25, 1.1],
            ["B", 1, 3.0, "surface", 0.25, 2.0],
            ["A", 2, 3, "surface", 0.25, 3.0],
            ["B", "2", 3, "surface", 0.25, 4.0],
            ["A", 3, 3, "surface", np.nan, 5.0],
            ["B", 3, 3, "surface", np.nan, 6.0],
        ],
        columns=[
            "model",
            "episode_id",
            "perturbation_seed",
            "perturbation_type",
            "perturbation_intensity",
            "c_rec_observed",
        ],
    )

    result = analysis.paired_or_unpaired_test(
        rows,
        group_col="model",
        group_a="A",
        group_b="B",
        metric="c_rec_observed",
        pair_cols=analysis.REBOUND_PAIR_COLUMNS,
        context="duplicate_and_missing",
    )

    assert result["n_pairs"] == 1
    assert result["n_duplicate_pair_keys_a"] == 1
    assert result["n_duplicate_pair_keys_b"] == 0
    assert result["n_invalid_pair_rows_a"] == 1
    assert result["n_invalid_pair_rows_b"] == 1
    assert result["test_name"] == "invalid_pairing_cells"
    assert result["note"] == "pair_key_values_missing;duplicate_pair_keys"
    duplicate = json.loads(result["duplicate_pair_keys_a"])[0]
    assert duplicate["episode_id"] == "1"
