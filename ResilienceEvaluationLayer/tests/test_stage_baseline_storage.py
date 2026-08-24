import json
from pathlib import Path
import sys
import types

import pytest
import yaml

try:
    import omegaconf  # noqa: F401
except ModuleNotFoundError:
    # The repository runtime depends on OmegaConf, but this focused storage
    # test can run in a lightweight environment because it passes plain dicts.
    module = types.ModuleType("omegaconf")

    class DictConfig(dict):
        pass

    class OmegaConf:
        pass

    module.DictConfig = DictConfig
    module.OmegaConf = OmegaConf
    sys.modules["omegaconf"] = module

# Avoid importing habitat_llm.evaluation.__init__, which eagerly imports the
# full Habitat simulator. These tests only exercise the lightweight evaluation
# subpackages and provide their normal filesystem package paths directly.
_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
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
if "habitat_llm.evaluation.core" not in sys.modules:
    core_package = types.ModuleType("habitat_llm.evaluation.core")
    core_package.__path__ = [str(_PACKAGE_ROOT / "evaluation" / "core")]
    sys.modules["habitat_llm.evaluation.core"] = core_package

from habitat_llm.evaluation.metrics.rebound import (
    StageBaseline,
    StageBaselineDim,
    StageBaselineEstimator,
)
from habitat_llm.evaluation.critic_export_contract import load_full_stability_export
from habitat_llm.evaluation.stage_baseline import StageBaselineEvidenceStore
from habitat_llm.evaluation.perturbation import PerturbationSpec
from habitat_llm.examples.resilience_helpers import (
    build_stage_baseline_signature,
    collect_critic_exports,
    generate_stage_baseline,
    merge_spec_csvs,
    normalize_rollout_rows,
    plan_stage_baseline_calibration,
    run_boundary_aggregation,
    run_stability_aggregation,
    resolve_stage_baseline_startup_candidate,
    resolve_stage_baseline_seed_lifecycle,
    resolve_stage_baseline_evidence_store,
    resolve_persistent_hash_baseline_path,
    spec_output_dir,
    update_episode_aggregated_baselines,
    write_csv_rows,
)


def _write_full_critic_export(
    path: Path,
    *,
    episode_id: str = "78",
    judge_model: str = "gpt-5.1",
    state_encoder_id: str = "encoder-a",
    value_checkpoint_id: str = "value-a",
) -> Path:
    payload = {
        "episode_id": episode_id,
        "task_family": "rearrange",
        "critic_phase": "reference",
        "judge_model": judge_model,
        "state_encoder_id": state_encoder_id,
        "value_checkpoint_id": value_checkpoint_id,
        "reward_shaper_valid": True,
        "critic_lifecycle_valid": True,
        "export_manifest": {
            "full_stability_trajectory": True,
            "minimal_schema": True,
            "record_mode": "state_only",
        },
        "selection_summary": {
            "total_transitions": 1,
            "selected_transitions": 1,
        },
        "records": [
            {
                "episode_id": episode_id,
                "transition_index": 0,
                "state_kind": "state",
                "task_family": "rearrange",
                "task_state_success": 1.0,
                "task_percent_complete": 1.0,
                "proposition_satisfied_fraction": 1.0,
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _config(
    judge_model: str,
    *,
    dataset_path: str = "data/datasets/partnr_episodes/v0_0/val_mini.json.gz",
    planner_engine: str = "Qwen3-8B-Instruct",
) -> dict:
    return {
        "habitat": {
            "dataset": {
                "data_path": dataset_path
            }
        },
        "episode_ids": [78],
        "evaluation": {
            "critic": {
                "llm_model": judge_model,
                "llm_base_url": "https://example.invalid/v1",
            },
            "adca": {
                "llm_model": judge_model,
                "llm_base_url": "https://example.invalid/v1",
            },
            "planner": {
                "_target_": "habitat_llm.planner.CentralizedLLMPlanner",
                "plan_config": {
                    "llm": {"generation_params": {"engine": planner_engine}}
                },
            },
        },
    }


def _resilience_cfg(store_dir: Path) -> dict:
    return {
        "output_subdir": "resilience",
        "baseline_json": "stage_baseline.json",
        "seeds": [0, 1],
        "intensities": [1.0],
        "stage_baseline": {
            "persistent_store_dir": str(store_dir),
            "runtime_subdir": "runtime",
            "statistics_subdir": "statistics",
            "archive_subdir": "archives",
            "episode_aggregate": {
                "enabled": True,
                "subdir": "by_episode",
                "merge_existing": True,
                "allow_partial_compose": True,
            },
        },
    }


def test_rollout_normalization_fails_closed_without_perturbation_audit():
    source = {
        "episode_id": "78",
        "perturbation_type": "surface_rewrite",
        "perturbation_realized": 0.5,
        "stress_lambda": 0.5,
    }

    normalized = normalize_rollout_rows([source])[0]

    assert normalized["stress_realized"] == 0.5
    assert normalized["stress_valid"] is False
    assert normalized["stress_invalid_reason"] == (
        "missing_runtime_perturbation_audit"
    )
    assert normalized["task_percent_complete"] == ""
    assert normalized["task_state_success"] == ""
    assert normalized["proposition_satisfied_fraction"] == ""
    assert normalized["task_outcome_valid"] is False
    assert "stress_valid" not in source


def test_rollout_normalization_prefers_frozen_stress_alignment_fields():
    source = {
        "episode_id": "78",
        "perturbation_type": "llm_instruction_context_stress",
        "stress_lambda": 0.4,
        # Generic values are deliberately inconsistent: the frozen bank
        # contract is authoritative for formal Stability/GE.
        "perturbation_realized": 0.9,
        "perturbation_valid": 1,
        "perturbation_stress_realized": 0.4,
        "perturbation_text_dose": 0.39,
        "perturbation_alignment_valid": 1,
    }

    normalized = normalize_rollout_rows([source])[0]

    assert normalized["stress_realized"] == 0.4
    assert normalized["stress_text_dose"] == 0.39
    assert normalized["stress_valid"] is True
    assert normalized["stress_alignment_valid"] is True


def test_stability_and_ge_share_the_same_complete_episode_seed_set(tmp_path):
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        "episode_id": "78",
                        "evaluation_propositions": [
                            {"function_name": "is_on_top", "args": {}}
                        ],
                        "evaluation_constraints": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    rows = []
    for seed, slope in ((0, 5.0), (1, 50.0)):
        for stress_lambda in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
            is_clean = stress_lambda == 0.0
            rows.append(
                {
                    "episode_id": "78",
                    "anchor_id": "78",
                    "judge_model": "gpt-5.1",
                    "stage_baseline_path": "/baseline/gpt-5.1/stage_baseline.json",
                    "stage_baseline_episode_id": "78",
                    "stage_baseline_episode_match": 1,
                    "state_encoder_id": "shared-state-encoder",
                    "reward_shaper_valid": 1,
                    "critic_lifecycle_valid": 1,
                    "perturbation_seed": seed,
                    "perturbation_type": (
                        "clean" if is_clean else "llm_instruction_context_stress"
                    ),
                    # Exercise the direct-runner input contract: normalization
                    # must derive stress_lambda from perturbation_intensity.
                    "perturbation_intensity": stress_lambda,
                    "perturbation_stress_family": (
                        "llm_instruction_context_stress"
                    ),
                    "perturbation_variant": f"variant_{seed}",
                    "perturbation_bank_path": "/banks/instruction_context_v1",
                    "perturbation_stress_realized": stress_lambda,
                    "perturbation_text_dose": stress_lambda,
                    "perturbation_alignment_valid": 1,
                    "perturbation_valid": 1,
                    "stability_trajectory_loss": slope * stress_lambda,
                    "stability_trajectory_loss_valid": 1,
                    "task_state_success": 1.0,
                    "task_percent_complete": 1.0,
                    "proposition_satisfied_fraction": 1.0,
                    "action_sequence": "Navigate|Pick|Place",
                    "rebound_c_rec": 0.4,
                    # Seed 1 has a Stability path but is not a valid GE path.
                    "rebound_c_rec_valid": int(seed == 0),
                    "unsafe_rate": 0.0,
                }
            )

    aggregates = run_stability_aggregation(
        rows,
        tmp_path,
        {
            "stage_baseline": {"clean_tag": "clean"},
            "boundary_margin": {"min_clean_samples": 1},
        },
        dataset_path=str(dataset_path),
    )

    assert aggregates["78"].n_paired_seeds == 1
    assert all(row["task_family"] == "placement_only" for row in rows)
    assert all(
        row["stability_shared_pair_eligible"] == int(row["perturbation_seed"] == 0)
        for row in rows
    )
    excluded = [row for row in rows if row["perturbation_seed"] == 1]
    assert all(row["stability_beta_at_lambda_valid"] == 0 for row in excluded)
    assert all(
        row["stability_beta_at_lambda_missing_reason"]
        == "excluded_from_shared_formal_pair_set"
        for row in excluded
    )


def test_critic_exports_are_limited_to_current_attempt_csv(tmp_path):
    spec = PerturbationSpec(kind="clean", seed=0, intensity=0.0)
    spec_dir = spec_output_dir(tmp_path, spec)
    old_export = (
        spec_dir
        / "workers"
        / "attempt_old"
        / "episode_78"
        / "analyses"
        / "critic_exports"
        / "critic_export-episode_78_old.json"
    )
    old_export.parent.mkdir(parents=True)
    old_export.write_text('{"episode_id": "78"}', encoding="utf-8")
    current_csv = spec_dir / "episode_metrics.csv"
    write_csv_rows(current_csv, [{"episode_id": "78"}])

    assert collect_critic_exports([spec], tmp_path) == []
    merged = merge_spec_csvs([spec], [current_csv], tmp_path / "merged.csv")
    assert not merged[0].get("critic_export_path")

    current_export = (
        spec_dir
        / "workers"
        / "attempt_current"
        / "episode_78"
        / "analyses"
        / "critic_exports"
        / "critic_export-episode_78_current.json"
    )
    current_export.parent.mkdir(parents=True)
    current_export.write_text('{"episode_id": "78"}', encoding="utf-8")
    write_csv_rows(
        current_csv,
        [{"episode_id": "78", "critic_export_path": str(current_export)}],
    )

    assert collect_critic_exports([spec], tmp_path) == [current_export]


def _incoming_baseline(samples: int, tau_star: float):
    key = ("rearrange", 0, "planning")
    source_id = f"fixture-{samples}-{tau_star}"
    return {
        key: StageBaseline(
            family=key[0],
            stage=key[1],
            modality=key[2],
            tau_star=tau_star,
            # Keep the recovery-feature normalizers fixed so this fixture
            # isolates weighted summary merging of the legacy tau_star field.
            by_dim={
                "plan": StageBaselineDim(tau_bar=1.0),
                "sim": StageBaselineDim(tau_bar=1.0),
            },
            n_samples=samples,
            n_transitions=max(30, samples),
            n_trajectories=3,
            n_seeds=2,
            trajectory_ids=[f"{source_id}-trajectory-{index}" for index in range(3)],
            remaining_work_by_trajectory={
                f"{source_id}-trajectory-{index}": float(index + 1)
                for index in range(3)
            },
            seed_ids=[0, 1],
            source_hashes=[source_id],
            formal_valid=True,
        )
    }


def test_stage_baseline_signature_uses_reward_judge_model(tmp_path):
    resilience_cfg = _resilience_cfg(tmp_path)
    signature = build_stage_baseline_signature(
        config=_config("gpt-5.1"),
        resilience_cfg=resilience_cfg,
        clean_specs=[],
    )

    assert signature["judge_model"] == "gpt-5.1"
    assert signature["judge_model_slug"] == "gpt-5.1"
    assert signature["judge_model_source"] == "evaluation.critic.llm_model"
    assert signature["judge_base_url"] == "https://example.invalid/v1"
    assert signature["planner_identity"] == (
        "Qwen3-8B-Instruct|habitat_llm.planner.CentralizedLLMPlanner"
    )
    assert signature["evaluated_policy_signature"]
    assert signature["baseline_config_signature"]
    assert signature["stage_baseline_schema_version"] == 3
    assert signature["recovery_feature_version"] == "recovery_features_v3"
    assert signature["require_successful_clean"] is False


def test_value_checkpoint_id_does_not_split_stage_baseline_contract(tmp_path):
    resilience_cfg = _resilience_cfg(tmp_path / "persistent")
    first_config = _config("gpt-5.1")
    second_config = _config("gpt-5.1")
    for config, value_id in ((first_config, "value-a"), (second_config, "value-b")):
        config["evaluation"]["critic"].update(
            {
                "state_encoder_id": "encoder-a",
                "value_checkpoint_id": value_id,
            }
        )

    first = build_stage_baseline_signature(
        config=first_config,
        resilience_cfg=resilience_cfg,
        clean_specs=[],
    )
    second = build_stage_baseline_signature(
        config=second_config,
        resilience_cfg=resilience_cfg,
        clean_specs=[],
    )

    assert first["value_checkpoint_id"] != second["value_checkpoint_id"]
    assert first["baseline_contract_id"] == second["baseline_contract_id"]
    resolve_stage_baseline_evidence_store(
        resilience_cfg["stage_baseline"], first
    ).ensure_contract()
    resolve_stage_baseline_evidence_store(
        resilience_cfg["stage_baseline"], second
    ).ensure_contract()


def test_evidence_aligns_by_judge_and_episode_not_value_id(tmp_path):
    store = StageBaselineEvidenceStore(
        tmp_path / "gpt-5.1",
        contract_id="contract-a",
        contract_metadata={
            "judge_model": "gpt-5.1",
            "state_encoder_id": "encoder-a",
            "alignment_scope": "judge_model_episode",
        },
    )
    first = _write_full_critic_export(
        tmp_path / "seed0.json", value_checkpoint_id="value-a"
    )
    second = _write_full_critic_export(
        tmp_path / "seed1.json", value_checkpoint_id="value-b"
    )
    result = store.ingest_exports(
        [first, second],
        source_metadata={
            str(first.resolve()): {
                "seed": 0,
                "episode_id": "78",
                "judge_model": "gpt-5.1",
            },
            str(second.resolve()): {
                "seed": 1,
                "episode_id": "78",
                "judge_model": "gpt-5.1",
            },
        },
    )

    assert result["accepted_count"] == 2
    records = store.list_evidence()
    assert {record.value_checkpoint_id for record in records} == {
        "value-a",
        "value-b",
    }
    assert {record.alignment_key for record in records} == {"gpt-5.1::78"}
    assert all(record.value_alignment_role == "diagnostic_only" for record in records)


def test_evidence_rejects_wrong_judge_or_episode(tmp_path):
    store = StageBaselineEvidenceStore(
        tmp_path / "gpt-5.1",
        contract_id="contract-a",
        contract_metadata={
            "judge_model": "gpt-5.1",
            "state_encoder_id": "encoder-a",
            "alignment_scope": "judge_model_episode",
        },
    )
    wrong_judge = _write_full_critic_export(
        tmp_path / "wrong_judge.json", judge_model="gpt-4o"
    )
    wrong_episode = _write_full_critic_export(tmp_path / "wrong_episode.json")
    result = store.ingest_exports(
        [wrong_judge, wrong_episode],
        source_metadata={
            str(wrong_judge.resolve()): {
                "seed": 0,
                "episode_id": "78",
                "judge_model": "gpt-5.1",
            },
            str(wrong_episode.resolve()): {
                "seed": 1,
                "episode_id": "131",
                "judge_model": "gpt-5.1",
            },
        },
    )

    assert result["accepted_count"] == 0
    assert {item["reason"] for item in result["excluded"]} == {
        "judge_model_mismatch",
        "episode_id_mismatch",
    }
    assert load_full_stability_export(wrong_judge).metadata["judge_model"] == "gpt-4o"


def test_snapshot_latest_uses_newest_available_snapshot(tmp_path):
    store = StageBaselineEvidenceStore(
        tmp_path / "gpt-5.1",
        contract_id="contract-a",
        contract_metadata={"judge_model": "gpt-5.1"},
    )
    diagnostic = tmp_path / "diagnostic.json"
    diagnostic.write_text(
        json.dumps({"metadata": {"formal_valid": False}}),
        encoding="utf-8",
    )
    published = store.publish_snapshot(
        diagnostic,
        baseline_id="diagnostic-a",
        metadata={"formal_valid": True},
        # Simulate a legacy caller that promoted its pre-write lifecycle flag.
        update_latest=True,
    )

    assert store.latest_snapshot() == published


def test_partial_calibration_schedules_only_missing_fixed_seed_cells(
    tmp_path,
):
    stage_cfg = {
        "persistent_store_dir": str(tmp_path / "s"),
        "reuse_policy": "reuse_and_extend",
        "extension_samples_per_episode": 0,
        "formal": {"min_independent_trajectories": 3},
        "seed_lifecycle": {
            "initial_calibration_seeds": [0, 1, 2],
            "evaluation_seeds": [0, 1, 2],
            "allow_paired_seeds": True,
            "extension_seed_start": 3,
        },
    }
    signature = {
        "baseline_contract_id": "contract-a",
        "judge_model": "gpt-5.1",
        "judge_model_slug": "gpt-5.1",
        "state_encoder_id": "encoder-a",
        "episode_ids": ["78", "131"],
    }
    store = resolve_stage_baseline_evidence_store(stage_cfg, signature)
    exports = []
    provenance = {}
    for episode_id, seeds in (("78", (1, 2)), ("131", (0, 1, 2))):
        for seed in seeds:
            path = _write_full_critic_export(
                tmp_path / f"e{episode_id}s{seed}.json",
                episode_id=episode_id,
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["calibration_seed_marker"] = seed
            path.write_text(json.dumps(payload), encoding="utf-8")
            exports.append(path)
            provenance[str(path.resolve())] = {
                "seed": seed,
                "episode_id": episode_id,
                "judge_model": "gpt-5.1",
                "reward_shaper_valid": True,
                "critic_lifecycle_valid": True,
            }
    assert store.ingest_exports(
        exports, source_metadata=provenance
    )["accepted_count"] == 5

    diagnostic = tmp_path / "partial_stage_baseline.json"
    diagnostic.write_text(
        json.dumps(
            {
                "metadata": {
                    "formal_valid": False,
                    "formal_episode_ids": ["78", "131"],
                },
                "stages": [
                    {
                        "family": "rearrange",
                        "stage": 0,
                        "modality": "__recovery__",
                        "formal_valid": False,
                        "formal_component_support": {
                            "remaining_work": {
                                "cells": [{"episode_id": "131"}]
                            }
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    store.publish_snapshot(
        diagnostic,
        baseline_id="partial-a",
        metadata={"formal_valid": False},
        update_latest=False,
    )

    first = plan_stage_baseline_calibration(
        stage_baseline_cfg=stage_cfg,
        signature=signature,
        episode_ids=["78", "131"],
    )
    assert first["scheduled"] == [
        {"seed": 0, "episode_ids": ["78"]}
    ]

    # No extension seeds are invented; an unrecorded fixed cell stays pending.
    second = plan_stage_baseline_calibration(
        stage_baseline_cfg=stage_cfg,
        signature=signature,
        episode_ids=["78", "131"],
    )
    assert second["scheduled"] == [
        {"seed": 0, "episode_ids": ["78"]}
    ]


def test_stability_lineage_uses_judge_episode_not_value_id(tmp_path):
    baseline_path = tmp_path / "stage_baseline.json"
    StageBaselineEstimator.write(
        _incoming_baseline(3, 2.0),
        baseline_path,
        metadata={
            "formal_valid": True,
            "baseline_contract_id": "contract-a",
            "state_encoder_id": "encoder-a",
            "value_checkpoint_id": "value-a",
            "judge_model": "gpt-5.1",
            "formal_episode_ids": ["78"],
        },
    )
    baseline_metadata = json.loads(
        baseline_path.read_text(encoding="utf-8")
    )["metadata"]
    export = _write_full_critic_export(
        tmp_path / "evaluation.json",
        value_checkpoint_id="value-b",
    )
    row = {
        "episode_id": "78",
        "anchor_id": "78",
        "task_family": "rearrange",
        "perturbation_type": "clean",
        "perturbation_seed": 3,
        "perturbation_intensity": 0.0,
        "stage_baseline_id": baseline_metadata["baseline_id"],
        "baseline_contract_id": "contract-a",
        "state_encoder_id": "encoder-a",
        "value_checkpoint_id": "value-b",
        "judge_model": "gpt-5.1",
        "reward_shaper_valid": 1,
        "critic_lifecycle_valid": 1,
        "critic_export_path": str(export),
        "task_percent_complete": 1.0,
        "proposition_satisfied_fraction": 1.0,
    }

    run_stability_aggregation(
        [row],
        tmp_path / "analysis",
        {},
        baseline_path=baseline_path,
    )

    assert row["stability_lineage_valid"] == 1
    assert row["stability_lineage_missing_reason"] == ""


def test_stage_baseline_seed_lifecycle_allows_overlap(tmp_path):
    resilience_cfg = _resilience_cfg(tmp_path)
    resilience_cfg["stage_baseline"]["seed_lifecycle"] = {
        "enabled": True,
        "calibration_seeds": [0, 1, 2],
        "evaluation_seeds": [2, 3, 4],
    }

    lifecycle = resolve_stage_baseline_seed_lifecycle(resilience_cfg)
    assert lifecycle["calibration_seeds"] == [0, 1, 2]
    assert lifecycle["evaluation_seeds"] == [2, 3, 4]


def test_stage_baseline_seed_lifecycle_allows_explicit_pairing(tmp_path):
    resilience_cfg = _resilience_cfg(tmp_path)
    resilience_cfg["stage_baseline"]["seed_lifecycle"] = {
        "enabled": True,
        "initial_calibration_seeds": [0, 1, 2],
        "evaluation_seeds": [0, 1, 2],
        "allow_paired_seeds": True,
    }

    lifecycle = resolve_stage_baseline_seed_lifecycle(resilience_cfg)

    assert lifecycle["calibration_seeds"] == [0, 1, 2]
    assert lifecycle["evaluation_seeds"] == [0, 1, 2]
    assert lifecycle["allow_paired_seeds"] is True


def test_stage_baseline_rejects_judge_override_that_differs_from_reward_critic(
    tmp_path,
):
    resilience_cfg = _resilience_cfg(tmp_path)
    resilience_cfg["stage_baseline"]["judge_model"] = "gpt-4o"

    with pytest.raises(ValueError, match="must match"):
        build_stage_baseline_signature(
            config=_config("gpt-5.1"),
            resilience_cfg=resilience_cfg,
            clean_specs=[],
        )
def test_stage_baseline_signature_changes_with_evaluated_planner():
    resilience_cfg = _resilience_cfg(Path("persistent"))

    qwen3 = build_stage_baseline_signature(
        config=_config("gpt-5.1", planner_engine="Qwen3-8B-Instruct"),
        resilience_cfg=resilience_cfg,
        clean_specs=[],
    )
    qwen35 = build_stage_baseline_signature(
        config=_config("gpt-5.1", planner_engine="Qwen3.5-9B-Instruct"),
        resilience_cfg=resilience_cfg,
        clean_specs=[],
    )

    assert qwen3["planner_config_signature"] != qwen35["planner_config_signature"]
    assert qwen3["signature"] != qwen35["signature"]
    assert (
        qwen3["episode_baseline_signature"]
        != qwen35["episode_baseline_signature"]
    )
    shard_cfg = _config("gpt-5.1", planner_engine="Qwen3-8B-Instruct")
    shard_cfg["episode_ids"] = [79]
    shard_resilience_cfg = _resilience_cfg(Path("persistent"))
    shard_resilience_cfg["seeds"] = [7]
    shard_signature = build_stage_baseline_signature(
        config=shard_cfg,
        resilience_cfg=shard_resilience_cfg,
        clean_specs=[],
    )
    assert qwen3["signature"] != shard_signature["signature"]
    assert (
        qwen3["episode_baseline_signature"]
        == shard_signature["episode_baseline_signature"]
    )
    relaxed_cfg = _resilience_cfg(Path("persistent"))
    relaxed_cfg["stage_baseline"]["formal"] = {
        "require_successful_clean": False,
    }
    relaxed_signature = build_stage_baseline_signature(
        config=_config("gpt-5.1", planner_engine="Qwen3-8B-Instruct"),
        resilience_cfg=relaxed_cfg,
        clean_specs=[],
    )
    assert qwen3["episode_baseline_signature"] == relaxed_signature[
        "episode_baseline_signature"
    ]


def test_run_local_baseline_requires_matching_metadata_signature(tmp_path):
    resilience_cfg = _resilience_cfg(tmp_path / "persistent")
    resilience_cfg["stage_baseline"]["evidence_store_enabled"] = False
    resilience_cfg["stage_baseline"]["episode_aggregate"]["enabled"] = False
    estimator = StageBaselineEstimator()
    run_local = tmp_path / "run" / "stage_baseline.json"
    original_signature = build_stage_baseline_signature(
        config=_config("gpt-5.1", planner_engine="Qwen3-8B-Instruct"),
        resilience_cfg=resilience_cfg,
        clean_specs=[],
    )
    estimator.write(
        _incoming_baseline(2, 2.0),
        run_local,
        metadata={"signature": original_signature},
    )

    matching = resolve_stage_baseline_startup_candidate(
        config=_config("gpt-5.1", planner_engine="Qwen3-8B-Instruct"),
        resilience_cfg=resilience_cfg,
        clean_specs=[],
        run_local_path=run_local,
    )
    mismatching = resolve_stage_baseline_startup_candidate(
        config=_config("gpt-5.1", planner_engine="Qwen3.5-9B-Instruct"),
        resilience_cfg=resilience_cfg,
        clean_specs=[],
        run_local_path=run_local,
    )

    assert matching["source"] == "run_local_hash"
    assert matching["run_local_hash_valid"] is True
    assert mismatching["source"] == "run_local_hash"
    assert mismatching["run_local_hash_valid"] is True


def test_explicit_override_requires_matching_judge_and_statistics(tmp_path):
    resilience_cfg = _resilience_cfg(tmp_path / "persistent")
    resilience_cfg["stage_baseline"]["seed_lifecycle"] = {
        "enabled": True,
        "calibration_seeds": [0, 1, 2],
        "evaluation_seeds": [3, 4, 5],
    }
    signature = build_stage_baseline_signature(
        config=_config("gpt-5.1"),
        resilience_cfg=resilience_cfg,
        clean_specs=[],
    )
    estimator = StageBaselineEstimator()
    formal_path = tmp_path / "formal.json"
    estimator.write(
        _incoming_baseline(30, 2.0),
        formal_path,
        metadata={"signature": signature, "formal_valid": True},
    )

    selected = resolve_stage_baseline_startup_candidate(
        config=_config("gpt-5.1"),
        resilience_cfg=resilience_cfg,
        clean_specs=[],
        run_local_path=tmp_path / "missing.json",
        baseline_override_path=formal_path,
    )
    assert selected["source"] == "explicit_override"

    overlapping_signature = dict(signature)
    overlapping_signature["calibration_seeds"] = [3, 4, 5]
    overlapping_path = tmp_path / "overlapping_seeds.json"
    estimator.write(
        _incoming_baseline(30, 2.0),
        overlapping_path,
        metadata={"signature": overlapping_signature, "formal_valid": True},
    )
    overlap_selected = resolve_stage_baseline_startup_candidate(
        config=_config("gpt-5.1"),
        resilience_cfg=resilience_cfg,
        clean_specs=[],
        run_local_path=tmp_path / "missing.json",
        baseline_override_path=overlapping_path,
    )
    assert overlap_selected["source"] == "explicit_override"

    diagnostic_path = tmp_path / "diagnostic.json"
    estimator.write(
        _incoming_baseline(30, 2.0),
        diagnostic_path,
        metadata={"signature": signature, "formal_valid": False},
    )
    diagnostic_selected = resolve_stage_baseline_startup_candidate(
        config=_config("gpt-5.1"),
        resilience_cfg=resilience_cfg,
        clean_specs=[],
        run_local_path=tmp_path / "missing.json",
        baseline_override_path=diagnostic_path,
    )
    assert diagnostic_selected["source"] == "explicit_override"


def test_generate_stage_baseline_preserves_frozen_artifacts(
    tmp_path,
    monkeypatch,
):
    resilience_cfg = _resilience_cfg(tmp_path / "persistent")
    resilience_cfg["stage_baseline"]["evidence_store_enabled"] = False
    resilience_cfg["baseline_json"] = "stage_baseline.json"
    resilience_cfg["stage_baseline"]["seed_lifecycle"] = {
        "enabled": True,
        "calibration_seeds": [0, 1, 2],
        "evaluation_seeds": [3, 4, 5],
    }
    clean_specs = [
        PerturbationSpec(kind="clean", seed=seed, intensity=0.0)
        for seed in (0, 1, 2)
    ]
    export_paths = [tmp_path / f"clean_seed{seed}.json" for seed in (0, 1, 2)]
    monkeypatch.setattr(
        "habitat_llm.examples.resilience_helpers.collect_critic_exports",
        lambda *_args, **_kwargs: export_paths,
    )
    monkeypatch.setattr(
        "habitat_llm.examples.resilience_helpers.collect_critic_export_provenance",
        lambda *_args, **_kwargs: {},
    )

    state = {"tau": 2.0, "seeds": [0, 1, 2]}

    def _fake_estimate(self, _paths, **_kwargs):
        trajectory_ids = [f"trajectory-{seed}" for seed in state["seeds"]]
        provenance = {
            trajectory_id: {
                "episode_id": "78",
                "seed": seed,
                "source_hash": f"hash-{state['tau']}-{seed}",
            }
            for trajectory_id, seed in zip(trajectory_ids, state["seeds"])
        }
        baseline = StageBaseline(
            family="rearrange",
            stage=0,
            modality="planning",
            tau_star=float(state["tau"]),
            n_samples=30,
            n_transitions=30,
            n_trajectories=len(trajectory_ids),
            n_seeds=len(state["seeds"]),
            trajectory_ids=trajectory_ids,
            seed_ids=list(state["seeds"]),
            source_hashes=sorted(
                record["source_hash"] for record in provenance.values()
            ),
            trajectory_provenance=provenance,
            remaining_work_by_trajectory={
                trajectory_id: 4.0 for trajectory_id in trajectory_ids
            },
            W_rem_star_by_episode_stage={"78": {0: 4.0}},
            formal_valid=len(trajectory_ids) >= 3,
        )
        baseline_map = {("rearrange", 0, "planning"): baseline}
        succeeded = [
            {
                "episode_id": "78",
                "seed": seed,
                "source_hash": f"hash-{state['tau']}-{seed}",
                "trajectory_id": f"trajectory-{seed}",
            }
            for seed in state["seeds"]
        ]
        stats = {
            "succeeded_trajectories": succeeded,
            "excluded_trajectories": [],
            "source_hashes": [row["source_hash"] for row in succeeded],
            "episode_export_counts": {"78": len(succeeded)},
            "episode_record_counts": {"78": 31 * len(succeeded)},
        }
        return baseline_map, {"78": baseline_map}, stats

    monkeypatch.setattr(
        StageBaselineEstimator,
        "estimate_grouped_from_exports",
        _fake_estimate,
    )
    config = _config("gpt-5.1")
    run_dir = tmp_path / "run"
    resilience_dir = run_dir / "resilience"

    first = generate_stage_baseline(
        base_results_dir=run_dir,
        resilience_output_dir=resilience_dir,
        resilience_cfg=resilience_cfg,
        clean_specs=clean_specs,
        config=config,
        update_episode_aggregate=False,
    )
    assert first == resilience_dir / "stage_baseline.json"
    signature = build_stage_baseline_signature(
        config=config,
        resilience_cfg=resilience_cfg,
        clean_specs=clean_specs,
    )
    canonical = resolve_persistent_hash_baseline_path(
        resilience_cfg["stage_baseline"],
        signature,
    )
    canonical_bytes = canonical.read_bytes()

    state["tau"] = 4.0
    second = generate_stage_baseline(
        base_results_dir=run_dir,
        resilience_output_dir=resilience_dir,
        resilience_cfg=resilience_cfg,
        clean_specs=clean_specs,
        config=config,
        update_episode_aggregate=False,
    )
    assert second != first
    assert canonical.read_bytes() == canonical_bytes
    assert list(canonical.parent.glob(f"{canonical.stem}__baseline-*.json"))

    latest_alias = canonical.parents[1] / "stage_baseline_latest.json"
    latest_formal_bytes = latest_alias.read_bytes()
    state["tau"] = 6.0
    state["seeds"] = [0, 1]
    diagnostic = generate_stage_baseline(
        base_results_dir=run_dir,
        resilience_output_dir=resilience_dir,
        resilience_cfg=resilience_cfg,
        clean_specs=clean_specs,
        config=config,
        update_episode_aggregate=False,
    )
    diagnostic_status = StageBaselineEstimator.artifact_status(
        diagnostic,
        require_formal=True,
    )
    assert diagnostic_status[0] is True
    assert diagnostic_status[1] == ""
    assert canonical.read_bytes() == canonical_bytes
    assert latest_alias.read_bytes() != latest_formal_bytes


def test_stage_baseline_id_is_independent_of_mapping_insertion_order(tmp_path):
    first_key = ("alpha", 0, "planning")
    second_key = ("beta", 1, "navigation")
    first = StageBaseline(
        family=first_key[0], stage=first_key[1], modality=first_key[2]
    )
    second = StageBaseline(
        family=second_key[0], stage=second_key[1], modality=second_key[2]
    )
    estimator = StageBaselineEstimator()
    left_path = tmp_path / "left.json"
    right_path = tmp_path / "right.json"
    estimator.write(
        {second_key: second, first_key: first},
        left_path,
        metadata={"formal_valid": True},
    )
    estimator.write(
        {first_key: first, second_key: second},
        right_path,
        metadata={"formal_valid": True},
    )

    left_payload = json.loads(left_path.read_text(encoding="utf-8"))
    right_payload = json.loads(right_path.read_text(encoding="utf-8"))
    assert left_payload["metadata"]["baseline_id"] == right_payload["metadata"][
        "baseline_id"
    ]
    assert left_payload["stages"] == right_payload["stages"]


def test_episode_baselines_isolate_same_stem_datasets_and_validate_merge(tmp_path):
    resilience_cfg = _resilience_cfg(tmp_path)
    estimator = StageBaselineEstimator()
    episode_stats = {
        "episode_export_counts": {"78": 1},
        "episode_record_counts": {"78": 2},
    }
    left_signature = build_stage_baseline_signature(
        config=_config(
            "gpt-5.1",
            dataset_path="data/split_a/shared.json.gz",
        ),
        resilience_cfg=resilience_cfg,
        clean_specs=[],
    )
    right_signature = build_stage_baseline_signature(
        config=_config(
            "gpt-5.1",
            dataset_path="data/split_b/shared.json.gz",
        ),
        resilience_cfg=resilience_cfg,
        clean_specs=[],
    )
    left = update_episode_aggregated_baselines(
        estimator=estimator,
        episode_baselines={"78": _incoming_baseline(2, 2.0)},
        episode_stats=episode_stats,
        stage_baseline_cfg=resilience_cfg["stage_baseline"],
        signature=left_signature,
    )
    right = update_episode_aggregated_baselines(
        estimator=estimator,
        episode_baselines={"78": _incoming_baseline(3, 3.0)},
        episode_stats=episode_stats,
        stage_baseline_cfg=resilience_cfg["stage_baseline"],
        signature=right_signature,
    )

    left_path = Path(left["episode_baseline_paths"]["78"])
    right_path = Path(right["episode_baseline_paths"]["78"])
    assert left_path != right_path
    assert StageBaselineEstimator.load(left_path)[
        ("rearrange", 0, "planning")
    ].n_samples == 2
    assert StageBaselineEstimator.load(right_path)[
        ("rearrange", 0, "planning")
    ].n_samples == 3

    payload = json.loads(left_path.read_text(encoding="utf-8"))
    payload["metadata"]["signature"]["episode_baseline_signature"] = "wrong"
    left_path.write_text(json.dumps(payload), encoding="utf-8")
    updated = update_episode_aggregated_baselines(
        estimator=estimator,
        episode_baselines={"78": _incoming_baseline(1, 1.0)},
        episode_stats=episode_stats,
        stage_baseline_cfg=resilience_cfg["stage_baseline"],
        signature=left_signature,
    )
    assert updated["episode_baseline_paths"]["78"]


def test_runtime_and_statistics_are_separate_and_judge_scoped(tmp_path):
    resilience_cfg = _resilience_cfg(tmp_path)
    resilience_cfg["stage_baseline"]["evidence_store_enabled"] = False
    estimator = StageBaselineEstimator()
    episode_stats = {
        "episode_export_counts": {"78": 1},
        "episode_record_counts": {"78": 6},
    }
    gpt51_signature = build_stage_baseline_signature(
        config=_config("gpt-5.1"),
        resilience_cfg=resilience_cfg,
        clean_specs=[],
    )

    first = update_episode_aggregated_baselines(
        estimator=estimator,
        episode_baselines={"78": _incoming_baseline(2, 2.0)},
        episode_stats=episode_stats,
        stage_baseline_cfg=resilience_cfg["stage_baseline"],
        signature=gpt51_signature,
        runtime_record_id="clean_seed0_lambda1.00",
    )
    second = update_episode_aggregated_baselines(
        estimator=estimator,
        episode_baselines={"78": _incoming_baseline(3, 4.0)},
        episode_stats=episode_stats,
        stage_baseline_cfg=resilience_cfg["stage_baseline"],
        signature=gpt51_signature,
        runtime_record_id="clean_seed1_lambda1.00",
    )

    runtime_dir = Path(first["episode_runtime_paths"]["78"]).parent
    runtime_files = sorted(runtime_dir.glob("stage_baseline__*.json"))
    assert len(runtime_files) == 2
    assert all("statistics" not in path.parts for path in runtime_files)

    statistics_path = Path(second["episode_baseline_paths"]["78"])
    assert statistics_path.parts[-2:] == ("episode_78", "stage_baseline.json")
    assert "by_episode" in statistics_path.parts
    assert any(part.startswith("val_mini.json__") for part in statistics_path.parts)
    assert any(part.startswith("identity-") for part in statistics_path.parts)
    merged = StageBaselineEstimator.load(statistics_path)
    merged_baseline = merged[("rearrange", 0, "planning")]
    assert merged_baseline.n_samples == 5
    assert merged_baseline.tau_star == 3.2

    statistics_payload = json.loads(statistics_path.read_text(encoding="utf-8"))
    assert statistics_payload["baseline_kind"] == "resilience_statistical_reference"
    assert statistics_payload["baseline_scope"] == "stage_anchor"
    assert statistics_payload["stages"][0]["baseline_scope"] == "stage_anchor"
    assert statistics_payload["stages"][0]["baseline_key"] == {
        "task_family": "rearrange",
        "stage": 0,
        "modality": "planning",
    }
    assert statistics_payload["metadata"]["judge_model"] == "gpt-5.1"
    assert statistics_payload["metadata"]["incoming_runtime_baseline_path"] in {
        str(path) for path in runtime_files
    }
    composed_parts = Path(first["episode_aggregate_composed_path"]).parts
    assert "composed" in composed_parts
    assert any(part.startswith("val_mini.json__") for part in composed_parts)
    assert any(part.startswith("identity-") for part in composed_parts)
    startup = resolve_stage_baseline_startup_candidate(
        config=_config("gpt-5.1"),
        resilience_cfg=resilience_cfg,
        clean_specs=[],
        run_local_path=tmp_path / "run_local" / "stage_baseline.json",
    )
    assert startup["source"] == "episode_aggregate_composed"
    assert startup["episode_aggregate_composed_exists"] is True
    assert startup["episode_aggregate_composed_valid"] is True

    gpt4o_signature = build_stage_baseline_signature(
        config=_config("gpt-4o"),
        resilience_cfg=resilience_cfg,
        clean_specs=[],
    )
    gpt4o = update_episode_aggregated_baselines(
        estimator=estimator,
        episode_baselines={"78": _incoming_baseline(7, 9.0)},
        episode_stats=episode_stats,
        stage_baseline_cfg=resilience_cfg["stage_baseline"],
        signature=gpt4o_signature,
        runtime_record_id="clean_seed0_lambda1.00",
    )
    gpt4o_path = Path(gpt4o["episode_baseline_paths"]["78"])
    assert "gpt-4o" in gpt4o_path.parts
    assert "gpt-5.1" not in gpt4o_path.parts
    gpt4o_merged = StageBaselineEstimator.load(gpt4o_path)
    assert gpt4o_merged[("rearrange", 0, "planning")].n_samples == 7


def test_llm_judge_comparison_config_has_symmetric_conditions():
    config_path = (
        Path(__file__).resolve().parents[1]
        / "conf"
        / "experiments"
        / "llm_judge_stability_compare.yaml"
    )
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    block = cfg["expq1"]["subexperiments"]["stability_judge_model_comparison"]
    conditions = {condition["name"]: condition for condition in block["conditions"]}

    assert set(conditions) == {"gpt_5_1", "gpt_4o"}
    assert block["summary_group_keys"][0] == "judge_model"
    assert block["summary_group_keys"][1] == "task_family"
    assert block["headline_metric"] == "stability_beta_family"
    assert block["comparison"]["reference"] == "gpt-5.1"
    assert block["comparison"]["candidate"] == "GPT-4o"
    assert block["comparison"]["output_filename"] == "judge_model_comparison.csv"
    assert block["comparison"]["pair_keys"] == [
        "episode_id",
        "perturbation_seed",
        "perturbation_type",
        "perturbation_intensity",
    ]
    grid = block["base_patch"]["resilience"]
    assert grid["perturbations"] == [
        "clean",
        "surface_rewrite",
        "filler_injection",
    ]
    assert grid["seeds"] == [0, 1, 2]
    shared_stage_baseline = cfg["expq1"]["shared"]["base_patch"]["resilience"][
        "stage_baseline"
    ]
    lifecycle = shared_stage_baseline["seed_lifecycle"]
    assert lifecycle == {
        "enabled": True,
        "initial_calibration_seeds": [0, 1, 2],
        "evaluation_seeds": [0, 1, 2],
        "allow_paired_seeds": True,
        "extension_seed_start": 3,
        "critic_training_seeds": [100, 101, 102],
    }
    assert shared_stage_baseline["episode_aggregate"]["allow_partial_compose"] is False
    for name, model in (("gpt_5_1", "gpt-5.1"), ("gpt_4o", "GPT-4o")):
        condition = conditions[name]
        assert condition["patch"]["evaluation"]["critic"]["llm_model"] == model
        assert condition["patch"]["evaluation"]["adca"]["llm_model"] == model
        assert (
            condition["patch"]["resilience"]["stage_baseline"]["judge_model"]
            == model
        )
        assert condition["phases"][0]["patch"] == {}


@pytest.mark.parametrize(
    "filename",
    ["expq1_ge.yaml", "expq1_ge_qwen35.yaml"],
)
def test_ge_configs_use_required_six_level_single_phase_stress_sweep(filename):
    config_path = (
        Path(__file__).resolve().parents[1]
        / "conf"
        / "experiments"
        / filename
    )
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    block = cfg["expq1"]["subexperiments"]["ge_capacity_validity"]
    shared_margin = cfg["expq1"]["shared"]["base_patch"]["resilience"][
        "boundary_margin"
    ]
    required_grid = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    assert shared_margin["stress_lambdas"] == required_grid
    lifecycle = cfg["expq1"]["shared"]["base_patch"]["resilience"][
        "stage_baseline"
    ]["seed_lifecycle"]
    assert lifecycle["initial_calibration_seeds"] == [0, 1, 2]
    assert lifecycle["evaluation_seeds"] == [0, 1, 2]
    assert lifecycle["allow_paired_seeds"] is True
    assert len(block["phases"]) == 1
    phase = block["phases"][0]
    resilience = phase["patch"]["resilience"]
    assert phase["label"] == "stress_sweep"
    assert resilience["mode"] == "margin"
    assert resilience["perturbations"] == [
        "clean",
        "llm_instruction_context_stress",
    ]
    assert resilience["seeds"] == [0, 1, 2]
    assert resilience["intensities"] == required_grid
    assert resilience["boundary_margin"]["stress_lambdas"] == required_grid


def test_boundary_results_separate_primary_statistics_and_diagnostics(tmp_path):
    analysis_dir = tmp_path / "result" / "analysis"
    cfg = _resilience_cfg(tmp_path / "persistent")
    cfg["stage_baseline"]["judge_model"] = "GPT-5.1"
    cfg["stage_baseline"]["judge_base_url"] = "https://example.invalid/v1"
    cfg["boundary_margin"] = {
        "min_clean_samples": 2,
        "reference_mode": "clean_quantile",
        "stress_lambdas": [0.0, 0.5],
    }
    rows = []
    for index in range(2):
        rows.append(
            {
                "episode_id": str(index),
                "anchor_id": str(index),
                "task_family": "placement_only",
                "perturbation_type": "clean",
                "stress_lambda": 0.0,
                "evolve_step": 0,
                "judge_model": "GPT-5.1",
                "judge_base_url": "https://example.invalid/v1",
                "task_percent_complete": 0.9,
                "task_state_success": 1,
                "proposition_satisfied_fraction": 0.9,
                "rebound_c_rec": 0.4,
                "rebound_c_rec_valid": 1,
                "stability_trajectory_loss": 0.1,
                "stability_trajectory_loss_valid": 1,
                "stability_beta_neighborhood": 0.1,
                "stability_beta_neighborhood_valid": 1,
                "unsafe_rate": 0.0,
            }
        )
    rows.append(
        {
            **rows[0],
            "episode_id": "0",
            "anchor_id": "0",
            "perturbation_type": "irrelevant_perturbation",
            "perturbation_valid": 1,
            "perturbation_realized": 0.5,
            "stress_lambda": 0.5,
            "task_percent_complete": 0.75,
            "proposition_satisfied_fraction": 0.75,
            "rebound_c_rec": 0.7,
            "stability_trajectory_loss": 0.2,
            "stability_beta_neighborhood": 0.2,
            "unsafe_rate": 0.02,
        }
    )

    run_stability_aggregation(rows, analysis_dir, cfg)
    run_boundary_aggregation(
        rows,
        analysis_dir,
        cfg,
        dataset_path="data/val_mini.json.gz",
    )

    assert (analysis_dir / "resilience_boundary_capacity.csv").exists()
    assert (analysis_dir / "statistics" / "ge_boundary_baseline.json").exists()
    assert (analysis_dir / "statistics" / "ge_boundary_cells.csv").exists()
    assert (
        analysis_dir / "diagnostics" / "ge_boundary_episode_diagnostics.csv"
    ).exists()
    persistent_root = tmp_path / "persistent" / "GPT-5.1"
    assert list(
        (persistent_root / "runtime" / "family_boundaries").rglob(
            "ge_boundary_clean_evidence__*.csv"
        )
    )
    assert (
        persistent_root
        / "statistics"
        / "family_boundaries"
        / "val_mini.json"
        / "ge_boundary_baseline_latest.json"
    ).exists()


def test_configured_judge_comparison_aligns_same_perturbation_rows(monkeypatch):
    hydra_module = types.ModuleType("hydra")
    hydra_core_module = types.ModuleType("hydra.core")
    hydra_config_module = types.ModuleType("hydra.core.hydra_config")

    class HydraConfig:
        @staticmethod
        def initialized():
            return False

    hydra_config_module.HydraConfig = HydraConfig
    monkeypatch.setitem(sys.modules, "hydra", hydra_module)
    monkeypatch.setitem(sys.modules, "hydra.core", hydra_core_module)
    monkeypatch.setitem(sys.modules, "hydra.core.hydra_config", hydra_config_module)

    runner_stub = types.ModuleType("habitat_llm.examples.planner_demo_resilience")
    runner_stub.DEFAULT_RESILIENCE_CFG = {}
    runner_stub.run_resilience_experiment = lambda *args, **kwargs: {}
    monkeypatch.setitem(
        sys.modules,
        "habitat_llm.examples.planner_demo_resilience",
        runner_stub,
    )

    import habitat_llm.examples.expq_helpers as expq_helpers

    build_configured_summary_comparison = (
        expq_helpers.build_configured_summary_comparison
    )

    rows = [
        {
            "judge_model": "gpt-5.1",
            "perturbation_type": "surface_rewrite",
            "perturbation_intensity": "1.0",
            "stability_beta_neighborhood_mean": 0.25,
        },
        {
            "judge_model": "gpt-4o",
            "perturbation_type": "surface_rewrite",
            "perturbation_intensity": "1.0",
            "stability_beta_neighborhood_mean": 0.4,
        },
    ]
    comparison = build_configured_summary_comparison(
        rows,
        {
            "group_key": "judge_model",
            "reference": "gpt-5.1",
            "candidate": "gpt-4o",
            "match_keys": ["perturbation_type", "perturbation_intensity"],
            "statistic": "mean",
            "metrics": ["stability_beta_neighborhood"],
        },
    )

    assert len(comparison) == 1
    result = comparison[0]
    assert result["stability_beta_neighborhood_reference"] == 0.25
    assert result["stability_beta_neighborhood_candidate"] == 0.4
    assert abs(
        result["stability_beta_neighborhood_delta_candidate_minus_reference"]
        - 0.15
    ) < 1.0e-12
    assert abs(result["stability_beta_neighborhood_relative_delta"] - 0.6) < 1.0e-12

    paired_rows = [
        {
            "judge_model": "gpt-5.1",
            "episode_id": 78,
            "perturbation_seed": 0,
            "perturbation_type": "surface_rewrite",
            "perturbation_intensity": 1.0,
            "task_family": "placement_only",
            "task_state_success": 1.0,
        },
        {
            "judge_model": "gpt-4o",
            "episode_id": 78,
            "perturbation_seed": 0,
            "perturbation_type": "surface_rewrite",
            "perturbation_intensity": 1.0,
            "task_family": "placement_only",
            "task_state_success": 0.0,
        },
        {
            "judge_model": "gpt-5.1",
            "episode_id": 304,
            "perturbation_seed": 0,
            "perturbation_type": "surface_rewrite",
            "perturbation_intensity": 1.0,
            "task_family": "mixed_temporal",
            "task_state_success": 1.0,
        },
        {
            "judge_model": "gpt-4o",
            "episode_id": 131,
            "perturbation_seed": 0,
            "perturbation_type": "surface_rewrite",
            "perturbation_intensity": 1.0,
            "task_family": "spatial_relation",
            "task_state_success": 1.0,
        },
    ]
    paired = expq_helpers.build_paired_rollout_comparison(
        paired_rows,
        {
            "group_key": "judge_model",
            "reference": "gpt-5.1",
            "candidate": "gpt-4o",
            "metrics": ["task_state_success"],
        },
    )

    assert len(paired) == 3
    assert {row["pairing_status"] for row in paired} == {
        "paired",
        "reference_only",
        "candidate_only",
    }
    paired_78 = next(row for row in paired if row["episode_id"] == "78")
    assert paired_78["task_state_success_delta_candidate_minus_reference"] == -1.0
    assert paired_78["paired_pair_count"] == 1
    assert paired_78["pair_coverage_reference"] == 0.5
    assert paired_78["pair_coverage_candidate"] == 0.5
    assert paired_78["pair_coverage_union"] == pytest.approx(1.0 / 3.0)

    # Matching an episode id alone is insufficient: seed, perturbation kind,
    # and intensity are all part of the experimental cell identity.
    misaligned = expq_helpers.build_paired_rollout_comparison(
        [
            {
                "judge_model": "gpt-5.1",
                "episode_id": 78,
                "perturbation_seed": 0,
                "perturbation_type": "surface_rewrite",
                "perturbation_intensity": 1.0,
                "task_state_success": 1.0,
            },
            {
                "judge_model": "gpt-4o",
                "episode_id": 78,
                "perturbation_seed": 1,
                "perturbation_type": "surface_rewrite",
                "perturbation_intensity": 1.0,
                "task_state_success": 1.0,
            },
            {
                "judge_model": "gpt-4o",
                "episode_id": 78,
                "perturbation_seed": 0,
                "perturbation_type": "filler_injection",
                "perturbation_intensity": 1.0,
                "task_state_success": 1.0,
            },
            {
                "judge_model": "gpt-4o",
                "episode_id": 78,
                "perturbation_seed": 0,
                "perturbation_type": "surface_rewrite",
                "perturbation_intensity": 0.5,
                "task_state_success": 1.0,
            },
        ],
        {
            "group_key": "judge_model",
            "reference": "gpt-5.1",
            "candidate": "gpt-4o",
            "pair_keys": list(expq_helpers.DEFAULT_JUDGE_PAIR_KEYS),
            "metrics": ["task_state_success"],
        },
    )
    assert len(misaligned) == 4
    assert all(row["pairing_status"] != "paired" for row in misaligned)
    assert all(row["paired_pair_count"] == 0 for row in misaligned)

    invalid_key = expq_helpers.build_paired_rollout_comparison(
        [
            {
                "judge_model": "gpt-5.1",
                "episode_id": "",
                "perturbation_seed": 0,
                "perturbation_type": "surface_rewrite",
                "perturbation_intensity": 1.0,
                "task_state_success": 1.0,
            },
            {
                "judge_model": "gpt-4o",
                "episode_id": "",
                "perturbation_seed": 0,
                "perturbation_type": "surface_rewrite",
                "perturbation_intensity": 1.0,
                "task_state_success": 1.0,
            },
        ],
        {
            "group_key": "judge_model",
            "reference": "gpt-5.1",
            "candidate": "gpt-4o",
            "metrics": ["task_state_success"],
        },
    )
    assert len(invalid_key) == 2
    assert all(row["pair_key_valid"] is False for row in invalid_key)
    assert all(row["paired_pair_count"] == 0 for row in invalid_key)
    assert {row["pairing_status"] for row in invalid_key} == {
        "reference_invalid_pair_key",
        "candidate_invalid_pair_key",
    }

    duplicate_key = expq_helpers.build_paired_rollout_comparison(
        [paired_rows[0], dict(paired_rows[0]), paired_rows[1]],
        {
            "group_key": "judge_model",
            "reference": "gpt-5.1",
            "candidate": "gpt-4o",
            "metrics": ["task_state_success"],
        },
    )
    assert len(duplicate_key) == 1
    duplicate = duplicate_key[0]
    assert duplicate["pairing_status"] == "duplicate_pair_key"
    assert duplicate["paired_pair_count"] == 0
    assert duplicate["reference_duplicate_pair_key_count"] == 1
    assert duplicate["task_state_success_delta_candidate_minus_reference"] == ""
