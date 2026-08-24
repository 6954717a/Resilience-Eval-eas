#!/usr/bin/env python3

import importlib.util
import sys
import types
from pathlib import Path


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _bootstrap_ext_modules():
    repo_root = Path(__file__).resolve().parents[2]
    ext_root = repo_root / "habitat_llm" / "evaluation" / "metrics" / "extensibility"
    pkg_paths = {
        "habitat_llm": repo_root / "habitat_llm",
        "habitat_llm.evaluation": repo_root / "habitat_llm" / "evaluation",
        "habitat_llm.evaluation.metrics": repo_root / "habitat_llm" / "evaluation" / "metrics",
        "habitat_llm.evaluation.metrics.extensibility": ext_root,
    }
    for name, path in pkg_paths.items():
        module = sys.modules.get(name)
        if module is None:
            module = types.ModuleType(name)
            module.__path__ = [str(path)]
            sys.modules[name] = module
    boundary_module = _load_module(
        "habitat_llm.evaluation.metrics.extensibility.boundary_margin",
        ext_root / "boundary_margin.py",
    )
    collector_module = _load_module(
        "habitat_llm.evaluation.metrics.extensibility.boundary_collector",
        ext_root / "boundary_collector.py",
    )
    ext_pkg = sys.modules["habitat_llm.evaluation.metrics.extensibility"]
    ext_pkg.BoundaryMargin = boundary_module.BoundaryMargin
    ext_pkg.BoundaryThresholds = boundary_module.BoundaryThresholds
    ext_pkg.BoundaryCollector = collector_module.BoundaryCollector
    return (
        boundary_module.BoundaryMargin,
        collector_module.BoundaryCollector,
        boundary_module.BoundaryThresholds,
    )


BoundaryMargin, BoundaryCollector, BoundaryThresholds = _bootstrap_ext_modules()
from habitat_llm.evaluation.metrics.statistical_baseline import (
    read_statistical_baselines,
    write_statistical_baselines,
)


def test_ge_consumes_raw_rebound_cost_not_bounded_display_index():
    missing = []
    sources = {}

    c_rec = BoundaryCollector._parse_c_rec(
        {
            "rebound_c_rec": 2.5,
            "rebound_c_rec_index_100": 99.0,
            "rebound_c_rec_valid": 1,
        },
        missing,
        sources,
    )

    assert c_rec == 2.5
    assert sources == {"c_rec": "rebound_c_rec"}
    assert missing == []

    missing = []
    sources = {}
    index_only = BoundaryCollector._parse_c_rec(
        {
            "rebound_c_rec_index_100": 80.0,
            "rebound_c_rec_valid": 1,
        },
        missing,
        sources,
    )
    assert index_only is None
    assert sources == {}
    assert missing == ["missing_rebound_c_rec"]


def test_ge_lineage_uses_judge_episode_and_ignores_value_checkpoint_id():
    collector = BoundaryCollector(
        thresholds=BoundaryThresholds(
            min_clean_samples=1,
            stress_lambdas=(0.0, 0.5),
        )
    )
    base = {
        "episode_id": "78",
        "anchor_id": "78",
        "task_family": "placement_only",
        "evolve_step": 0,
        "judge_model": "gpt-5.1",
        "stage_baseline_id": "baseline-a",
        "baseline_contract_id": "contract-a",
        "state_encoder_id": "encoder-a",
        "stage_baseline_episode_id": "78",
        "stage_baseline_episode_match": 1,
        "reward_shaper_valid": 1,
        "critic_lifecycle_valid": 1,
        "task_percent_complete": 0.9,
        "proposition_satisfied_fraction": 0.9,
        "rebound_c_rec": 0.4,
        "rebound_c_rec_valid": 1,
        "stability_beta_at_lambda": 0.1,
        "stability_beta_at_lambda_valid": 1,
        "unsafe_rate": 0.0,
    }
    rows = [
        {
            **base,
            "perturbation_type": "clean",
            "stress_lambda": 0.0,
            "value_checkpoint_id": "value-a",
        },
        {
            **base,
            "perturbation_type": "irrelevant_perturbation",
            "perturbation_valid": 1,
            "perturbation_realized": 0.5,
            "stress_lambda": 0.5,
            "value_checkpoint_id": "value-b",
        },
    ]

    collector.aggregate(rows)

    assert all(row.valid for row in collector.episode_evidence)
    assert all(
        "mixed_baseline_lineage" not in row.missing_reasons
        for row in collector.episode_evidence
    )


def test_boundary_collector_does_not_double_count_collision_in_cbf_load():
    collector = BoundaryCollector(thresholds=BoundaryThresholds())
    rows = [
        {
            "episode_id": "78",
            "anchor_id": "78",
            "task_family": "placement_only",
            "stress_lambda": 0.5,
            "evolve_step": 0,
            "task_percent_complete": 0.8,
            "proposition_satisfied_fraction": 0.75,
            "rebound_c_rec": 1.2,
            "rebound_c_rec_valid": 1,
            "stability_beta_at_lambda": 0.12,
            "stability_beta_at_lambda_valid": 1,
            "safety_collision_rate": 0.02,
            "stability_p_cbf": 4.0,
            "total_step_count": 100,
        }
    ]
    margins = collector.aggregate(rows)

    assert len(margins) == 1
    evidence = collector.episode_evidence[0]
    assert evidence.valid
    assert evidence.safety_load == 0.04
    assert evidence.metric_sources["unsafe_rate"] == "stability_p_cbf/total_step_count"
    evidence_dict = evidence.to_dict()
    assert evidence_dict["ge_episode_margin"] == evidence.episode_margin
    assert evidence_dict["ge_episode_hard_gate_pass"] == int(evidence.hard_gate_pass)
    assert "metric_source_map" not in evidence_dict
    assert "pc_ratio_raw" not in evidence_dict
    evidence_diag = evidence.to_diagnostic_dict()
    assert evidence_diag["metric_source_map"]["unsafe_rate"] == "stability_p_cbf/total_step_count"
    margin = next(iter(margins.values()))
    assert margin.cell_valid
    assert margin.n_episodes == 1


def test_boundary_collector_rejects_proxy_when_formal_stability_is_missing():
    collector = BoundaryCollector(thresholds=BoundaryThresholds())
    rows = [
        {
            "episode_id": "79",
            "anchor_id": "79",
            "task_family": "placement_only",
            "stress_lambda": 0.2,
            "evolve_step": 0,
            "task_percent_complete": 0.85,
            "proposition_satisfied_fraction": 0.8,
            "rebound_c_rec": 0.9,
            "rebound_c_rec_valid": 1,
            "stability_value_delta_variance": 0.18,
            "unsafe_rate": 0.01,
        }
    ]
    collector.aggregate(rows)

    evidence = collector.episode_evidence[0]
    assert not evidence.valid
    assert evidence.stability_load is None
    assert "missing_formal_stability_beta_at_lambda" in evidence.missing_reasons


def test_boundary_collector_rejects_explicitly_invalid_perturbation():
    collector = BoundaryCollector(thresholds=BoundaryThresholds())
    collector.aggregate(
        [
            {
                "episode_id": "invalid-stress",
                "anchor_id": "invalid-stress",
                "task_family": "placement_only",
                "perturbation_type": "irrelevant_perturbation",
                "perturbation_valid": 0,
                "perturbation_reason": "planner_visible_change_not_observed",
                "stress_lambda": 0.5,
                "evolve_step": 0,
                "task_percent_complete": 0.8,
                "proposition_satisfied_fraction": 0.8,
                "rebound_c_rec": 1.0,
                "rebound_c_rec_valid": 1,
                "stability_beta_at_lambda": 0.1,
                "stability_beta_at_lambda_valid": 1,
                "unsafe_rate": 0.01,
            }
        ]
    )

    evidence = collector.episode_evidence[0]
    assert not evidence.valid
    assert "invalid_perturbation_realization" in evidence.missing_reasons
    assert (
        evidence.metric_sources["perturbation_realization"]
        == "planner_visible_change_not_observed"
    )


def test_boundary_collector_rejects_nonclean_row_without_runtime_audit():
    collector = BoundaryCollector(thresholds=BoundaryThresholds())
    collector.aggregate(
        [
            {
                "episode_id": "missing-audit",
                "anchor_id": "missing-audit",
                "task_family": "placement_only",
                "perturbation_type": "surface_rewrite",
                "stress_lambda": 0.5,
                "evolve_step": 0,
                "task_percent_complete": 0.8,
                "proposition_satisfied_fraction": 0.8,
                "rebound_c_rec": 1.0,
                "rebound_c_rec_valid": 1,
                "stability_beta_at_lambda": 0.1,
                "stability_beta_at_lambda_valid": 1,
                "unsafe_rate": 0.01,
            }
        ]
    )

    evidence = collector.episode_evidence[0]
    assert not evidence.valid
    assert "invalid_perturbation_realization" in evidence.missing_reasons
    assert (
        evidence.metric_sources["perturbation_realization"]
        == "missing_runtime_perturbation_audit"
    )


def test_invalid_rebound_row_creates_invalid_cell():
    collector = BoundaryCollector(thresholds=BoundaryThresholds(stress_lambdas=(0.0, 0.5, 1.0)))
    rows = [
        {
            "episode_id": "80",
            "anchor_id": "80",
            "task_family": "placement_only",
            "stress_lambda": 0.5,
            "evolve_step": 0,
            "task_percent_complete": 0.7,
            "proposition_satisfied_fraction": 0.7,
            "rebound_c_rec_valid": 0,
            "rebound_c_rec_missing_reason": "stage_baseline_missing",
            "stability_beta_at_lambda": 0.1,
            "stability_beta_at_lambda_valid": 1,
            "unsafe_rate": 0.01,
        }
    ]
    margins = collector.aggregate(rows)
    collector.finalize_audc(margins)

    assert collector.invalid_rows
    cell = margins[("placement_only", 0.5, 0)]
    assert not cell.cell_valid
    assert cell.cell_missing_reason == "stage_baseline_missing"
    assert cell.boundary_lambda_star is None
    assert collector.capacity_rows[0]["ge_valid_lambda_coverage"] == 0.0
    assert collector.capacity_rows[0]["ge_capacity_valid"] == 0
    assert collector.capacity_rows[0]["ge_audc_f"] == ""


def test_boundary_margin_uses_bottom_k_soft_margin_and_hard_gate():
    thresholds = BoundaryThresholds(bottom_k=2, ratio_clip_max=2.0, hard_gate_delta=0.05)
    margin = BoundaryMargin.compute(
        family="placement_only",
        stress_lambda=0.3,
        evolve_step=0,
        pc_f=0.63,
        q_f=1.0,
        u_f=0.01,
        c_rec_p90_f=0.5,
        v_stab_p90_f=0.1,
        thresholds=thresholds,
    )

    assert abs(margin.pc_term - 0.9) < 1e-9
    assert margin.safe_term == 2.0
    assert margin.to_dict()["ge_ratio_pc"] == margin.pc_term
    assert margin.get_summary_for_csv()["ge_margin_soft"] == margin.margin
    assert "pc_term_raw" not in margin.to_dict()
    assert "metric_source_map" not in margin.to_dict()
    assert margin.to_diagnostic_dict()["pc_term_raw"] == margin.pc_term_raw
    assert margin.margin > 0.0
    assert not margin.hard_gate_pass
    assert margin.hard_boundary_hit


def test_capacity_summary_uses_discrete_lambda_grid():
    thresholds = BoundaryThresholds(stress_lambdas=(0.0, 0.5, 1.0))
    collector = BoundaryCollector(thresholds=thresholds)
    rows = [
        {
            "episode_id": "78",
            "anchor_id": "78",
            "task_family": "placement_only",
            "stress_lambda": 0.0,
            "evolve_step": 0,
            "task_percent_complete": 0.9,
            "proposition_satisfied_fraction": 0.9,
            "rebound_c_rec": 0.5,
            "rebound_c_rec_valid": 1,
            "stability_beta_at_lambda": 0.1,
            "stability_beta_at_lambda_valid": 1,
            "unsafe_rate": 0.01,
        },
        {
            "episode_id": "78",
            "anchor_id": "78",
            "task_family": "placement_only",
            "stress_lambda": 0.5,
            "evolve_step": 0,
            "task_percent_complete": 0.82,
            "proposition_satisfied_fraction": 0.82,
            "rebound_c_rec": 0.8,
            "rebound_c_rec_valid": 1,
            "stability_beta_at_lambda": 0.12,
            "stability_beta_at_lambda_valid": 1,
            "unsafe_rate": 0.02,
        },
    ]
    margins = collector.aggregate(rows)
    audc_map = collector.finalize_audc(margins)

    assert ("placement_only", 0) not in audc_map
    cell = margins[("placement_only", 0.5, 0)]
    assert cell.boundary_lambda_star is None
    assert cell.get_summary_for_csv()["ge_boundary_lambda_star"] == ""
    assert abs(cell.valid_lambda_coverage - (2.0 / 3.0)) < 1e-9
    assert cell.missing_lambdas == "1"
    assert cell.capacity_missing_reason == "incomplete_configured_stress_grid"


def test_family_ratio_vector_uses_lower_and_upper_quantiles():
    thresholds = BoundaryThresholds(quantile_alpha=0.25, stress_lambdas=(0.0,))
    collector = BoundaryCollector(thresholds=thresholds)
    rows = []
    for idx, (pc, c_rec) in enumerate(
        [(0.5, 0.2), (0.7, 0.4), (0.9, 0.6), (1.0, 1.0)]
    ):
        rows.append(
            {
                "episode_id": str(idx),
                "anchor_id": str(idx),
                "task_family": "placement_only",
                "stress_lambda": 0.0,
                "evolve_step": 0,
                "task_percent_complete": pc,
                "proposition_satisfied_fraction": pc,
                "rebound_c_rec": c_rec,
                "rebound_c_rec_valid": 1,
                "stability_beta_at_lambda": c_rec / 10.0,
                "stability_beta_at_lambda_valid": 1,
                "unsafe_rate": c_rec / 100.0,
            }
        )

    margins = collector.aggregate(rows)
    margin = margins[("placement_only", 0.0, 0)]

    assert abs(margin.pc_f - 0.65) < 1e-9
    assert abs(margin.c_rec_p90_f - 0.7) < 1e-9
    assert margin.to_diagnostic_dict()["c_rec_tail_f"] == margin.c_rec_p90_f
    assert margin.to_dict()["ge_ratio_rec"] == margin.rec_term


def test_clean_distribution_calibrates_judge_scoped_family_boundary():
    thresholds = BoundaryThresholds(
        tau_pc=0.1,
        tau_q=0.1,
        tau_safe=9.0,
        tau_rec=9.0,
        tau_stab=9.0,
        quantile_alpha=0.25,
        min_clean_samples=2,
        benefit_retention_ratio=0.9,
        cost_tolerance_ratio=1.25,
        stress_lambdas=(0.0, 0.5),
    )
    collector = BoundaryCollector(thresholds=thresholds)
    rows = []
    for index, pc in enumerate((0.80, 0.90, 1.00)):
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
                "task_percent_complete": pc,
                "proposition_satisfied_fraction": pc,
                "rebound_c_rec": 0.2 + 0.1 * index,
                "rebound_c_rec_valid": 1,
                "stability_beta_at_lambda": 0.1 + 0.01 * index,
                "stability_beta_at_lambda_valid": 1,
                "unsafe_rate": 0.01 * index,
            }
        )
    rows.append(
        {
            **rows[0],
            "episode_id": "stress",
            "anchor_id": "stress",
            "perturbation_type": "irrelevant_perturbation",
            "perturbation_valid": 1,
            "perturbation_realized": 0.5,
            "stress_lambda": 0.5,
            "task_percent_complete": 0.70,
            "proposition_satisfied_fraction": 0.70,
            "rebound_c_rec": 0.6,
            "stability_beta_at_lambda": 0.2,
            "unsafe_rate": 0.04,
        }
    )

    collector.aggregate(rows)
    baseline = collector.statistical_baselines[("placement_only", 0)]
    resolved = collector._resolved_thresholds("placement_only", 0)
    evolved_resolved = collector._resolved_thresholds("placement_only", 5)

    assert baseline.valid
    assert baseline.n_samples == 3
    assert baseline.judge_model == "GPT-5.1"
    assert abs(
        baseline.references["task_percent_complete"].lower_quantile - 0.85
    ) < 1e-12
    assert abs(resolved.tau_pc - 0.85 * 0.9) < 1e-12
    assert abs(resolved.tau_rec - 0.35 * 1.25) < 1e-12
    assert resolved.tau_pc != thresholds.tau_pc
    assert resolved.tau_rec != thresholds.tau_rec
    assert evolved_resolved.tau_pc == resolved.tau_pc
    assert evolved_resolved.tau_rec == resolved.tau_rec
    assert all(
        row.boundary_reference_valid for row in collector.episode_evidence
    )


def test_judge_scoped_boundary_baseline_round_trip(tmp_path):
    collector = BoundaryCollector(
        thresholds=BoundaryThresholds(min_clean_samples=2)
    )
    rows = []
    for index in range(2):
        rows.append(
            {
                "episode_id": str(index),
                "task_family": "placement_only",
                "perturbation_type": "clean",
                "stress_lambda": 0.0,
                "evolve_step": 0,
                "judge_model": "GPT-4o",
                "task_percent_complete": 0.9,
                "proposition_satisfied_fraction": 0.9,
                "rebound_c_rec": 0.4,
                "rebound_c_rec_valid": 1,
                "stability_beta_at_lambda": 0.1,
                "stability_beta_at_lambda_valid": 1,
                "unsafe_rate": 0.0,
            }
        )
    collector.aggregate(rows)
    path = write_statistical_baselines(
        collector.statistical_baselines,
        tmp_path / "GPT-4o" / "statistics" / "family_boundaries" / "baseline.json",
    )
    restored = read_statistical_baselines(path)

    assert restored[("placement_only", 0)].judge_model == "GPT-4o"
    assert restored[("placement_only", 0)].valid


def test_degradation_rate_and_cliff_diagnostic():
    thresholds = BoundaryThresholds(
        stress_lambdas=(0.0, 0.5, 1.0),
        degradation_rate_threshold=1.0,
    )
    collector = BoundaryCollector(thresholds=thresholds)
    margins = {
        ("placement_only", 0.0, 0): BoundaryMargin(
            family="placement_only",
            stress_lambda=0.0,
            evolve_step=0,
            margin=0.6,
            hard_gate_pass=True,
            cell_valid=True,
        ),
        ("placement_only", 0.5, 0): BoundaryMargin(
            family="placement_only",
            stress_lambda=0.5,
            evolve_step=0,
            margin=0.3,
            hard_gate_pass=True,
            cell_valid=True,
        ),
        ("placement_only", 1.0, 0): BoundaryMargin(
            family="placement_only",
            stress_lambda=1.0,
            evolve_step=0,
            margin=-0.2,
            hard_gate_pass=False,
            cell_valid=True,
        ),
    }

    collector.finalize_audc(margins)

    assert margins[("placement_only", 0.0, 0)].degradation_rate == 0.0
    assert abs(margins[("placement_only", 0.5, 0)].degradation_rate - 0.6) < 1e-9
    assert abs(margins[("placement_only", 1.0, 0)].degradation_rate - 1.0) < 1e-9
    assert margins[("placement_only", 1.0, 0)].cliff_flag
    assert collector.capacity_rows[0]["ge_max_degradation_rate"] == 1.0
    assert collector.capacity_rows[0]["ge_cliff_lambda"] == 1.0


def test_expq1_ge_config_declares_capacity_headline_and_matching_grid():
    try:
        from omegaconf import OmegaConf
    except ModuleNotFoundError:
        return
    if not hasattr(OmegaConf, "load"):
        return

    config_path = (
        Path(__file__).resolve().parents[2]
        / "habitat_llm"
        / "conf"
        / "experiments"
        / "expq1.yaml"
    )
    cfg = OmegaConf.load(config_path)
    ge_cfg = cfg.expq1.subexperiments.ge_validity
    stress_grid = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    assert ge_cfg.headline_metric == "ge_boundary_lambda_star"
    assert list(ge_cfg.postprocess) == ["stability", "margin"]
    assert list(ge_cfg.summary_group_keys) == [
        "condition_name",
        "task_family",
        "stress_family",
        "stress_lambda",
    ]
    assert len(ge_cfg.phases) == 1
    phase = ge_cfg.phases[0]
    assert phase.label == "stress_sweep"
    assert phase.patch.resilience.mode == "margin"
    assert list(phase.patch.resilience.perturbations) == [
        "clean",
        "llm_instruction_context_stress",
    ]
    assert list(phase.patch.resilience.seeds) == [0, 1, 2]
    assert list(phase.patch.resilience.intensities) == stress_grid
    assert list(phase.patch.resilience.boundary_margin.stress_lambdas) == stress_grid


def _formal_ge_row(*, seed, stress_lambda, margin_scale=1.0):
    is_clean = abs(float(stress_lambda)) <= 1.0e-12
    row = {
        "episode_id": "78",
        "anchor_id": "78",
        "task_family": "placement_only",
        "judge_model": "gpt-5.1",
        "stage_baseline_path": "/baseline/gpt-5.1/stage_baseline.json",
        "stage_baseline_episode_id": "78",
        "stage_baseline_episode_match": 1,
        "stage_baseline_formal_valid": 1,
        "state_encoder_id": "shared-state-encoder",
        "reward_shaper_valid": 1,
        "critic_lifecycle_valid": 1,
        "perturbation_seed": seed,
        "perturbation_type": (
            "clean" if is_clean else "llm_instruction_context_stress"
        ),
        "stress_lambda": stress_lambda,
        "evolve_step": 0,
        "perturbation_stress_family": "llm_instruction_context_stress",
        "perturbation_variant": f"variant_{seed}",
        "perturbation_bank_path": "/banks/instruction_context_v1",
        "perturbation_stress_realized": stress_lambda,
        "perturbation_text_dose": stress_lambda,
        "perturbation_alignment_valid": 1,
        "task_percent_complete": 0.9 * margin_scale,
        "proposition_satisfied_fraction": 0.9 * margin_scale,
        "rebound_c_rec": 0.4,
        "rebound_c_rec_valid": 1,
        "stability_beta_at_lambda": 0.1,
        "stability_beta_at_lambda_valid": 1,
        "unsafe_rate": 0.0,
    }
    if not is_clean:
        row["perturbation_valid"] = 1
    return row


def test_formal_ge_uses_one_complete_episode_seed_set_for_all_six_cells():
    collector = BoundaryCollector(
        thresholds=BoundaryThresholds(min_clean_samples=1)
    )
    rows = [
        _formal_ge_row(seed=0, stress_lambda=stress_lambda)
        for stress_lambda in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    ]
    # Seed 1 is incomplete and must be excluded from every lambda, not just 1.0.
    rows.extend(
        _formal_ge_row(seed=1, stress_lambda=stress_lambda)
        for stress_lambda in (0.0, 0.2, 0.4, 0.6, 0.8)
    )

    margins = collector.aggregate(rows)
    audc = collector.finalize_audc(margins)

    assert len(margins) == 6
    assert all(len(key) == 5 for key in margins)
    assert all(key[2] == "llm_instruction_context_stress" for key in margins)
    assert all(margin.cell_valid for margin in margins.values())
    assert all(margin.n_episodes == 1 for margin in margins.values())
    assert len(audc) == 1
    capacity = collector.capacity_rows[0]
    assert capacity["judge_model"] != "unconfigured-judge"
    assert capacity["stress_family"] == "llm_instruction_context_stress"
    assert capacity["ge_total_episode_seed_pairs"] == 2
    assert capacity["ge_complete_episode_seed_pairs"] == 1
    assert capacity["ge_episode_seed_pair_coverage"] == 0.5


def test_formal_ge_rejects_insufficient_clean_reference_samples():
    collector = BoundaryCollector(
        thresholds=BoundaryThresholds(min_clean_samples=2)
    )
    rows = [
        _formal_ge_row(seed=0, stress_lambda=stress_lambda)
        for stress_lambda in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    ]

    margins = collector.aggregate(rows)
    collector.finalize_audc(margins)

    assert all(not margin.cell_valid for margin in margins.values())
    assert {
        margin.cell_missing_reason for margin in margins.values()
    } == {"insufficient_clean_reference_samples"}
    assert collector.capacity_rows[0]["ge_capacity_valid"] == 0


def test_formal_ge_lambda_star_is_continuous_prefix_and_reports_reentry():
    collector = BoundaryCollector(thresholds=BoundaryThresholds())
    values = {
        0.0: 0.5,
        0.2: 0.3,
        0.4: 0.1,
        0.6: -0.2,
        0.8: 0.1,
        1.0: 0.2,
    }
    margins = {}
    for stress_lambda, value in values.items():
        margin = BoundaryMargin(
            family="placement_only",
            stress_lambda=stress_lambda,
            evolve_step=0,
            margin=value,
            hard_gate_pass=value >= 0.0,
            cell_valid=True,
        )
        collector._attach_margin_scope(
            margin,
            judge_model="gpt-5.1",
            stress_family="llm_instruction_context_stress",
            formal=True,
            pairing_stats={"total_pairs": 1, "complete_pairs": 1},
        )
        margins[
            (
                "gpt-5.1",
                "placement_only",
                "llm_instruction_context_stress",
                stress_lambda,
                0,
            )
        ] = margin

    collector.finalize_audc(margins)

    capacity = collector.capacity_rows[0]
    assert capacity["ge_boundary_lambda_star"] == 0.4
    assert capacity["ge_hard_lambda_star"] == 0.4
    assert capacity["ge_monotonicity_violation"] == 1
