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
    contract_module = _load_module(
        "habitat_llm.evaluation.metrics.extensibility.contract_margin",
        ext_root / "contract_margin.py",
    )
    margin_module = _load_module(
        "habitat_llm.evaluation.metrics.extensibility.margin_collector",
        ext_root / "margin_collector.py",
    )
    ext_pkg = sys.modules["habitat_llm.evaluation.metrics.extensibility"]
    ext_pkg.ContractMargin = contract_module.ContractMargin
    ext_pkg.MarginThresholds = contract_module.MarginThresholds
    ext_pkg.MarginCollector = margin_module.MarginCollector
    return (
        contract_module.ContractMargin,
        margin_module.MarginCollector,
        contract_module.MarginThresholds,
    )


ContractMargin, MarginCollector, MarginThresholds = _bootstrap_ext_modules()


def test_margin_collector_parses_valid_row_with_combined_safety_load():
    collector = MarginCollector(thresholds=MarginThresholds())
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
            "stability_beta_neighborhood": 0.12,
            "stability_beta_neighborhood_valid": 1,
            "safety_collision_rate": 0.02,
            "stability_p_cbf": 4.0,
            "total_step_count": 100,
        }
    ]
    margins = collector.aggregate(rows)

    assert len(margins) == 1
    evidence = collector.episode_evidence[0]
    assert evidence.valid
    assert evidence.safety_load == 0.06
    assert evidence.metric_sources["unsafe_rate"] == "safety_collision_rate+stability_p_cbf/total_step_count"
    evidence_dict = evidence.to_dict()
    assert evidence_dict["ge_episode_margin"] == evidence.episode_margin
    assert evidence_dict["ge_episode_hard_gate_pass"] == int(evidence.hard_gate_pass)
    assert "metric_source_map" not in evidence_dict
    assert "pc_ratio_raw" not in evidence_dict
    evidence_diag = evidence.to_diagnostic_dict()
    assert evidence_diag["metric_source_map"]["unsafe_rate"] == "safety_collision_rate+stability_p_cbf/total_step_count"
    margin = next(iter(margins.values()))
    assert margin.cell_valid
    assert margin.n_episodes == 1


def test_margin_collector_uses_proxy_stability_load_when_neighborhood_missing():
    collector = MarginCollector(thresholds=MarginThresholds())
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
    assert evidence.valid
    assert evidence.stability_load == 0.18
    assert evidence.metric_sources["stability_load"] == "stability_value_delta_variance:proxy"


def test_invalid_rebound_row_creates_invalid_cell():
    collector = MarginCollector(thresholds=MarginThresholds(stress_lambdas=(0.0, 0.5, 1.0)))
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
            "stability_beta_neighborhood": 0.1,
            "stability_beta_neighborhood_valid": 1,
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


def test_contract_margin_uses_bottom_k_soft_margin_and_hard_gate():
    thresholds = MarginThresholds(bottom_k=2, ratio_clip_max=2.0, hard_gate_delta=0.05)
    margin = ContractMargin.compute(
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
    thresholds = MarginThresholds(stress_lambdas=(0.0, 0.5, 1.0))
    collector = MarginCollector(thresholds=thresholds)
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
            "stability_beta_neighborhood": 0.1,
            "stability_beta_neighborhood_valid": 1,
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
            "stability_beta_neighborhood": 0.12,
            "stability_beta_neighborhood_valid": 1,
            "unsafe_rate": 0.02,
        },
    ]
    margins = collector.aggregate(rows)
    audc_map = collector.finalize_audc(margins)

    assert ("placement_only", 0) in audc_map
    cell = margins[("placement_only", 0.5, 0)]
    assert cell.boundary_lambda_star == 0.5
    assert cell.cpsc_lambda_star == 0.5
    assert cell.get_summary_for_csv()["ge_cpsc_lambda_star"] == 0.5
    assert abs(cell.valid_lambda_coverage - (2.0 / 3.0)) < 1e-9
    assert cell.missing_lambdas == "1"


def test_family_ratio_vector_uses_lower_and_upper_quantiles():
    thresholds = MarginThresholds(quantile_alpha=0.25, stress_lambdas=(0.0,))
    collector = MarginCollector(thresholds=thresholds)
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
                "stability_beta_neighborhood": c_rec / 10.0,
                "stability_beta_neighborhood_valid": 1,
                "unsafe_rate": c_rec / 100.0,
            }
        )

    margins = collector.aggregate(rows)
    margin = margins[("placement_only", 0.0, 0)]

    assert abs(margin.pc_f - 0.65) < 1e-9
    assert abs(margin.c_rec_p90_f - 0.7) < 1e-9
    assert margin.to_diagnostic_dict()["c_rec_tail_f"] == margin.c_rec_p90_f
    assert margin.to_dict()["ge_ratio_rec"] == margin.rec_term


def test_degradation_rate_and_cliff_diagnostic():
    thresholds = MarginThresholds(
        stress_lambdas=(0.0, 0.5, 1.0),
        degradation_rate_threshold=1.0,
    )
    collector = MarginCollector(thresholds=thresholds)
    margins = {
        ("placement_only", 0.0, 0): ContractMargin(
            family="placement_only",
            stress_lambda=0.0,
            evolve_step=0,
            margin=0.6,
            hard_gate_pass=True,
            cell_valid=True,
        ),
        ("placement_only", 0.5, 0): ContractMargin(
            family="placement_only",
            stress_lambda=0.5,
            evolve_step=0,
            margin=0.3,
            hard_gate_pass=True,
            cell_valid=True,
        ),
        ("placement_only", 1.0, 0): ContractMargin(
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

    assert ge_cfg.headline_metric == "ge_cpsc_lambda_star"
    assert list(ge_cfg.postprocess) == ["stability", "margin"]
    assert list(ge_cfg.summary_group_keys) == [
        "condition_name",
        "task_family",
        "stress_lambda",
    ]
    assert list(ge_cfg.phases[0].patch.resilience.contract_margin.stress_lambdas) == stress_grid
    assert list(ge_cfg.phases[1].patch.resilience.intensities) == stress_grid[1:]
    assert list(ge_cfg.phases[1].patch.resilience.contract_margin.stress_lambdas) == stress_grid
