#!/usr/bin/env python3
# isort: skip_file
"""Multiprocess launcher for the Exp-Q2 outer stress suite.

The normal ``planner_demo_expq2_outer`` entry point calls ``run_named_suite``
directly. That path eventually invokes ``planner_demo_mp_new.run_planner``
from inside the resilience rollout helper, so the usual top-level
``num_proc`` split in ``planner_demo_mp_new.run_eval`` is bypassed.

This launcher parallelizes at the Exp-Q2 suite layer by spawning subprocess
workers. In the default ``episode_shards`` mode, workers receive disjoint
episode-id shards and write into isolated output roots; the parent then merges
their raw rollouts and recomputes suite-level summaries. ``replicas`` mode runs
the same full suite in every worker, useful for repeated stress tests.
"""
from __future__ import annotations

import csv
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from habitat_llm.examples.expq_helpers import (
    DEFAULT_RESILIENCE_CFG,
    _as_list,
    _as_mapping,
    _build_condition_list,
    _compose_runner_patch,
    _finalize_subexperiment,
    _ge_analysis_outputs,
    _git_revision,
    merge_config_patch,
    run_named_suite,
    run_margin_aggregation,
    run_stability_aggregation,
    write_csv_rows,
    write_json,
)
from habitat_llm.examples.resilience_helpers import (
    extract_resilience_cfg,
    normalize_rollout_rows,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SUITE_KEY = "expq2_outer"
MP_KEY = "expq2_outer_mp"
LOGGER = logging.getLogger(__name__)


def _mp_cfg(config: DictConfig) -> Dict[str, Any]:
    raw = OmegaConf.select(config, MP_KEY)
    if raw is None:
        return {}
    resolved = OmegaConf.to_container(raw, resolve=True)
    return dict(resolved) if isinstance(resolved, Mapping) else {}


def _suite_cfg(config: DictConfig) -> Dict[str, Any]:
    cfg = _as_mapping(OmegaConf.select(config, SUITE_KEY))
    if not cfg:
        raise ValueError(f"Missing suite config `{SUITE_KEY}`.")
    return cfg


def _suite_output_root(config: DictConfig, suite_cfg: Mapping[str, Any]) -> Path:
    return Path(str(config.paths.results_dir)) / str(
        suite_cfg.get("output_subdir", SUITE_KEY)
    )


def _read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _chunk_list(values: Sequence[Any], num_chunks: int) -> List[List[Any]]:
    if num_chunks <= 0:
        return []
    chunks: List[List[Any]] = []
    base = len(values) // num_chunks
    extra = len(values) % num_chunks
    start = 0
    for idx in range(num_chunks):
        size = base + (1 if idx < extra else 0)
        end = start + size
        if start < end:
            chunks.append(list(values[start:end]))
        start = end
    return chunks


def _configured_episode_ids(config: DictConfig, suite_cfg: Mapping[str, Any]) -> List[Any]:
    mp_ids = _mp_cfg(config).get("episode_ids")
    if mp_ids:
        return list(mp_ids)

    root_ids = config.get("episode_ids", None)
    if root_ids:
        return list(root_ids)

    for block_cfg_raw in _as_mapping(suite_cfg.get("subexperiments")).values():
        block_cfg = dict(block_cfg_raw)
        if not bool(block_cfg.get("enabled", True)):
            continue
        block_ids = block_cfg.get("episode_ids")
        if block_ids:
            return list(block_ids)

    working_set = _as_mapping(suite_cfg.get("shared", {})).get("working_set")
    return list(working_set or [])


def _apply_worker_patch(config: DictConfig, mp_cfg: Mapping[str, Any]) -> None:
    worker_results_dir = str(mp_cfg.get("worker_results_dir") or "").strip()
    worker_episode_ids = list(mp_cfg.get("worker_episode_ids") or [])
    mode = str(mp_cfg.get("mode") or "episode_shards")

    OmegaConf.set_struct(config, False)
    config.num_proc = 1
    if worker_results_dir:
        config.paths.results_dir = worker_results_dir

    if mode == "episode_shards" and worker_episode_ids:
        subexperiments = _as_mapping(OmegaConf.select(config, f"{SUITE_KEY}.subexperiments"))
        for block_name, block_cfg_raw in subexperiments.items():
            if not bool(dict(block_cfg_raw).get("enabled", True)):
                continue
            OmegaConf.update(
                config,
                f"{SUITE_KEY}.subexperiments.{block_name}.episode_ids",
                worker_episode_ids,
                merge=False,
            )

    # Keep worker baselines local by default. Concurrent writes to the shared
    # persistent baseline directory can corrupt quick multiprocess test runs.
    if worker_results_dir and not bool(mp_cfg.get("share_persistent_baseline", False)):
        OmegaConf.update(
            config,
            "resilience.stage_baseline.persistent_store_dir",
            str((Path(worker_results_dir) / "_stage_baselines").as_posix()),
            merge=False,
        )
    OmegaConf.set_struct(config, True)


def _is_internal_mp_arg(arg: str) -> bool:
    normalized = arg.lstrip("+")
    return (
        normalized.startswith(f"{MP_KEY}.")
        or normalized.startswith(f"{MP_KEY}=")
        or normalized.startswith("num_proc=")
        or normalized.startswith("+num_proc=")
    )


def _worker_command(
    *,
    worker_index: int,
    worker_results_dir: Path,
    mode: str,
    episode_ids: Optional[Sequence[Any]],
    share_persistent_baseline: bool,
) -> List[str]:
    base_args = [arg for arg in sys.argv[1:] if not _is_internal_mp_arg(arg)]
    cmd = [
        sys.executable,
        "-m",
        "habitat_llm.examples.planner_demo_expq2_outer_mp",
        *base_args,
        "num_proc=1",
        f"paths.results_dir={worker_results_dir.as_posix()}",
        f"++{MP_KEY}.worker=true",
        f"++{MP_KEY}.worker_index={worker_index}",
        f"++{MP_KEY}.mode={mode}",
        f"++{MP_KEY}.worker_results_dir={worker_results_dir.as_posix()}",
        f"++{MP_KEY}.share_persistent_baseline={str(share_persistent_baseline).lower()}",
    ]
    if episode_ids is not None:
        literal = ",".join(str(value) for value in episode_ids)
        cmd.append(f"++{MP_KEY}.worker_episode_ids=[{literal}]")
    return cmd


def _launch_workers(config: DictConfig) -> List[Dict[str, Any]]:
    suite_cfg = _suite_cfg(config)
    mp_cfg = _mp_cfg(config)
    mode = str(mp_cfg.get("mode") or "episode_shards")
    if mode not in {"episode_shards", "replicas"}:
        raise ValueError(f"Unsupported {MP_KEY}.mode={mode!r}")

    requested = int(config.get("num_proc", 1) or 1)
    if requested <= 1:
        return []

    parent_output_root = _suite_output_root(config, suite_cfg)
    worker_root = parent_output_root / "_mp_workers"
    worker_root.mkdir(parents=True, exist_ok=True)

    if mode == "episode_shards":
        episode_ids = _configured_episode_ids(config, suite_cfg)
        if not episode_ids:
            raise ValueError(
                "episode_shards mode requires episode ids in "
                f"{SUITE_KEY}.subexperiments.*.episode_ids, {SUITE_KEY}.shared.working_set, "
                f"or ++{MP_KEY}.episode_ids=[...]"
            )
        shards = _chunk_list(episode_ids, min(requested, len(episode_ids)))
    else:
        replicas = int(mp_cfg.get("replicas") or requested)
        shards = [None for _ in range(replicas)]  # type: ignore[list-item]

    share_persistent = bool(mp_cfg.get("share_persistent_baseline", False))
    workers: List[Dict[str, Any]] = []
    processes: List[tuple[subprocess.Popen[Any], Dict[str, Any]]] = []

    for worker_index, shard in enumerate(shards):
        worker_dir = worker_root / f"worker_{worker_index:02d}"
        cmd = _worker_command(
            worker_index=worker_index,
            worker_results_dir=worker_dir,
            mode=mode,
            episode_ids=shard,
            share_persistent_baseline=share_persistent,
        )
        worker_record = {
            "worker_index": worker_index,
            "mode": mode,
            "episode_ids": list(shard) if shard is not None else None,
            "results_dir": str(worker_dir),
            "suite_output_root": str(worker_dir / str(suite_cfg.get("output_subdir", SUITE_KEY))),
            "command": cmd,
        }
        workers.append(worker_record)
        LOGGER.info(
            "Starting worker %d/%d mode=%s episodes=%s -> %s",
            worker_index + 1,
            len(shards),
            mode,
            worker_record["episode_ids"],
            worker_dir,
        )
        processes.append((subprocess.Popen(cmd), worker_record))

    failures: List[Dict[str, Any]] = []
    for process, worker_record in processes:
        return_code = process.wait()
        worker_record["return_code"] = return_code
        if return_code != 0:
            failures.append(worker_record)

    write_json(
        parent_output_root / "mp_worker_manifest.json",
        {
            "suite_name": SUITE_KEY,
            "mode": mode,
            "num_proc": requested,
            "workers": workers,
            "git_revision": _git_revision(),
        },
    )
    if failures:
        failed = ", ".join(
            f"{item['worker_index']} (rc={item['return_code']})" for item in failures
        )
        raise RuntimeError(f"Exp-Q2 outer worker failure(s): {failed}")
    return workers


def _collect_worker_subexperiment_rows(
    workers: Sequence[Mapping[str, Any]],
    block_name: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for worker in workers:
        worker_root = Path(str(worker["suite_output_root"]))
        path = worker_root / str(block_name) / "raw_rollouts.csv"
        for row in _read_csv_rows(path):
            row["mp_worker_index"] = worker.get("worker_index", "")
            row["mp_worker_mode"] = worker.get("mode", "")
            rows.append(row)
    return rows


def _postprocess_condition_rows(
    *,
    config: DictConfig,
    suite_cfg: Mapping[str, Any],
    block_name: str,
    block_cfg: Mapping[str, Any],
    condition_cfg: Mapping[str, Any],
    subexperiment_dir: Path,
    rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    condition_name = str(condition_cfg.get("name") or "default")
    condition_label = str(condition_cfg.get("label") or condition_name)
    condition_dir = subexperiment_dir / "conditions" / condition_name
    analysis_dir = condition_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    condition_post_cfg = merge_config_patch(
        config,
        _compose_runner_patch(
            suite_cfg,
            block_cfg,
            condition_cfg,
            {"patch": {}},
            condition_dir,
        ),
    )
    resilience_cfg = extract_resilience_cfg(condition_post_cfg, DEFAULT_RESILIENCE_CFG)
    dataset_path = OmegaConf.select(config, "habitat.dataset.data_path")
    normalized_rows = normalize_rollout_rows(
        rows,
        clean_tag=str(
            (resilience_cfg.get("stage_baseline", {}) or {}).get("clean_tag", "clean")
        ),
        dataset_path=str(dataset_path) if dataset_path else None,
    )
    postprocess = [str(mode) for mode in _as_list(block_cfg.get("postprocess"))]
    if "stability" in postprocess:
        run_stability_aggregation(normalized_rows, analysis_dir, resilience_cfg)
    if "margin" in postprocess:
        run_margin_aggregation(
            normalized_rows,
            analysis_dir,
            resilience_cfg,
            dataset_path=str(dataset_path) if dataset_path else None,
        )
    raw_path = write_csv_rows(analysis_dir / "raw_rollouts.csv", normalized_rows)
    return {
        "condition_name": condition_name,
        "condition_label": condition_label,
        "rows": normalized_rows,
        "phase_runs": [],
        "baseline_path": None,
        "analysis_dir": str(analysis_dir),
        "condition_raw_rollouts_csv": str(raw_path),
        "postprocess": postprocess,
        "ge_outputs": _ge_analysis_outputs(analysis_dir),
    }


def _merge_worker_outputs(config: DictConfig, workers: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    suite_cfg = _suite_cfg(config)
    output_root = _suite_output_root(config, suite_cfg)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "suite_name": SUITE_KEY,
        "output_root": str(output_root),
        "subexperiments": [],
        "mp_workers": [
            {key: value for key, value in worker.items() if key != "command"}
            for worker in workers
        ],
        "git_revision": _git_revision(),
        "config_name": HydraConfig.get().job.config_name if HydraConfig.initialized() else None,
    }

    for block_name, block_cfg_raw in _as_mapping(suite_cfg.get("subexperiments")).items():
        block_cfg = dict(block_cfg_raw)
        if not bool(block_cfg.get("enabled", True)):
            continue

        subexperiment_dir = output_root / str(block_name)
        subexperiment_dir.mkdir(parents=True, exist_ok=True)
        worker_rows = _collect_worker_subexperiment_rows(workers, str(block_name))

        combined_rows: List[Dict[str, Any]] = []
        condition_runs: List[Dict[str, Any]] = []
        for condition_cfg in _build_condition_list(block_cfg):
            condition_name = str(condition_cfg.get("name") or "default")
            condition_rows = [
                dict(row)
                for row in worker_rows
                if str(row.get("condition_name") or "default") == condition_name
            ]
            condition_result = _postprocess_condition_rows(
                config=config,
                suite_cfg=suite_cfg,
                block_name=str(block_name),
                block_cfg=block_cfg,
                condition_cfg=condition_cfg,
                subexperiment_dir=subexperiment_dir,
                rows=condition_rows,
            )
            combined_rows.extend(condition_result["rows"])
            condition_runs.append(condition_result)

        manifest["subexperiments"].append(
            _finalize_subexperiment(
                config,
                SUITE_KEY,
                suite_cfg,
                str(block_name),
                block_cfg,
                subexperiment_dir,
                combined_rows,
                condition_runs,
            )
        )

    write_json(output_root / "suite_manifest.json", manifest)
    return manifest


@hydra.main(
    config_path="../conf",
    config_name="experiments/expq2_outer",
    version_base=None,
)
def main(config: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    started = time.time()
    mp_cfg = _mp_cfg(config)

    if bool(mp_cfg.get("worker", False)):
        _apply_worker_patch(config, mp_cfg)
        LOGGER.info(
            "Running Exp-Q2 outer worker index=%s mode=%s results_dir=%s episodes=%s",
            mp_cfg.get("worker_index"),
            mp_cfg.get("mode", "episode_shards"),
            config.paths.results_dir,
            mp_cfg.get("worker_episode_ids"),
        )
        run_named_suite(config, SUITE_KEY)
        return

    if int(config.get("num_proc", 1) or 1) <= 1:
        run_named_suite(config, SUITE_KEY)
        return

    workers = _launch_workers(config)
    manifest = _merge_worker_outputs(config, workers)
    LOGGER.info(
        "Merged Exp-Q2 outer multiprocess outputs into %s (%d subexperiment(s)) in %.1fs",
        manifest["output_root"],
        len(manifest.get("subexperiments", [])),
        time.time() - started,
    )


if __name__ == "__main__":
    main()
