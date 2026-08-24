from __future__ import annotations

import asyncio
import contextlib
import csv
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

from app.mcp.tools.habitat_adapter import materialize_to_sandbox
from app.instruction.instruction_case_codegen_service import instruction_case_codegen_service
from app.instruction.session_store import instruction_session_store


TERMINAL_STATES = {"completed", "failed", "cancelled"}
STAGE_SEQUENCE = [
    "preflight",
    "materialize",
    "habitat_run",
    "collect_metrics",
    "video_postprocess",
    "finalize",
]
MAX_LIVE_LOG_LINES = 120


def _now() -> str:
    return datetime.now().isoformat()


def _project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if parent.name == "GroupIntelligenceFront":
            return parent.parent
    return here.parents[3]


def _habitat_root() -> Path:
    env_root = (
        os_environ("EAB_HABITAT_ROOT")
        or os_environ("EVO_AGENT_BUILDER_ROOT")
        or os_environ("HABITAT_LLM_CONF_ROOT")
    )
    if env_root:
        candidate = Path(env_root)
        if (candidate / "habitat_llm" / "conf").exists():
            return candidate / "habitat_llm"
        if (candidate / "conf").exists():
            return candidate
    return _project_root() / "habitat_llm"


def os_environ(name: str) -> Optional[str]:
    import os

    value = os.environ.get(name)
    return str(value).strip() if value else None


def _habitat_conf_root() -> Path:
    return _habitat_root() / "conf"


def _resolve_config_path(raw_path: str) -> Path:
    candidate = Path(str(raw_path).strip())
    project_root = _project_root()
    habitat_root = _habitat_root()
    conf_root = _habitat_conf_root()

    options: List[Path] = []
    if candidate.is_absolute():
        options.append(candidate)
    else:
        options.extend(
            [
                project_root / candidate,
                habitat_root / candidate,
                conf_root / candidate,
            ]
        )

    seen = set()
    for path in options:
        normalized = path.resolve()
        if normalized in seen:
            continue
        seen.add(normalized)
        if normalized.exists():
            return normalized
    return conf_root / candidate


def _to_hydra_config_name(baseline_path: Path, fallback_raw: str) -> str:
    conf_root = _habitat_conf_root().resolve()
    try:
        relative = baseline_path.resolve().relative_to(conf_root)
        return relative.as_posix()
    except Exception:
        return str(fallback_raw).strip().replace("\\", "/")


def _dig(payload: Dict[str, Any], path: List[str]) -> Optional[Any]:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _load_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None or not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _resolve_dataset_path(dataset_raw: str) -> Tuple[Optional[Path], List[Path]]:
    value = str(dataset_raw or "").strip().strip('"').strip("'")
    if not value:
        return None, []
    candidate = Path(value)
    if candidate.is_absolute():
        return (candidate if candidate.exists() else None), [candidate]

    project_root = _project_root()
    habitat_root = _habitat_root()
    options = [
        project_root / candidate,
        habitat_root / candidate,
    ]
    for path in options:
        if path.exists():
            return path, options
    return None, options


def _discover_dataset_fallback() -> Optional[Path]:
    env_path = os_environ("EAB_DATASET_PATH")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p
    root = _project_root()
    candidates = [
        root / "data" / "datasets" / "partnr_episodes" / "v0_0" / "val_mini.json.gz",
        root / "habitat_llm" / "data" / "datasets" / "partnr_episodes" / "v0_0" / "val_mini.json.gz",
        root / "habitat_llm" / "output" / "results" / "val_mini.json.gz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _float_or_default(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _read_last_csv_row(csv_path: Path) -> Dict[str, Any]:
    if not csv_path.exists():
        return {}
    rows: List[Dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    return rows[-1] if rows else {}


def _read_json_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _discover_latest_mp4(results_root: Path) -> Optional[Path]:
    if not results_root.exists():
        return None
    videos = [path for path in results_root.rglob("*.mp4") if path.is_file()]
    if not videos:
        return None
    videos.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return videos[0]


def _pick_python_bin() -> str:
    configured = os_environ("EAB_HABITAT_PYTHON")
    if configured:
        return configured
    return sys.executable


def _command_exists(command_name: str) -> bool:
    return shutil.which(command_name) is not None


def _build_hydra_dataset_path(dataset_path: Path) -> str:
    project_root = _project_root().resolve()
    try:
        relative = dataset_path.resolve().relative_to(project_root)
        return relative.as_posix()
    except Exception:
        return dataset_path.resolve().as_posix()


def _trim_lines(text: str, limit: int = MAX_LIVE_LOG_LINES) -> List[str]:
    rows = [line for line in (text or "").splitlines() if line is not None]
    if len(rows) <= limit:
        return rows
    return rows[-limit:]


def _deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _is_path_like(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    return any(token in text for token in ["/", "\\", ":"])


def _derive_python_from_conda(conda_env: str) -> Optional[str]:
    env_text = str(conda_env or "").strip()
    if not env_text:
        return None
    if _is_path_like(env_text):
        root = Path(env_text)
        candidates = [root / "bin" / "python", root / "python.exe"]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
    return None


def _build_runtime_context(request: Dict[str, Any]) -> Dict[str, Any]:
    runtime_raw = request.get("runtime_env") if isinstance(request.get("runtime_env"), dict) else {}
    conda_env = str(runtime_raw.get("conda_env") or os_environ("EAB_RUNTIME_CONDA_ENV") or "").strip()
    cwd = str(runtime_raw.get("workdir") or os_environ("EAB_RUNTIME_WORKDIR") or str(_project_root())).strip()
    cwd = str(Path(cwd).expanduser())
    python_bin = str(runtime_raw.get("python_bin") or "").strip()
    errors: List[str] = []
    if not python_bin:
        derived_python = _derive_python_from_conda(conda_env)
        if derived_python:
            python_bin = derived_python
        elif conda_env and _is_path_like(conda_env):
            errors.append(
                "conda_env is path-like but python was not found under it. "
                "Provide runtime_env.python_bin or fix runtime_env.conda_env."
            )
        else:
            python_bin = _pick_python_bin()

    env_patch: Dict[str, str] = {}
    cuda_bin = str(runtime_raw.get("cuda_bin") or os_environ("EAB_CUDA_BIN") or "").strip()
    if cuda_bin:
        env_patch["PATH"] = f"{cuda_bin}{os.pathsep}{os.environ.get('PATH', '')}"

    if runtime_raw.get("cuda_devices"):
        env_patch["CUDA_VISIBLE_DEVICES"] = str(runtime_raw.get("cuda_devices")).strip()
    if runtime_raw.get("hf_endpoint") or os_environ("HF_ENDPOINT"):
        env_patch["HF_ENDPOINT"] = str(runtime_raw.get("hf_endpoint") or os_environ("HF_ENDPOINT")).strip()
    if runtime_raw.get("hf_home") or os_environ("HF_HOME"):
        env_patch["HF_HOME"] = str(runtime_raw.get("hf_home") or os_environ("HF_HOME")).strip()
    if runtime_raw.get("transformers_cache") or os_environ("TRANSFORMERS_CACHE"):
        env_patch["TRANSFORMERS_CACHE"] = str(
            runtime_raw.get("transformers_cache") or os_environ("TRANSFORMERS_CACHE")
        ).strip()
    if runtime_raw.get("openai_api_key") or os_environ("OPENAI_API_KEY"):
        env_patch["OPENAI_API_KEY"] = str(runtime_raw.get("openai_api_key") or os_environ("OPENAI_API_KEY")).strip()

    extra_exports = runtime_raw.get("env_exports")
    if isinstance(extra_exports, dict):
        for key, value in extra_exports.items():
            if key and value is not None:
                env_patch[str(key)] = str(value)

    summary = {
        "conda_env": conda_env or None,
        "cwd": cwd,
        "python_bin": python_bin,
        "cuda_bin": cuda_bin or None,
        "cuda_devices": env_patch.get("CUDA_VISIBLE_DEVICES"),
        "hf_endpoint": env_patch.get("HF_ENDPOINT"),
        "hf_home": env_patch.get("HF_HOME"),
        "transformers_cache": env_patch.get("TRANSFORMERS_CACHE"),
        "openai_api_key_set": bool(env_patch.get("OPENAI_API_KEY")),
    }
    return {
        "cwd": cwd,
        "python_bin": python_bin,
        "env": env_patch,
        "summary": summary,
        "errors": errors,
    }


class SandboxValidationManager:
    def __init__(self):
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._tasks: Dict[str, asyncio.Task] = {}
        self._active_processes: Dict[str, asyncio.subprocess.Process] = {}

    def create_job(
        self,
        *,
        spec_id: str,
        spec: Dict[str, Any],
        request: Dict[str, Any],
    ) -> Dict[str, Any]:
        job_id = str(uuid4())
        now = _now()
        job = {
            "job_id": job_id,
            "spec_id": spec_id,
            "validation_mode": request.get("validation_mode") or "instruction_case",
            "status": "pending",
            "stage": "queued",
            "created_at": now,
            "updated_at": now,
            "stage_started_at": now,
            "stage_elapsed_sec": 0.0,
            "heartbeat_at": now,
            "request": deepcopy(request),
            "checks": [],
            "command": None,
            "process": {
                "pid": None,
                "kind": None,
                "status": "idle",
                "started_at": None,
                "ended_at": None,
                "command": None,
                "cwd": None,
            },
            "live_logs": {
                "stdout_tail": [],
                "stderr_tail": [],
                "last_output_at": None,
            },
            "stage_detail": {
                "step": "queued",
                "message": "Queued",
            },
            "metrics": {},
            "grade_report": None,
            "instruction_case_summary": None,
            "outputs": {},
            "videos": {
                "raw": {"available": False, "path": None, "media_url": None},
                "processed": {"available": False, "path": None, "media_url": None},
                "postprocess_skipped": False,
            },
            "manifest": None,
            "error": None,
            "summary": None,
        }
        self._jobs[job_id] = job
        self._tasks[job_id] = asyncio.create_task(self._run_job(job_id, spec, request))
        return deepcopy(job)

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        job = self._jobs.get(job_id)
        if job is None:
            return None
        if job.get("status") not in TERMINAL_STATES:
            stage_started_at = job.get("stage_started_at")
            if stage_started_at:
                try:
                    elapsed = (datetime.now() - datetime.fromisoformat(stage_started_at)).total_seconds()
                    job["stage_elapsed_sec"] = round(max(elapsed, 0.0), 3)
                except Exception:
                    pass
        job["can_cancel"] = job.get("status") not in TERMINAL_STATES
        return deepcopy(job)

    def get_media_path(self, job_id: str, kind: str) -> Optional[str]:
        job = self._jobs.get(job_id)
        if job is None:
            return None
        if kind not in {"raw", "processed"}:
            return None
        media = (job.get("videos") or {}).get(kind) or {}
        path = media.get("path")
        if not path:
            return None
        if not Path(path).exists():
            return None
        return str(path)

    def cancel_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        job = self._jobs.get(job_id)
        if job is None:
            return None
        if job.get("status") in TERMINAL_STATES:
            return deepcopy(job)
        job["status"] = "cancelled"
        job["heartbeat_at"] = _now()
        job["updated_at"] = _now()
        self._set_stage_detail(
            job_id,
            step="cancel_requested",
            message="Cancellation requested by user.",
        )
        proc = self._active_processes.get(job_id)
        if proc and proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            process_meta = job.get("process") or {}
            process_meta["status"] = "killed"
            process_meta["ended_at"] = _now()
            job["process"] = process_meta
        task = self._tasks.get(job_id)
        if task and not task.done():
            task.cancel()
        self._append_check(
            job_id,
            stage=job.get("stage") or "queued",
            name="job_cancelled",
            ok=False,
            detail="Cancelled by user request.",
        )
        return deepcopy(job)

    def _set_stage_detail(self, job_id: str, *, step: str, message: str) -> None:
        job = self._jobs[job_id]
        job["stage_detail"] = {"step": step, "message": message}
        job["heartbeat_at"] = _now()
        job["updated_at"] = _now()

    def _append_live_log(self, job_id: str, stream: str, line: str) -> None:
        if stream not in {"stdout_tail", "stderr_tail"}:
            return
        job = self._jobs[job_id]
        logs = job.get("live_logs") or {}
        rows = list(logs.get(stream) or [])
        rows.append(line.rstrip("\n"))
        if len(rows) > MAX_LIVE_LOG_LINES:
            rows = rows[-MAX_LIVE_LOG_LINES:]
        logs[stream] = rows
        logs["last_output_at"] = _now()
        job["live_logs"] = logs
        job["heartbeat_at"] = _now()
        job["updated_at"] = _now()

    def _set_stage(self, job_id: str, stage: str, status: str = "running") -> None:
        job = self._jobs[job_id]
        job["stage"] = stage
        job["status"] = status
        job["stage_started_at"] = _now()
        job["stage_elapsed_sec"] = 0.0
        job["heartbeat_at"] = _now()
        job["updated_at"] = _now()

    def _append_check(
        self,
        job_id: str,
        *,
        stage: str,
        name: str,
        ok: bool,
        detail: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        job = self._jobs[job_id]
        checks = job.get("checks") or []
        checks.append(
            {
                "timestamp": _now(),
                "stage": stage,
                "name": name,
                "ok": bool(ok),
                "detail": detail or "",
                "payload": payload or {},
            }
        )
        job["checks"] = checks
        job["heartbeat_at"] = _now()
        job["updated_at"] = _now()

    async def _consume_stream(
        self,
        job_id: str,
        reader: Optional[asyncio.StreamReader],
        stream_key: str,
        sink: List[str],
    ) -> None:
        if reader is None:
            return
        while True:
            line = await reader.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace")
            sink.append(text)
            self._append_live_log(job_id, stream_key, text)

    async def _run_command_with_tracking(
        self,
        *,
        job_id: str,
        step_name: str,
        command: str,
        timeout: int,
        cwd: str,
        env: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        start_dt = datetime.now()
        job = self._jobs[job_id]
        exec_env = os.environ.copy()
        if env:
            exec_env.update(env)

        self._set_stage_detail(job_id, step=step_name, message=f"Starting command: {command}")
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=exec_env,
        )
        self._active_processes[job_id] = process
        job["command"] = command
        job["process"] = {
            "pid": process.pid,
            "kind": step_name,
            "status": "running",
            "started_at": _now(),
            "ended_at": None,
            "command": command,
            "cwd": cwd,
        }
        stdout_rows: List[str] = []
        stderr_rows: List[str] = []
        stdout_task = asyncio.create_task(self._consume_stream(job_id, process.stdout, "stdout_tail", stdout_rows))
        stderr_task = asyncio.create_task(self._consume_stream(job_id, process.stderr, "stderr_tail", stderr_rows))

        timed_out = False
        try:
            loop = asyncio.get_running_loop()
            deadline = loop.time() + float(timeout)
            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    timed_out = True
                    with contextlib.suppress(ProcessLookupError):
                        process.kill()
                    await process.wait()
                    break
                try:
                    await asyncio.wait_for(process.wait(), timeout=min(1.0, remaining))
                    break
                except asyncio.TimeoutError:
                    job["heartbeat_at"] = _now()
                    job["updated_at"] = _now()
                    self._set_stage_detail(
                        job_id,
                        step=step_name,
                        message=f"Running pid={process.pid} ...",
                    )
        except asyncio.CancelledError:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            with contextlib.suppress(Exception):
                await process.wait()
            raise
        finally:
            await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
            self._active_processes.pop(job_id, None)

        elapsed = round((datetime.now() - start_dt).total_seconds(), 3)
        return_code = int(process.returncode or 0)
        status = "success" if return_code == 0 and not timed_out else ("timeout" if timed_out else "error")
        job["process"] = {
            "pid": process.pid,
            "kind": step_name,
            "status": status,
            "started_at": job.get("process", {}).get("started_at"),
            "ended_at": _now(),
            "command": command,
            "cwd": cwd,
        }
        self._set_stage_detail(
            job_id,
            step=step_name,
            message=(
                f"Command finished: status={status}, return_code={return_code}, elapsed={elapsed}s"
            ),
        )
        return {
            "status": status,
            "return_code": return_code,
            "stdout": "".join(stdout_rows),
            "stderr": "".join(stderr_rows),
            "execution_time": elapsed,
            "pid": process.pid,
            "command": command,
        }

    async def _resolve_dataset_via_hydra(
        self,
        *,
        job_id: str,
        python_bin: str,
        config_name: str,
        planner_module: str,
        cwd: str,
        env: Optional[Dict[str, str]] = None,
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        command = (
            f'"{python_bin}" -m {planner_module} '
            f'--config-name "{config_name}" --cfg job --resolve'
        )
        result = await self._run_command_with_tracking(
            job_id=job_id,
            step_name="hydra_resolve_dataset",
            command=command,
            timeout=120,
            cwd=cwd,
            env=env,
        )
        if result["status"] != "success":
            return None, result
        if yaml is not None:
            try:
                resolved = yaml.safe_load(result["stdout"] or "") or {}
                value = _dig(resolved, ["habitat", "dataset", "data_path"])
                if isinstance(value, str) and value.strip():
                    return value.strip(), result
            except Exception:
                pass
        match = re.search(r"(?m)^\s*data_path:\s*(.+?)\s*$", result["stdout"] or "")
        if not match:
            return None, result
        return str(match.group(1)).strip().strip('"').strip("'"), result

    async def _run_job(self, job_id: str, spec: Dict[str, Any], request: Dict[str, Any]) -> None:
        started_at = datetime.now()
        stage = "preflight"
        self._set_stage(job_id, stage=stage, status="running")
        self._set_stage_detail(job_id, step="baseline_resolve", message="Resolving baseline config.")

        try:
            validation_mode = str(request.get("validation_mode") or "instruction_case").strip() or "instruction_case"
            packet = spec.get("instruction_packet") or {}
            baseline_raw = (
                request.get("baseline_config")
                or packet.get("runtime_baseline")
                or "baselines/qwen3_centralized_zero_shot_react_summary_vllm_with_rebound.yaml"
            )
            baseline_path = _resolve_config_path(str(baseline_raw))
            if not baseline_path.exists():
                self._append_check(
                    job_id,
                    stage=stage,
                    name="baseline_exists",
                    ok=False,
                    detail=f"Baseline YAML not found: {baseline_path}",
                )
                raise RuntimeError(f"Baseline YAML not found: {baseline_path}")
            self._append_check(
                job_id,
                stage=stage,
                name="baseline_exists",
                ok=True,
                detail=str(baseline_path),
            )

            runtime_context = _build_runtime_context(request)
            runtime_cwd = str(runtime_context.get("cwd") or _project_root())
            runtime_env = runtime_context.get("env") or {}
            runtime_errors = runtime_context.get("errors") or []
            if runtime_errors:
                detail = "; ".join(str(item) for item in runtime_errors)
                self._append_check(
                    job_id,
                    stage=stage,
                    name="runtime_init",
                    ok=False,
                    detail=detail,
                )
                raise RuntimeError(detail)
            if not Path(runtime_cwd).exists():
                self._append_check(
                    job_id,
                    stage=stage,
                    name="runtime_workdir",
                    ok=False,
                    detail=f"Runtime workdir not found: {runtime_cwd}",
                )
                raise RuntimeError(f"Runtime workdir not found: {runtime_cwd}")
            self._append_check(
                job_id,
                stage=stage,
                name="runtime_init",
                ok=True,
                detail=f"cwd={runtime_cwd}",
                payload=runtime_context.get("summary") or {},
            )
            planner_module = str(
                request.get("planner_module") or "habitat_llm.examples.planner_demo_mp_new"
            ).strip()

            session_id = request.get("session_id") or packet.get("session_id")
            linked_messages: List[Dict[str, Any]] = []
            if session_id:
                session_payload = instruction_session_store.get_session(str(session_id))
                if session_payload:
                    linked_messages = list(session_payload.get("linked_messages") or [])
                    self._append_check(
                        job_id,
                        stage=stage,
                        name="session_linked_messages",
                        ok=True,
                        detail=f"session={session_id}, linked_messages={len(linked_messages)}",
                    )
                else:
                    self._append_check(
                        job_id,
                        stage=stage,
                        name="session_linked_messages",
                        ok=False,
                        detail=f"session not found: {session_id}",
                    )

            python_bin = str(runtime_context.get("python_bin") or _pick_python_bin())
            self._set_stage_detail(job_id, step="runtime_check", message="Checking python runtime.")
            if not Path(python_bin).exists() and not _command_exists(python_bin):
                self._append_check(
                    job_id,
                    stage=stage,
                    name="python_runtime",
                    ok=False,
                    detail=f"Python runtime not found: {python_bin}",
                )
                raise RuntimeError(
                    f"Python runtime not found: {python_bin}. Set EAB_HABITAT_PYTHON to a valid interpreter."
                )
            self._append_check(
                job_id,
                stage=stage,
                name="python_runtime",
                ok=True,
                detail=f"Using python runtime: {python_bin}",
            )

            config_name = _to_hydra_config_name(baseline_path, str(baseline_raw))
            baseline_yaml = _load_yaml(baseline_path)
            self._set_stage_detail(job_id, step="dataset_resolve", message="Resolving dataset path.")
            dataset_raw = request.get("dataset_path_override") or _dig(
                baseline_yaml, ["habitat", "dataset", "data_path"]
            )
            if not dataset_raw:
                dataset_raw, hydra_resolve_result = await self._resolve_dataset_via_hydra(
                    job_id=job_id,
                    python_bin=python_bin,
                    config_name=config_name,
                    planner_module=planner_module,
                    cwd=runtime_cwd,
                    env=runtime_env,
                )
                self._append_check(
                    job_id,
                    stage=stage,
                    name="hydra_dataset_resolve",
                    ok=hydra_resolve_result.get("status") == "success",
                    detail=(
                        f"status={hydra_resolve_result.get('status')}, "
                        f"return_code={hydra_resolve_result.get('return_code')}, "
                        f"pid={hydra_resolve_result.get('pid')}"
                    ),
                    payload={
                        "pid": hydra_resolve_result.get("pid"),
                        "execution_time": hydra_resolve_result.get("execution_time"),
                        "stderr_tail": _trim_lines(hydra_resolve_result.get("stderr") or "", 10),
                    },
                )
            dataset_abs, dataset_candidates = _resolve_dataset_path(str(dataset_raw or ""))
            if dataset_abs is None:
                fallback_dataset = _discover_dataset_fallback()
                if fallback_dataset is not None:
                    dataset_abs = fallback_dataset
                    self._append_check(
                        job_id,
                        stage=stage,
                        name="dataset_fallback",
                        ok=True,
                        detail=f"Using fallback dataset: {dataset_abs}",
                    )
                else:
                    self._append_check(
                        job_id,
                        stage=stage,
                        name="dataset_fallback",
                        ok=False,
                        detail="No fallback dataset found. Set EAB_DATASET_PATH to provide one.",
                    )
            if dataset_abs is None:
                candidate_lines = ", ".join(str(path) for path in dataset_candidates) if dataset_candidates else "none"
                self._append_check(
                    job_id,
                    stage=stage,
                    name="dataset_exists",
                    ok=False,
                    detail=f"Dataset path missing or not found. value={dataset_raw} candidates={candidate_lines}",
                )
                raise RuntimeError(
                    "Dataset path missing or not found. "
                    "Provide dataset_path_override or update baseline dataset path."
                )
            self._append_check(
                job_id,
                stage=stage,
                name="dataset_exists",
                ok=True,
                detail=str(dataset_abs),
            )

            timeout_sec = int(request.get("timeout_sec") or 900)
            episode_indices = request.get("episode_indices") or [0]
            if not isinstance(episode_indices, list) or not episode_indices:
                episode_indices = [0]
            planner_overrides_raw = request.get("planner_overrides") or []
            planner_overrides: List[str] = []
            if isinstance(planner_overrides_raw, list):
                planner_overrides = [str(item).strip() for item in planner_overrides_raw if str(item).strip()]

            runtime_spec = deepcopy(spec)
            if validation_mode == "instruction_case":
                self._set_stage_detail(
                    job_id,
                    step="instruction_case_codegen",
                    message="Generating instruction-case artifacts from disambiguated session context.",
                )
                codegen_bundle = await instruction_case_codegen_service.generate_instruction_case_artifacts(
                    spec=runtime_spec,
                    linked_messages=linked_messages,
                    baseline_config=str(baseline_raw),
                )
                for source_name in ["skill_class_code", "flow_draft_code"]:
                    source_code = str(codegen_bundle.get(source_name) or "")
                    try:
                        compile(source_code, f"<{source_name}>", "exec")
                        self._append_check(
                            job_id,
                            stage=stage,
                            name=f"{source_name}_py_compile",
                            ok=True,
                            detail="py_compile check passed.",
                        )
                    except Exception as exc:
                        self._append_check(
                            job_id,
                            stage=stage,
                            name=f"{source_name}_py_compile",
                            ok=False,
                            detail=str(exc),
                        )
                        raise RuntimeError(f"{source_name} py_compile failed: {exc}") from exc

                runtime_spec = _deep_merge_dict(
                    runtime_spec,
                    {
                        "middle_representation": {
                            "discussion_digest": codegen_bundle.get("discussion_digest") or {},
                            "skill_class_code": codegen_bundle.get("skill_class_code") or "",
                            "flow_draft_code": codegen_bundle.get("flow_draft_code") or "",
                            "logic_graph_mermaid": codegen_bundle.get("logic_graph_mermaid") or "",
                            "instruction_case": codegen_bundle.get("instruction_case") or {},
                        },
                        "derived_artifacts": {
                            "skill_logic.py": codegen_bundle.get("skill_class_code") or "",
                            "flow_draft.py": codegen_bundle.get("flow_draft_code") or "",
                            "logic_graph.mmd": codegen_bundle.get("logic_graph_mermaid") or "",
                            "instruction_case.json": codegen_bundle.get("instruction_case") or {},
                            "prompt_templates": codegen_bundle.get("prompt_templates") or {},
                        },
                        "runtime_contract": codegen_bundle.get("runtime_contract") or {},
                    },
                )
                self._jobs[job_id]["instruction_case_summary"] = codegen_bundle.get("instruction_case_summary")
                self._append_check(
                    job_id,
                    stage=stage,
                    name="instruction_case_codegen",
                    ok=True,
                    detail="Generated skill class, flow draft, logic graph, and instruction case artifacts.",
                    payload={
                        "messages_used_count": (codegen_bundle.get("instruction_case_summary") or {}).get(
                            "messages_used_count", 0
                        ),
                        "grounding_mode": (codegen_bundle.get("instruction_case_summary") or {}).get(
                            "grounding_mode"
                        ),
                    },
                )

            stage = "materialize"
            self._set_stage(job_id, stage=stage, status="running")
            self._set_stage_detail(job_id, step="materialize_to_sandbox", message="Materializing sandbox artifacts.")
            manifest = materialize_to_sandbox(spec["id"], runtime_spec)
            self._jobs[job_id]["manifest"] = manifest
            sandbox_root = Path(manifest["sandbox_root"])
            output_root = sandbox_root / "output"
            results_root = output_root / "results"
            logs_root = output_root / "logs"
            video_root = output_root / "video"
            results_root.mkdir(parents=True, exist_ok=True)
            logs_root.mkdir(parents=True, exist_ok=True)
            video_root.mkdir(parents=True, exist_ok=True)
            self._jobs[job_id]["outputs"] = {
                "sandbox_root": str(sandbox_root),
                "results_root": str(results_root),
                "logs_root": str(logs_root),
                "video_root": str(video_root),
            }
            self._append_check(
                job_id,
                stage=stage,
                name="materialize_to_sandbox",
                ok=True,
                detail=str(sandbox_root),
            )

            stage = "habitat_run"
            self._set_stage(job_id, stage=stage, status="running")
            hydra_dataset_value = _build_hydra_dataset_path(dataset_abs)
            episode_literal = json.dumps(episode_indices, ensure_ascii=False)
            instruction_case_path = (
                manifest.get("generated_files", {}).get("instruction_case.json")
                or str(sandbox_root / "artifacts" / "instruction_case.json")
            )
            if validation_mode == "instruction_case":
                override_parts: List[str] = []
                for item in planner_overrides:
                    escaped_item = str(item).replace('"', '\\"')
                    override_parts.append(f'--planner-override "{escaped_item}"')
                override_args = " ".join(override_parts)
                command = (
                    f'"{python_bin}" -m habitat_llm.examples.instruction_case_validate '
                    f'--config-name "{config_name}" '
                    f'--instruction-case "{instruction_case_path}" '
                    f'--results-dir "{results_root.as_posix()}" '
                    f'--dataset-path "{hydra_dataset_value}" '
                    f'--episode-indices "{episode_literal}" '
                    f"--timeout-sec {timeout_sec} "
                    f'--planner-module "{planner_module}" '
                    f'--python-bin "{python_bin}"'
                )
                if override_args:
                    command = f"{command} {override_args}"
                step_name = "instruction_case_validate"
            else:
                command = (
                    f'"{python_bin}" -m {planner_module} '
                    f'--config-name "{config_name}" '
                    f'num_proc=1 num_runs_per_episode=1 ++episode_indices={episode_literal} '
                    f'evaluation.save_video=True '
                    f'paths.results_dir="{results_root.as_posix()}" '
                    f'habitat.dataset.data_path="{hydra_dataset_value}"'
                )
                step_name = "planner_run"

            result = await self._run_command_with_tracking(
                job_id=job_id,
                step_name=step_name,
                command=command,
                timeout=timeout_sec,
                cwd=runtime_cwd,
                env={
                    **runtime_env,
                    "EVO_HABITAT_SANDBOX_ROOT": str(sandbox_root),
                },
            )
            stdout_path = logs_root / "habitat_stdout.log"
            stderr_path = logs_root / "habitat_stderr.log"
            stdout_path.write_text(result.get("stdout") or "", encoding="utf-8")
            stderr_path.write_text(result.get("stderr") or "", encoding="utf-8")
            self._jobs[job_id]["outputs"].update(
                {"stdout_path": str(stdout_path), "stderr_path": str(stderr_path)}
            )
            self._append_check(
                job_id,
                stage=stage,
                name=step_name,
                ok=result.get("status") == "success",
                detail=(
                    f"status={result.get('status')}, return_code={result.get('return_code')}, "
                    f"pid={result.get('pid')}"
                ),
                payload={
                    "pid": result.get("pid"),
                    "execution_time": result.get("execution_time"),
                    "stdout_tail": _trim_lines(result.get("stdout") or "", 20),
                    "stderr_tail": _trim_lines(result.get("stderr") or "", 20),
                },
            )
            if result.get("status") != "success":
                raise RuntimeError(
                    f"Habitat run failed: status={result.get('status')} return_code={result.get('return_code')}"
                )

            stage = "collect_metrics"
            self._set_stage(job_id, stage=stage, status="running")
            self._set_stage_detail(job_id, step="collect_metrics", message="Collecting metrics and grading.")
            metrics: Dict[str, Any] = {}
            grade_report: Dict[str, Any] = {}
            episode_csv = results_root / "episode_result_log.csv"
            if validation_mode == "instruction_case":
                metrics = _read_json_file(results_root / "metrics.json")
                grade_report = _read_json_file(results_root / "grade_report.json")
                if not metrics:
                    metrics_row = _read_last_csv_row(episode_csv)
                    metrics = {
                        "task_state_success": _float_or_default(metrics_row.get("task_state_success"), 0.0),
                        "task_percent_complete": _float_or_default(
                            metrics_row.get("task_percent_complete"), 0.0
                        ),
                        "runtime": _float_or_default(metrics_row.get("runtime"), 0.0),
                    }
                if not grade_report:
                    success_val = _float_or_default(metrics.get("task_state_success"), 0.0)
                    percent_val = _float_or_default(metrics.get("task_percent_complete"), 0.0)
                    grade_report = {
                        "grade": "A"
                        if success_val >= 1.0 and percent_val >= 0.9
                        else "B"
                        if success_val >= 1.0 or percent_val >= 0.9
                        else "C",
                        "reasons": ["fallback_grade_rule"],
                        "subscores": {
                            "task_state_success": success_val,
                            "task_percent_complete": percent_val,
                            "disambiguation_consistent": True,
                            "runtime_stable": True,
                        },
                    }
            else:
                metrics_row = _read_last_csv_row(episode_csv)
                task_state_success = _float_or_default(metrics_row.get("task_state_success"), 0.0)
                task_percent_complete = _float_or_default(metrics_row.get("task_percent_complete"), 0.0)
                runtime = _float_or_default(metrics_row.get("runtime"), 0.0)
                pass_rule_met = (task_state_success >= 1.0) or (task_percent_complete >= 0.9)
                metrics = {
                    "task_state_success": task_state_success,
                    "task_percent_complete": task_percent_complete,
                    "runtime": runtime,
                    "pass_rule": "task_state_success>=1 OR task_percent_complete>=0.9",
                    "pass_rule_met": pass_rule_met,
                }
                grade_report = {
                    "grade": "A" if pass_rule_met else "C",
                    "reasons": ["legacy_pass_rule"],
                    "subscores": {
                        "task_state_success": task_state_success,
                        "task_percent_complete": task_percent_complete,
                    },
                }
            self._jobs[job_id]["metrics"] = metrics
            self._jobs[job_id]["grade_report"] = grade_report
            self._jobs[job_id]["outputs"]["episode_result_csv"] = str(episode_csv) if episode_csv.exists() else None
            if (results_root / "metrics.json").exists():
                self._jobs[job_id]["outputs"]["metrics_json"] = str(results_root / "metrics.json")
            if (results_root / "grade_report.json").exists():
                self._jobs[job_id]["outputs"]["grade_report_json"] = str(results_root / "grade_report.json")
            if (results_root / "run_events.jsonl").exists():
                self._jobs[job_id]["outputs"]["run_events_jsonl"] = str(results_root / "run_events.jsonl")
            self._append_check(
                job_id,
                stage=stage,
                name="collect_episode_metrics",
                ok=bool(metrics),
                detail=f"metrics collected; grade={grade_report.get('grade') or '-'}",
                payload={
                    "metrics": metrics,
                    "grade_report": grade_report,
                },
            )

            stage = "video_postprocess"
            self._set_stage(job_id, stage=stage, status="running")
            self._set_stage_detail(job_id, step="collect_video", message="Collecting raw video.")
            raw_video_source = _discover_latest_mp4(results_root)
            if raw_video_source is None:
                self._append_check(
                    job_id,
                    stage=stage,
                    name="discover_raw_video",
                    ok=False,
                    detail=f"No mp4 generated under {results_root}",
                )
                raise RuntimeError("No MP4 output found in sandbox results.")

            raw_video_path = video_root / "raw.mp4"
            shutil.copy2(raw_video_source, raw_video_path)
            self._jobs[job_id]["videos"]["raw"] = {
                "available": True,
                "path": str(raw_video_path),
                "media_url": f"/api/capabilities/sandbox-jobs/{job_id}/media/raw",
                "source_path": str(raw_video_source),
            }
            self._append_check(
                job_id,
                stage=stage,
                name="raw_video_collected",
                ok=True,
                detail=str(raw_video_path),
            )

            profile = request.get("video_profile") or {}
            speed = float(profile.get("speed") or 0.5)
            duration_sec = float(profile.get("duration_sec") or 12.0)
            processed_video_path = video_root / "demo_slow_0.5x_12s.mp4"
            self._set_stage_detail(
                job_id,
                step="video_postprocess",
                message=f"Generating demo video (speed={speed}x, duration={duration_sec}s).",
            )
            processed_ok, postprocess_detail = await self._postprocess_video(
                raw_video_path=raw_video_path,
                processed_video_path=processed_video_path,
                speed=speed,
                duration_sec=duration_sec,
            )
            if processed_ok:
                self._jobs[job_id]["videos"]["processed"] = {
                    "available": True,
                    "path": str(processed_video_path),
                    "media_url": f"/api/capabilities/sandbox-jobs/{job_id}/media/processed",
                }
                self._append_check(
                    job_id,
                    stage=stage,
                    name="video_postprocess",
                    ok=True,
                    detail=postprocess_detail,
                )
            else:
                self._jobs[job_id]["videos"]["postprocess_skipped"] = True
                self._append_check(
                    job_id,
                    stage=stage,
                    name="video_postprocess",
                    ok=False,
                    detail=postprocess_detail,
                )

            stage = "finalize"
            self._set_stage(job_id, stage=stage, status="running")
            self._set_stage_detail(job_id, step="finalize", message="Finalizing sandbox summary.")
            planner_ok = result.get("status") == "success"
            raw_ok = bool(self._jobs[job_id]["videos"]["raw"]["available"])
            grade = str((grade_report or {}).get("grade") or "C").upper()
            success = planner_ok and raw_ok
            elapsed_sec = round((datetime.now() - started_at).total_seconds(), 3)

            self._jobs[job_id]["summary"] = {
                "success": success,
                "elapsed_sec": elapsed_sec,
                "planner_status": result.get("status"),
                "grade": grade,
                "metrics_passed": grade in {"A", "B"},
                "raw_video_available": raw_ok,
                "validation_mode": validation_mode,
            }
            self._jobs[job_id]["status"] = "completed" if success else "failed"
            self._jobs[job_id]["stage"] = "finalize"
            self._jobs[job_id]["updated_at"] = _now()
            self._append_check(
                job_id,
                stage=stage,
                name="sandbox_finalize",
                ok=success,
                detail=(
                    f"Sandbox validation completed with grade {grade}."
                    if success
                    else "Sandbox validation failed."
                ),
                payload=self._jobs[job_id]["summary"],
            )
        except asyncio.CancelledError:
            self._jobs[job_id]["status"] = "cancelled"
            self._jobs[job_id]["error"] = "Sandbox validation cancelled."
            self._jobs[job_id]["updated_at"] = _now()
            self._set_stage_detail(job_id, step="cancelled", message="Sandbox validation cancelled.")
            self._append_check(
                job_id,
                stage=stage,
                name="job_cancelled",
                ok=False,
                detail="Cancelled while running.",
            )
            return
        except Exception as exc:
            self._jobs[job_id]["status"] = "failed"
            self._jobs[job_id]["stage"] = stage
            self._jobs[job_id]["error"] = str(exc)
            self._jobs[job_id]["updated_at"] = _now()
            self._set_stage_detail(job_id, step="exception", message=str(exc))
            self._append_check(
                job_id,
                stage=stage,
                name="job_exception",
                ok=False,
                detail=str(exc),
            )
        finally:
            self._active_processes.pop(job_id, None)
            self._tasks.pop(job_id, None)

    async def _postprocess_video(
        self,
        *,
        raw_video_path: Path,
        processed_video_path: Path,
        speed: float,
        duration_sec: float,
    ) -> Tuple[bool, str]:
        if speed <= 0:
            return False, "Invalid speed value."
        if duration_sec <= 0:
            return False, "Invalid duration value."

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            setpts = 1.0 / speed
            cmd = [
                ffmpeg,
                "-y",
                "-i",
                str(raw_video_path),
                "-filter:v",
                f"setpts={setpts}*PTS,fps=30,tpad=stop_mode=clone:stop_duration={duration_sec}",
                "-t",
                str(duration_sec),
                "-an",
                str(processed_video_path),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode == 0 and processed_video_path.exists():
                return True, "Processed with ffmpeg."
            return False, f"ffmpeg failed: {proc.stderr[-400:] if proc.stderr else 'unknown error'}"

        imageio_spec = importlib.util.find_spec("imageio")
        if imageio_spec is None:
            return False, "Postprocess skipped: ffmpeg/imageio unavailable."

        try:
            import imageio.v2 as imageio
        except Exception:
            return False, "Postprocess skipped: unable to import imageio."

        try:
            reader = imageio.get_reader(str(raw_video_path))
            meta = reader.get_meta_data() or {}
            fps = float(meta.get("fps") or 30.0)
            frames = [frame for frame in reader]
            reader.close()
            if not frames:
                return False, "Postprocess skipped: no frames in raw video."
            target_frames = max(1, int(round(duration_sec * fps)))
            slowdown_factor = 1.0 / speed
            writer = imageio.get_writer(str(processed_video_path), fps=fps)
            for index in range(target_frames):
                source_index = min(int(index / slowdown_factor), len(frames) - 1)
                writer.append_data(frames[source_index])
            writer.close()
            return True, "Processed with imageio fallback."
        except Exception as exc:
            return False, f"Postprocess skipped: imageio fallback failed ({exc})."


sandbox_validation_manager = SandboxValidationManager()


