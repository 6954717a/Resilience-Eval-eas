"""Bounded subprocess scheduling primitives for resilience evaluation.

The scheduler itself uses threads only to wait for isolated worker processes.
Habitat-Sim, planners, and critics remain inside the worker processes so their
native allocations are reclaimed when each episode finishes.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import tempfile
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ExecutionSettings:
    """Resource limits for one simulation host."""

    max_parallel_workers: int = 1
    episodes_per_worker: int = 1
    min_available_memory_gb: float = 48.0

    @classmethod
    def from_mapping(cls, value: Optional[Mapping[str, Any]]) -> "ExecutionSettings":
        raw = value or {}
        settings = cls(
            max_parallel_workers=int(raw.get("max_parallel_workers", 1)),
            episodes_per_worker=int(raw.get("episodes_per_worker", 1)),
            min_available_memory_gb=float(raw.get("min_available_memory_gb", 48.0)),
        )
        if settings.max_parallel_workers < 1:
            raise ValueError("max_parallel_workers must be >= 1")
        if settings.episodes_per_worker < 1:
            raise ValueError("episodes_per_worker must be >= 1")
        if settings.min_available_memory_gb < 0:
            raise ValueError("min_available_memory_gb must be >= 0")
        return settings


@dataclass
class JobOutcome:
    index: int
    job_id: str
    status: str
    result: Any = None
    error: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None


@dataclass
class BatchRun:
    outcomes: List[JobOutcome]
    max_active_workers: int


class BatchExecutionError(RuntimeError):
    """Raised after all already-started workers exit following a job failure."""

    def __init__(self, report: BatchRun):
        self.report = report
        failed = [item for item in report.outcomes if item.status == "failed"]
        details = "; ".join(
            f"{item.job_id}: {item.error or 'unknown error'}" for item in failed
        )
        super().__init__(f"{len(failed)} rollout worker(s) failed: {details}")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_attempt_dir(worker_root: Path) -> Path:
    """Atomically create a fresh directory for one worker batch attempt."""

    root = Path(worker_root)
    root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="attempt_", dir=root))


def read_available_memory_gb(meminfo_path: Path = Path("/proc/meminfo")) -> Optional[float]:
    """Return Linux MemAvailable in GiB, or ``None`` on other platforms."""

    try:
        for line in meminfo_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                kib = float(line.split()[1])
                return kib / (1024.0 * 1024.0)
    except (OSError, ValueError, IndexError):
        return None
    return None


def chunk_episode_ids(
    episode_ids: Sequence[Any], episodes_per_worker: int
) -> List[Tuple[str, ...]]:
    if episodes_per_worker < 1:
        raise ValueError("episodes_per_worker must be >= 1")
    normalized = [str(episode_id) for episode_id in episode_ids]
    return [
        tuple(normalized[index : index + episodes_per_worker])
        for index in range(0, len(normalized), episodes_per_worker)
    ]


def run_bounded_jobs(
    jobs: Sequence[Tuple[str, Any]],
    worker: Callable[[Any], Any],
    settings: ExecutionSettings,
    *,
    memory_reader: Callable[[], Optional[float]] = read_available_memory_gb,
    continue_after_worker_failure: bool = False,
) -> BatchRun:
    """Execute jobs with a fixed worker cap and a Linux memory admission gate.

    By default, a worker failure stops admission of new jobs. Episode-isolated
    callers may set ``continue_after_worker_failure`` so a bad episode is
    recorded while later episodes still run. Scheduler-level failures such as
    an unmet memory reserve remain fatal in both modes. Returned outcomes always
    follow input order.
    """

    indexed_jobs = [
        (index, str(job_id), payload)
        for index, (job_id, payload) in enumerate(jobs)
    ]
    if not indexed_jobs:
        return BatchRun(outcomes=[], max_active_workers=0)

    outcomes: dict[int, JobOutcome] = {}
    active: dict[Future[JobOutcome], int] = {}
    next_index = 0
    failure_detected = False
    max_active = 0

    def collect_completed(completed: Sequence[Future[JobOutcome]]) -> None:
        nonlocal failure_detected
        for future in completed:
            active.pop(future, None)
            outcome = future.result()
            outcomes[outcome.index] = outcome
            if outcome.status == "failed" and not continue_after_worker_failure:
                failure_detected = True

    def invoke(index: int, job_id: str, payload: Any) -> JobOutcome:
        started_at = utc_now_iso()
        try:
            result = worker(payload)
            return JobOutcome(
                index=index,
                job_id=job_id,
                status="completed",
                result=result,
                started_at=started_at,
                finished_at=utc_now_iso(),
            )
        except Exception as exc:  # worker errors are aggregated by the parent
            return JobOutcome(
                index=index,
                job_id=job_id,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
                started_at=started_at,
                finished_at=utc_now_iso(),
            )

    with ThreadPoolExecutor(max_workers=settings.max_parallel_workers) as executor:
        while next_index < len(indexed_jobs) or active:
            # Observe failures that completed between scheduler iterations
            # before admitting another worker.
            collect_completed([future for future in active if future.done()])
            while (
                not failure_detected
                and next_index < len(indexed_jobs)
                and len(active) < settings.max_parallel_workers
            ):
                available_gb = memory_reader()
                if (
                    available_gb is not None
                    and available_gb < settings.min_available_memory_gb
                ):
                    if active:
                        break
                    index, job_id, _payload = indexed_jobs[next_index]
                    outcomes[index] = JobOutcome(
                        index=index,
                        job_id=job_id,
                        status="failed",
                        error=(
                            f"available memory {available_gb:.2f} GiB is below "
                            f"the {settings.min_available_memory_gb:.2f} GiB reserve"
                        ),
                        started_at=utc_now_iso(),
                        finished_at=utc_now_iso(),
                    )
                    next_index += 1
                    failure_detected = True
                    break

                index, job_id, payload = indexed_jobs[next_index]
                future = executor.submit(invoke, index, job_id, payload)
                active[future] = index
                next_index += 1
                max_active = max(max_active, len(active))

            if not active:
                break

            completed, _ = wait(tuple(active), return_when=FIRST_COMPLETED)
            collect_completed(list(completed))

        if failure_detected:
            for index, job_id, _payload in indexed_jobs[next_index:]:
                outcomes[index] = JobOutcome(
                    index=index,
                    job_id=job_id,
                    status="not_started",
                    error="not started after an earlier worker failure",
                )

    report = BatchRun(
        outcomes=[outcomes[index] for index in range(len(indexed_jobs))],
        max_active_workers=max_active,
    )
    if failure_detected:
        raise BatchExecutionError(report)
    return report
