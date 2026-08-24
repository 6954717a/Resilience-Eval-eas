"""Append-only clean trajectory evidence for cumulative StageBaseline fits."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import uuid

from filelock import FileLock

from habitat_llm.evaluation.critic_export_contract import load_full_stability_export
from habitat_llm.evaluation.stage_baseline.alignment import (
    STAGE_BASELINE_ALIGNMENT_SCOPE,
    judge_models_match,
    stage_baseline_alignment_key,
)


STAGE_BASELINE_EVIDENCE_SCHEMA_VERSION = 1
logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _temporary_path(path: Path) -> Path:
    """Return a process-safe temporary peer for an atomic file install."""

    return path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_gzip_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    try:
        with gzip.open(temporary, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle, separators=(",", ":"), ensure_ascii=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class StageBaselineEvidence:
    evidence_id: str
    contract_id: str
    episode_id: str
    seed: int
    source_hash: str
    trajectory_path: str
    total_transitions: int
    judge_model: str
    state_encoder_id: str
    value_checkpoint_id: str
    reward_shaper_valid: bool
    critic_lifecycle_valid: bool
    created_at: str
    alignment_key: str = ""
    value_alignment_role: str = "diagnostic_only"
    schema_version: int = STAGE_BASELINE_EVIDENCE_SCHEMA_VERSION

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "StageBaselineEvidence":
        judge_model = str(payload.get("judge_model") or "")
        episode_id = str(payload.get("episode_id") or "")
        return cls(
            evidence_id=str(payload.get("evidence_id") or ""),
            contract_id=str(payload.get("contract_id") or ""),
            episode_id=episode_id,
            seed=int(payload.get("seed", -1)),
            source_hash=str(payload.get("source_hash") or ""),
            trajectory_path=str(payload.get("trajectory_path") or ""),
            total_transitions=int(payload.get("total_transitions", 0) or 0),
            judge_model=judge_model,
            state_encoder_id=str(payload.get("state_encoder_id") or ""),
            value_checkpoint_id=str(payload.get("value_checkpoint_id") or ""),
            reward_shaper_valid=bool(payload.get("reward_shaper_valid", False)),
            critic_lifecycle_valid=bool(
                payload.get("critic_lifecycle_valid", False)
            ),
            created_at=str(payload.get("created_at") or ""),
            alignment_key=str(payload.get("alignment_key") or "")
            or stage_baseline_alignment_key(judge_model, episode_id),
            value_alignment_role=str(
                payload.get("value_alignment_role") or "diagnostic_only"
            ),
            schema_version=int(
                payload.get(
                    "schema_version", STAGE_BASELINE_EVIDENCE_SCHEMA_VERSION
                )
            ),
        )


class StageBaselineEvidenceStore:
    """Persistent evidence and immutable snapshots for one baseline contract."""

    def __init__(
        self,
        root: Path,
        *,
        contract_id: str,
        contract_metadata: Mapping[str, Any],
    ) -> None:
        self.root = Path(root)
        self.contract_id = str(contract_id)
        if not self.contract_id:
            raise ValueError("StageBaseline evidence store requires a contract_id")
        self.contract_root = self.root / "contracts" / self.contract_id
        self.evidence_root = self.contract_root / "evidence"
        self.snapshot_root = self.contract_root / "snapshots"
        self.contract_path = self.contract_root / "contract.json"
        self.contract_metadata = dict(contract_metadata)

    def _mutation_lock(self) -> FileLock:
        self.contract_root.mkdir(parents=True, exist_ok=True)
        return FileLock(str(self.contract_root / ".evidence_store.lock"))

    def ensure_contract(self) -> None:
        expected = {
            "schema_version": STAGE_BASELINE_EVIDENCE_SCHEMA_VERSION,
            "contract_id": self.contract_id,
            "contract": self.contract_metadata,
        }
        with self._mutation_lock():
            if self.contract_path.exists():
                with open(self.contract_path, "r", encoding="utf-8") as handle:
                    existing = json.load(handle)
                if existing != expected:
                    raise ValueError(
                        "StageBaseline contract metadata conflicts with persistent "
                        f"contract {self.contract_id}"
                    )
                return
            _atomic_json(self.contract_path, expected)

    def list_evidence(
        self,
        episode_ids: Optional[Sequence[Any]] = None,
        *,
        require_reward_shaper: bool = False,
    ) -> List[StageBaselineEvidence]:
        if not self.evidence_root.exists():
            return []
        allowed = None if episode_ids is None else {str(value) for value in episode_ids}
        records: List[StageBaselineEvidence] = []
        for path in sorted(self.evidence_root.glob("*/*/manifest.json")):
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    record = StageBaselineEvidence.from_mapping(json.load(handle))
            except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
                continue
            if record.schema_version != STAGE_BASELINE_EVIDENCE_SCHEMA_VERSION:
                continue
            if record.contract_id != self.contract_id:
                continue
            if not judge_models_match(
                record.judge_model, self.contract_metadata.get("judge_model")
            ):
                continue
            if record.alignment_key != stage_baseline_alignment_key(
                record.judge_model, record.episode_id
            ):
                continue
            if allowed is not None and record.episode_id not in allowed:
                continue
            if require_reward_shaper and not record.reward_shaper_valid:
                continue
            if not (self.contract_root / record.trajectory_path).exists():
                continue
            records.append(record)
        return records

    def evidence_seed_map(
        self, episode_ids: Optional[Sequence[Any]] = None
    ) -> Dict[str, List[int]]:
        values: Dict[str, set[int]] = {}
        for record in self.list_evidence(episode_ids):
            values.setdefault(record.episode_id, set()).add(int(record.seed))
        return {
            episode_id: sorted(seeds)
            for episode_id, seeds in sorted(values.items())
        }

    def ingest_exports(
        self,
        export_paths: Sequence[Path],
        *,
        source_metadata: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> Dict[str, Any]:
        self.ensure_contract()
        with self._mutation_lock():
            return self._ingest_exports_locked(
                export_paths,
                source_metadata=source_metadata,
            )

    def _ingest_exports_locked(
        self,
        export_paths: Sequence[Path],
        *,
        source_metadata: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> Dict[str, Any]:
        existing = self.list_evidence(require_reward_shaper=False)
        by_source = {record.source_hash: record for record in existing}
        accepted: List[StageBaselineEvidence] = []
        reused: List[str] = []
        conflicts: List[Dict[str, Any]] = []
        excluded: List[Dict[str, Any]] = []

        for raw_path in export_paths:
            path = Path(raw_path)
            export = load_full_stability_export(path)
            if not export.valid:
                excluded.append({"path": str(path), "reason": export.missing_reason})
                continue
            episode_ids = {
                str(record.get("episode_id") or "") for record in export.records
            }
            episode_ids.discard("")
            if len(episode_ids) != 1:
                excluded.append(
                    {"path": str(path), "reason": "critic_export_not_single_episode"}
                )
                continue
            episode_id = next(iter(episode_ids))
            resolved = str(path.resolve())
            provenance = dict(
                (source_metadata or {}).get(
                    resolved, (source_metadata or {}).get(str(path), {})
                )
                or {}
            )
            try:
                seed = int(provenance.get("seed"))
            except (TypeError, ValueError):
                excluded.append(
                    {"path": str(path), "reason": "calibration_seed_missing"}
                )
                continue
            source_hash = _sha256_file(path)
            if source_hash in by_source:
                reused.append(by_source[source_hash].evidence_id)
                continue
            export_metadata = dict(export.metadata or {})
            export_judge_model = str(export_metadata.get("judge_model") or "").strip()
            provenance_judge_model = str(provenance.get("judge_model") or "").strip()
            incoming_judge_model = export_judge_model or provenance_judge_model
            if not incoming_judge_model:
                excluded.append(
                    {"path": str(path), "reason": "judge_model_missing"}
                )
                continue
            if not judge_models_match(
                incoming_judge_model, self.contract_metadata.get("judge_model")
            ):
                excluded.append(
                    {"path": str(path), "reason": "judge_model_mismatch"}
                )
                continue
            provenance_episode_id = str(provenance.get("episode_id") or "").strip()
            if provenance_episode_id and provenance_episode_id != episode_id:
                excluded.append(
                    {"path": str(path), "reason": "episode_id_mismatch"}
                )
                continue
            reward_shaper_valid = bool(
                export_metadata.get(
                    "reward_shaper_valid",
                    provenance.get("reward_shaper_valid", False),
                )
            )
            critic_lifecycle_valid = bool(
                export_metadata.get(
                    "critic_lifecycle_valid",
                    provenance.get("critic_lifecycle_valid", False),
                )
            )
            evidence_id = f"seed_{seed}_{uuid.uuid4().hex[:12]}"
            evidence_dir = self.evidence_root / episode_id / evidence_id
            trajectory_path = evidence_dir / "trajectory.json.gz"
            manifest_path = evidence_dir / "manifest.json"
            payload = {
                "episode_id": episode_id,
                "task_family": str(export.records[0].get("task_family") or "other"),
                "schema_version": 1,
                "export_manifest": {
                    "full_stability_trajectory": True,
                    "minimal_schema": True,
                    "record_mode": "state_only",
                },
                "selection_summary": {
                    "total_transitions": int(export.total_transitions),
                    "selected_transitions": int(export.selected_transitions),
                },
                "records": list(export.records),
            }
            _atomic_gzip_json(trajectory_path, payload)
            record = StageBaselineEvidence(
                evidence_id=evidence_id,
                contract_id=self.contract_id,
                episode_id=episode_id,
                seed=seed,
                source_hash=source_hash,
                trajectory_path=str(trajectory_path.relative_to(self.contract_root)),
                total_transitions=int(export.total_transitions),
                judge_model=incoming_judge_model,
                state_encoder_id=str(
                    export_metadata.get("state_encoder_id")
                    or provenance.get("state_encoder_id")
                    or ""
                ),
                value_checkpoint_id=str(
                    export_metadata.get("value_checkpoint_id")
                    or provenance.get("value_checkpoint_id")
                    or ""
                ),
                reward_shaper_valid=reward_shaper_valid,
                critic_lifecycle_valid=critic_lifecycle_valid,
                created_at=_utc_now(),
                alignment_key=stage_baseline_alignment_key(
                    incoming_judge_model, episode_id
                ),
                value_alignment_role="diagnostic_only",
            )
            _atomic_json(manifest_path, asdict(record))
            accepted.append(record)
            by_source[source_hash] = record

        return {
            "accepted": [asdict(record) for record in accepted],
            "accepted_count": len(accepted),
            "reused_evidence_ids": sorted(set(reused)),
            "conflicts": conflicts,
            "excluded": excluded,
        }

    def estimator_inputs(
        self, episode_ids: Optional[Sequence[Any]] = None
    ) -> Tuple[List[Path], Dict[str, Dict[str, Any]], List[StageBaselineEvidence]]:
        records = self.list_evidence(episode_ids)
        paths: List[Path] = []
        metadata: Dict[str, Dict[str, Any]] = {}
        for record in records:
            path = self.contract_root / record.trajectory_path
            paths.append(path)
            provenance = {
                "episode_id": record.episode_id,
                "seed": int(record.seed),
                "source_hash": record.source_hash,
                "evidence_id": record.evidence_id,
                "reward_shaper_valid": record.reward_shaper_valid,
                "critic_lifecycle_valid": record.critic_lifecycle_valid,
                "judge_model": record.judge_model,
                "alignment_scope": STAGE_BASELINE_ALIGNMENT_SCOPE,
                "alignment_key": record.alignment_key,
                "state_encoder_id": record.state_encoder_id,
                "value_checkpoint_id": record.value_checkpoint_id,
                "value_alignment_role": record.value_alignment_role,
            }
            metadata[str(path)] = provenance
            metadata[str(path.resolve())] = provenance
        return paths, metadata, records

    def latest_snapshot(self) -> Optional[Path]:
        candidates: List[Tuple[float, str, Path]] = []
        pointer = self.contract_root / "latest.json"
        if pointer.exists():
            try:
                with open(pointer, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                path = self.contract_root / str(
                    payload.get("snapshot_path") or ""
                )
                if path.is_file():
                    candidates.append(
                        (
                            self._snapshot_timestamp(
                                payload.get("updated_at"), pointer
                            ),
                            str(payload.get("baseline_id") or path.parent.name),
                            path,
                        )
                    )
            except (OSError, UnicodeError, json.JSONDecodeError, TypeError):
                pass

        # ``latest.json`` is a mutable convenience alias. Some network filesystems
        # mounts reject rename-over-existing with EPERM, so a successfully
        # published immutable snapshot may be newer than this pointer. Scan
        # formal manifests to recover that orphan without rerunning clean
        # calibration or deleting evidence.
        if self.snapshot_root.exists():
            for manifest_path in self.snapshot_root.glob("*/manifest.json"):
                try:
                    with open(manifest_path, "r", encoding="utf-8") as handle:
                        manifest = json.load(handle)
                    snapshot_path = self.contract_root / str(
                        manifest.get("snapshot_path") or ""
                    )
                    if not snapshot_path.is_file():
                        continue
                    candidates.append(
                        (
                            self._snapshot_timestamp(
                                manifest.get("published_at"), manifest_path
                            ),
                            str(
                                manifest.get("baseline_id")
                                or snapshot_path.parent.name
                            ),
                            snapshot_path,
                        )
                    )
                except (
                    OSError,
                    UnicodeError,
                    json.JSONDecodeError,
                    TypeError,
                ):
                    continue
        if not candidates:
            return None
        return max(candidates, key=lambda item: (item[0], item[1]))[2]

    @staticmethod
    def _snapshot_timestamp(value: Any, fallback_path: Path) -> float:
        text = str(value or "").strip()
        if text:
            try:
                return datetime.fromisoformat(
                    text.replace("Z", "+00:00")
                ).timestamp()
            except (TypeError, ValueError, OverflowError):
                pass
        try:
            return float(fallback_path.stat().st_mtime)
        except OSError:
            return 0.0

    def archive_legacy_diagnostic(
        self,
        source_path: Path,
        *,
        reason: str,
    ) -> Optional[Path]:
        """Preserve an incompatible pre-evidence baseline without promoting it."""

        source = Path(source_path)
        if not source.is_file():
            return None
        source_hash = _sha256_file(source)
        target_dir = self.root / "legacy_diagnostic" / source_hash[:24]
        target = target_dir / source.name
        if not target.exists():
            target_dir.mkdir(parents=True, exist_ok=True)
            temporary = _temporary_path(target)
            try:
                shutil.copy2(source, temporary)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
            _atomic_json(
                target_dir / "manifest.json",
                {
                    "source_path": str(source),
                    "source_hash": source_hash,
                    "reason": str(reason),
                    "formal_valid": False,
                    "archived_at": _utc_now(),
                },
            )
        return target

    def publish_snapshot(
        self,
        source_path: Path,
        *,
        baseline_id: str,
        metadata: Mapping[str, Any],
        update_latest: bool = True,
    ) -> Path:
        self.ensure_contract()
        with self._mutation_lock():
            return self._publish_snapshot_locked(
                source_path,
                baseline_id=baseline_id,
                metadata=metadata,
                update_latest=update_latest,
            )

    def _publish_snapshot_locked(
        self,
        source_path: Path,
        *,
        baseline_id: str,
        metadata: Mapping[str, Any],
        update_latest: bool,
    ) -> Path:
        snapshot_dir = self.snapshot_root / str(baseline_id)
        snapshot_path = snapshot_dir / "stage_baseline.json"
        if snapshot_path.exists():
            if _sha256_file(snapshot_path) != _sha256_file(Path(source_path)):
                raise ValueError(f"StageBaseline snapshot id collision: {baseline_id}")
        else:
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            temporary = _temporary_path(snapshot_path)
            try:
                shutil.copy2(source_path, temporary)
                temporary.replace(snapshot_path)
            finally:
                temporary.unlink(missing_ok=True)
        snapshot_metadata = {
            **dict(metadata),
            "baseline_id": str(baseline_id),
            "snapshot_path": str(snapshot_path.relative_to(self.contract_root)),
            "published_at": _utc_now(),
            "latest_eligible": bool(update_latest),
        }
        manifest_path = snapshot_dir / "manifest.json"
        if not manifest_path.exists():
            _atomic_json(manifest_path, snapshot_metadata)
        if update_latest:
            try:
                _atomic_json(
                    self.contract_root / "latest.json",
                    {
                        "contract_id": self.contract_id,
                        "baseline_id": str(baseline_id),
                        "snapshot_path": str(
                            snapshot_path.relative_to(self.contract_root)
                        ),
                        "updated_at": _utc_now(),
                    },
                )
            except OSError as exc:
                # The immutable snapshot and its manifest are authoritative.
                # On network filesystems, rename-over-existing may raise EPERM even after
                # both files have been durably written. Do not invalidate the
                # current evaluation merely because its mutable alias is stale.
                logger.warning(
                    "Published StageBaseline snapshot %s, but could not update "
                    "mutable latest.json alias (%s). Future loads will recover "
                    "the newest formal snapshot from manifests.",
                    snapshot_path,
                    exc,
                )
        return snapshot_path


__all__ = [
    "STAGE_BASELINE_EVIDENCE_SCHEMA_VERSION",
    "StageBaselineEvidence",
    "StageBaselineEvidenceStore",
]
