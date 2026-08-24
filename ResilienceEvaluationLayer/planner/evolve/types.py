from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EvolutionLesson:
    lesson_id: str
    lesson_type: str
    scene_id: str
    task_signature: str
    graph_snapshot_hash: str
    trigger: str
    decision: str
    confidence: float = 0.0
    source: str = "system"
    evidence: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    failure_type: str = ""
    related_skills: List[str] = field(default_factory=list)
    memory_step: int = 0
    created_at: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvolutionLesson":
        return cls(
            lesson_id=str(data.get("lesson_id", "")),
            lesson_type=str(data.get("lesson_type", "")),
            scene_id=str(data.get("scene_id", "")),
            task_signature=str(data.get("task_signature", "")),
            graph_snapshot_hash=str(data.get("graph_snapshot_hash", "")),
            trigger=str(data.get("trigger", "")),
            decision=str(data.get("decision", "")),
            confidence=float(data.get("confidence", 0.0) or 0.0),
            source=str(data.get("source", "system")),
            evidence=dict(data.get("evidence", {}) or {}),
            tags=list(data.get("tags", []) or []),
            failure_type=str(data.get("failure_type", "")),
            related_skills=list(data.get("related_skills", []) or []),
            memory_step=int(data.get("memory_step", 0) or 0),
            created_at=float(data.get("created_at", 0.0) or 0.0),
        )


@dataclass
class EvolutionPolicyOverlay:
    scene_id: str = ""
    task_signature: str = ""
    graph_snapshot_hash: str = ""
    current_task_guidance: List[str] = field(default_factory=list)
    candidate_risks: List[str] = field(default_factory=list)
    preferred_recovery: List[str] = field(default_factory=list)
    preferred_skill_order: List[str] = field(default_factory=list)
    lesson_refs: List[str] = field(default_factory=list)
    confidence: float = 0.0
    source: str = "memory"

    def is_empty(self) -> bool:
        return not any(
            [
                self.current_task_guidance,
                self.candidate_risks,
                self.preferred_recovery,
                self.preferred_skill_order,
            ]
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_prompt_context(self) -> str:
        lines: List[str] = []
        if self.current_task_guidance:
            lines.append("Current Task Guidance:")
            lines.extend(f"- {item}" for item in self.current_task_guidance[:4])
        if self.candidate_risks:
            lines.append("Candidate Risks:")
            lines.extend(f"- {item}" for item in self.candidate_risks[:4])
        if self.preferred_recovery:
            lines.append("Preferred Recovery:")
            lines.extend(f"- {item}" for item in self.preferred_recovery[:3])
        if self.preferred_skill_order:
            lines.append("Preferred Skill Order:")
            lines.append(", ".join(self.preferred_skill_order[:6]))
        return "\n".join(lines)


@dataclass
class EvolutionEpisodeRecord:
    record_id: str
    scene_id: str
    episode_id: str
    episode_filename: str
    run_id: int
    instruction: str
    task_signature: str
    graph_snapshot_hash_before: str = ""
    graph_snapshot_hash_after: str = ""
    world_model_before: Dict[str, Any] = field(default_factory=dict)
    world_model_after: Dict[str, Any] = field(default_factory=dict)
    action_history: List[Dict[str, Any]] = field(default_factory=list)
    response_history: List[Dict[str, Any]] = field(default_factory=list)
    skill_outcomes: List[Dict[str, Any]] = field(default_factory=list)
    critic_summary: Dict[str, Any] = field(default_factory=dict)
    adca_summary: Dict[str, Any] = field(default_factory=dict)
    rebound_summary: Dict[str, Any] = field(default_factory=dict)
    cyclevla_summary: Dict[str, Any] = field(default_factory=dict)
    failure_type: str = ""
    recovery_mode: str = ""
    success: bool = False
    human_feedback: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvolutionEpisodeRecord":
        return cls(
            record_id=str(data.get("record_id", "")),
            scene_id=str(data.get("scene_id", "")),
            episode_id=str(data.get("episode_id", "")),
            episode_filename=str(data.get("episode_filename", "")),
            run_id=int(data.get("run_id", 0) or 0),
            instruction=str(data.get("instruction", "")),
            task_signature=str(data.get("task_signature", "")),
            graph_snapshot_hash_before=str(data.get("graph_snapshot_hash_before", "")),
            graph_snapshot_hash_after=str(data.get("graph_snapshot_hash_after", "")),
            world_model_before=dict(data.get("world_model_before", {}) or {}),
            world_model_after=dict(data.get("world_model_after", {}) or {}),
            action_history=list(data.get("action_history", []) or []),
            response_history=list(data.get("response_history", []) or []),
            skill_outcomes=list(data.get("skill_outcomes", []) or []),
            critic_summary=dict(data.get("critic_summary", {}) or {}),
            adca_summary=dict(data.get("adca_summary", {}) or {}),
            rebound_summary=dict(data.get("rebound_summary", {}) or {}),
            cyclevla_summary=dict(data.get("cyclevla_summary", {}) or {}),
            failure_type=str(data.get("failure_type", "")),
            recovery_mode=str(data.get("recovery_mode", "")),
            success=bool(data.get("success", False)),
            human_feedback=list(data.get("human_feedback", []) or []),
            metadata=dict(data.get("metadata", {}) or {}),
        )
