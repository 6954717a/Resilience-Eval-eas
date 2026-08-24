"""
LLM review helpers for SE5.5 and VN7.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from habitat_llm.evaluation.experiments.common import ensure_dir, write_json, write_jsonl


STATE_REVIEW_SYSTEM_PROMPT = """You are auditing representation fidelity for a task-execution system.
Return strict JSON with keys:
- preserved_information: list[str]
- confused_information: list[str]
- omitted_information: list[str]
- omission_harms_task_judgement: bool
- summary: str
Only mention information that is directly supported by the review packet.
"""


VALUE_REVIEW_SYSTEM_PROMPT = """You are auditing whether a value estimate matches task-execution potential.
Return strict JSON with keys:
- preferred_state: "a" | "b" | "tie"
- rationale: list[str]
- future_success_cues: list[str]
- misleading_cues: list[str]
Base the judgement only on the supplied packet.
"""


def build_state_review_packet(
    *,
    review_id: str,
    source_summary: Dict[str, Any],
    record: Dict[str, Any],
    nearest_neighbors: List[Dict[str, Any]],
    probe_predictions: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "review_id": review_id,
        "type": "state_fidelity",
        "source_summary": source_summary,
        "state_encoder_text": {
            "text_desc": record.get("text_desc"),
            "text_segments": record.get("text_segments", []),
        },
        "state_encoder_numerical": {
            "feature_names": record.get("feature_names", []),
            "feature_categories": record.get("feature_categories", []),
            "numerical_features": record.get("numerical_features", []),
            "category_presence": record.get("category_presence", {}),
        },
        "compressed_neighbors": nearest_neighbors,
        "probe_predictions": probe_predictions,
    }


def build_value_review_packet(
    *,
    review_id: str,
    record_a: Dict[str, Any],
    record_b: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "review_id": review_id,
        "type": "value_pairwise",
        "state_a": {
            "task_family": record_a.get("task_family"),
            "step_count": record_a.get("step_count"),
            "task_percent_complete": record_a.get("task_percent_complete"),
            "proposition_satisfied_fraction": record_a.get("proposition_satisfied_fraction"),
            "predicted_value": record_a.get("predicted_value"),
            "return_target": record_a.get("return_target"),
            "text_desc": record_a.get("text_desc"),
        },
        "state_b": {
            "task_family": record_b.get("task_family"),
            "step_count": record_b.get("step_count"),
            "task_percent_complete": record_b.get("task_percent_complete"),
            "proposition_satisfied_fraction": record_b.get("proposition_satisfied_fraction"),
            "predicted_value": record_b.get("predicted_value"),
            "return_target": record_b.get("return_target"),
            "text_desc": record_b.get("text_desc"),
        },
    }


def save_review_packets(
    output_dir: str | Path,
    review_name: str,
    packets: Iterable[Dict[str, Any]],
) -> Dict[str, str]:
    output_dir = ensure_dir(output_dir)
    packets = list(packets)
    json_path = write_json(output_dir / f"{review_name}.json", {"packets": packets})
    jsonl_path = write_jsonl(output_dir / f"{review_name}.jsonl", packets)
    return {"json": str(json_path), "jsonl": str(jsonl_path)}


def maybe_run_openai_reviews(
    packets: List[Dict[str, Any]],
    *,
    model: Optional[str],
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if not model:
        return []
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key or "dummy")
    outputs: List[Dict[str, Any]] = []
    for packet in packets:
        system_prompt = (
            STATE_REVIEW_SYSTEM_PROMPT
            if packet.get("type") == "state_fidelity"
            else VALUE_REVIEW_SYSTEM_PROMPT
        )
        completion = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(packet, ensure_ascii=False)},
            ],
        )
        content = completion.choices[0].message.content or "{}"
        outputs.append(
            {
                "review_id": packet.get("review_id"),
                "type": packet.get("type"),
                "model": model,
                "raw_json": json.loads(content),
            }
        )
    return outputs

