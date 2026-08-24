"""Deterministic Critic component checkpoints and parent-side value training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from habitat_llm.evaluation.critic_export_contract import load_full_stability_export


CRITIC_COMPONENT_SCHEMA_VERSION = 1


def preflight_reward_shaper(critic_config: Mapping[str, Any]) -> Dict[str, Any]:
    """Issue one real structured Judge request before formal collection."""

    config = dict(critic_config)
    if not bool(config.get("use_llm_shaping", False)):
        return {"valid": True, "reason": "llm_shaping_disabled"}
    try:
        from openai import OpenAI
        import os

        from habitat_llm.evaluation.llm_evaluator import LLMRewardShaper

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return {"valid": False, "reason": "OPENAI_API_KEY_missing"}
        client = OpenAI(
            base_url=str(config.get("llm_base_url") or "") or None,
            api_key=api_key,
        )
        shaper = LLMRewardShaper(client, config)
        valid, reason, result = shaper.preflight()
        return {"valid": bool(valid), "reason": str(reason), "result": result}
    except Exception as exc:
        return {
            "valid": False,
            "reason": f"reward_shaper_preflight_failed:{type(exc).__name__}",
        }


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


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_dict_digest(state_dict: Mapping[str, Any], identity: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(
        json.dumps(
            identity,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    )
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _load_payload(path: Path) -> Dict[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError(f"Critic checkpoint payload is not a mapping: {path}")
    return dict(payload)


def checkpoint_component_id(path: Optional[str], component: str) -> str:
    if not path:
        return ""
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        return ""
    payload = _load_payload(checkpoint_path)
    metadata = payload.get("metadata") or {}
    metadata_key = (
        "value_checkpoint_id" if component == "value_network" else f"{component}_id"
    )
    recorded = str(metadata.get(metadata_key) or "")
    if recorded:
        return recorded
    state_dict = payload.get(component)
    if not isinstance(state_dict, Mapping):
        return ""
    return _state_dict_digest(state_dict, {"component": component})


def ensure_bootstrap_critic_components(
    critic_config: Mapping[str, Any],
    *,
    model_root: Path,
    judge_model: str,
) -> Dict[str, str]:
    """Create deterministic shared-encoder and Judge bootstrap-value weights."""

    import torch

    from habitat_llm.evaluation.critic import A2CCritic
    from habitat_llm.evaluation.networks.state_encoder import (
        STATE_ENCODER_FEATURE_SCHEMA_VERSION,
    )

    config = dict(critic_config)
    initialization_seed = int(config.get("initialization_seed", 47668090))
    state_identity = {
        "schema_version": CRITIC_COMPONENT_SCHEMA_VERSION,
        "initialization_seed": initialization_seed,
        "state_encoder": config.get("state_encoder", {}),
        "feature_schema_version": STATE_ENCODER_FEATURE_SCHEMA_VERSION,
    }
    value_identity = {
        "schema_version": CRITIC_COMPONENT_SCHEMA_VERSION,
        "initialization_seed": initialization_seed,
        "judge_model": str(judge_model),
        "state_dim": (config.get("state_encoder", {}) or {}).get("dims", {}).get(
            "output", 128
        ),
        "value_hidden_dims": config.get("value_hidden_dims", [256, 128, 64]),
        "dropout_rate": config.get("dropout_rate", 0.1),
    }
    state_key = _canonical_digest(state_identity)[:20]
    value_key = _canonical_digest(value_identity)[:20]
    state_path = Path(model_root) / "state_encoder" / state_key / "state_encoder.pt"
    value_path = (
        Path(model_root)
        / "value_network"
        / str(judge_model).replace("/", "-")
        / value_key
        / "bootstrap_value.pt"
    )

    if not state_path.exists() or not value_path.exists():
        bootstrap_config = dict(config)
        bootstrap_config.update(
            {
                "use_llm_shaping": False,
                "use_llm_offline": False,
                "save_checkpoints": False,
                "device": "cpu",
                "update_value_network": False,
                "freeze_state_encoder": True,
                "state_encoder_checkpoint": "",
                "value_checkpoint": "",
            }
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(initialization_seed)
            critic = A2CCritic(bootstrap_config, env_interface=None)
        if not state_path.exists():
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_dict = critic.state_encoder.state_dict()
            state_encoder_id = _state_dict_digest(state_dict, state_identity)
            torch.save(
                {
                    "schema_version": CRITIC_COMPONENT_SCHEMA_VERSION,
                    "state_encoder": state_dict,
                    "metadata": {
                        "state_encoder_id": state_encoder_id,
                        "identity": state_identity,
                    },
                },
                state_path,
            )
        if not value_path.exists():
            value_path.parent.mkdir(parents=True, exist_ok=True)
            value_dict = critic.value_network.state_dict()
            value_checkpoint_id = _state_dict_digest(value_dict, value_identity)
            torch.save(
                {
                    "schema_version": CRITIC_COMPONENT_SCHEMA_VERSION,
                    "value_network": value_dict,
                    "metadata": {
                        "value_checkpoint_id": value_checkpoint_id,
                        "identity": value_identity,
                        "trained": False,
                    },
                },
                value_path,
            )

    return {
        "state_encoder_checkpoint": str(state_path),
        "state_encoder_id": checkpoint_component_id(
            str(state_path), "state_encoder"
        ),
        "bootstrap_value_checkpoint": str(value_path),
        "bootstrap_value_checkpoint_id": checkpoint_component_id(
            str(value_path), "value_network"
        ),
    }


def train_value_checkpoint_from_exports(
    critic_config: Mapping[str, Any],
    export_paths: Sequence[Path],
    *,
    bootstrap_checkpoint: Path,
    output_path: Path,
    judge_model: str,
    state_encoder_id: str,
) -> Dict[str, Any]:
    """Fit one Judge value head from isolated-worker transition exports."""

    import torch

    from habitat_llm.evaluation.networks import ValueNetwork

    states = []
    targets = []
    used_exports = []
    excluded = []
    for raw_path in export_paths:
        path = Path(raw_path)
        export = load_full_stability_export(path)
        if not export.valid:
            excluded.append({"path": str(path), "reason": export.missing_reason})
            continue
        added = 0
        for record in export.records:
            state_vector = record.get("state_vector")
            target = record.get("return_target")
            if not isinstance(state_vector, list) or target in (None, ""):
                continue
            try:
                vector = np.asarray(state_vector, dtype=np.float32)
                numeric_target = float(target)
            except (TypeError, ValueError):
                continue
            if vector.ndim != 1 or not np.all(np.isfinite(vector)):
                continue
            if not np.isfinite(numeric_target):
                continue
            states.append(vector)
            targets.append(numeric_target)
            added += 1
        if added:
            used_exports.append(str(path))
    if not states:
        return {
            "valid": False,
            "reason": "no_value_training_transitions",
            "used_exports": used_exports,
            "excluded": excluded,
        }

    config = dict(critic_config)
    lifecycle = dict(config.get("lifecycle", {}) or {})
    device = str(config.get("device", "cpu"))
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    seed = int(config.get("initialization_seed", 47668090))
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    state_dim = int(states[0].shape[0])
    model = ValueNetwork(
        state_dim=state_dim,
        hidden_dims=config.get("value_hidden_dims", [256, 128, 64]),
        dropout_rate=float(config.get("dropout_rate", 0.1)),
        device=device,
    )
    bootstrap_payload = _load_payload(Path(bootstrap_checkpoint))
    model.load_state_dict(bootstrap_payload["value_network"])
    model.train()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(config.get("value_lr", 1.0e-3))
    )
    loss_fn = torch.nn.MSELoss()
    x = torch.as_tensor(np.stack(states), dtype=torch.float32, device=device)
    y = torch.as_tensor(np.asarray(targets), dtype=torch.float32, device=device)
    epochs = int(lifecycle.get("value_training_epochs", 20))
    batch_size = max(1, int(lifecycle.get("value_training_batch_size", 256)))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    final_loss = float("nan")
    for _ in range(max(1, epochs)):
        order = torch.randperm(len(x), generator=generator)
        for start in range(0, len(x), batch_size):
            indices = order[start : start + batch_size].to(device)
            predicted = model(x[indices])
            loss = loss_fn(predicted, y[indices])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            final_loss = float(loss.detach().cpu().item())
    model.eval()
    identity = {
        "schema_version": CRITIC_COMPONENT_SCHEMA_VERSION,
        "judge_model": str(judge_model),
        "state_encoder_id": str(state_encoder_id),
        "source_hashes": sorted(_file_digest(Path(path)) for path in used_exports),
        "training_samples": len(states),
        "epochs": max(1, epochs),
        "value_hidden_dims": config.get("value_hidden_dims", [256, 128, 64]),
    }
    state_dict = model.state_dict()
    value_checkpoint_id = _state_dict_digest(state_dict, identity)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": CRITIC_COMPONENT_SCHEMA_VERSION,
            "value_network": state_dict,
            "metadata": {
                "value_checkpoint_id": value_checkpoint_id,
                "identity": identity,
                "trained": True,
                "final_loss": final_loss,
            },
        },
        output_path,
    )
    return {
        "valid": True,
        "value_checkpoint": str(output_path),
        "value_checkpoint_id": value_checkpoint_id,
        "training_samples": len(states),
        "used_exports": used_exports,
        "excluded": excluded,
        "final_loss": final_loss,
    }


__all__ = [
    "CRITIC_COMPONENT_SCHEMA_VERSION",
    "checkpoint_component_id",
    "ensure_bootstrap_critic_components",
    "preflight_reward_shaper",
    "train_value_checkpoint_from_exports",
]
