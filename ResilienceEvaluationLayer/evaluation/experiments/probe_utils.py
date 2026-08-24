"""
Lightweight linear probes for StateEncoder and ValueNetwork analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ProbeResult:
    task_type: str
    epochs: int
    input_dim: int
    output_dim: int
    metrics: Dict[str, float]
    baseline_metrics: Dict[str, float]


def _split_indices(num_rows: int, seed: int, train_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(num_rows, dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    split = max(1, min(num_rows - 1, int(round(num_rows * train_ratio)))) if num_rows > 1 else 1
    return indices[:split], indices[split:]


def _standardize(
    train_x: np.ndarray,
    eval_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (train_x - mean) / std, (eval_x - mean) / std


def _majority_baseline(y_train: np.ndarray, y_eval: np.ndarray, task_type: str) -> Dict[str, float]:
    if y_eval.size == 0:
        return {}
    if task_type == "regression":
        pred = np.full_like(y_eval, fill_value=float(y_train.mean()), dtype=np.float32)
        mae = float(np.mean(np.abs(pred - y_eval)))
        rmse = float(np.sqrt(np.mean((pred - y_eval) ** 2)))
        return {"mae": mae, "rmse": rmse}
    if task_type == "multilabel":
        probs = (y_train.mean(axis=0, keepdims=True) >= 0.5).astype(np.float32)
        pred = np.repeat(probs, y_eval.shape[0], axis=0)
        accuracy = float(np.mean((pred == y_eval).astype(np.float32)))
        exact = float(np.mean(np.all(pred == y_eval, axis=1).astype(np.float32)))
        return {"label_accuracy": accuracy, "exact_match": exact}
    if task_type == "binary":
        label = 1.0 if float(y_train.mean()) >= 0.5 else 0.0
        pred = np.full_like(y_eval, fill_value=label, dtype=np.float32)
        return {"accuracy": float(np.mean((pred == y_eval).astype(np.float32)))}
    # multiclass
    bincount = np.bincount(y_train.astype(np.int64))
    label = int(np.argmax(bincount)) if bincount.size else 0
    pred = np.full_like(y_eval, fill_value=label, dtype=np.int64)
    return {"accuracy": float(np.mean((pred == y_eval).astype(np.float32)))}


def _evaluate_predictions(
    logits_or_values: torch.Tensor,
    targets: torch.Tensor,
    task_type: str,
) -> Dict[str, float]:
    if logits_or_values.numel() == 0 or targets.numel() == 0:
        return {}
    if task_type == "regression":
        pred = logits_or_values.squeeze(-1)
        mae = torch.mean(torch.abs(pred - targets)).item()
        rmse = torch.sqrt(torch.mean((pred - targets) ** 2)).item()
        return {"mae": float(mae), "rmse": float(rmse)}
    if task_type == "binary":
        pred = (torch.sigmoid(logits_or_values.squeeze(-1)) >= 0.5).float()
        accuracy = (pred == targets).float().mean().item()
        return {"accuracy": float(accuracy)}
    if task_type == "multilabel":
        pred = (torch.sigmoid(logits_or_values) >= 0.5).float()
        label_accuracy = (pred == targets).float().mean().item()
        exact_match = torch.all(pred == targets, dim=1).float().mean().item()
        return {"label_accuracy": float(label_accuracy), "exact_match": float(exact_match)}
    pred = torch.argmax(logits_or_values, dim=1)
    accuracy = (pred == targets.long()).float().mean().item()
    return {"accuracy": float(accuracy)}


def train_linear_probe(
    features: np.ndarray,
    targets: np.ndarray,
    task_type: str,
    *,
    seed: int = 0,
    epochs: int = 200,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
    train_ratio: float = 0.8,
    device: str = "cpu",
) -> ProbeResult:
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(targets)
    if x.ndim != 2:
        raise ValueError("features must have shape [N, D].")
    if x.shape[0] != y.shape[0]:
        raise ValueError("features and targets must have the same number of rows.")
    if x.shape[0] < 2:
        raise ValueError("Need at least two rows to fit a probe.")
    if x.shape[1] == 0:
        return ProbeResult(
            task_type=task_type,
            epochs=0,
            input_dim=0,
            output_dim=0,
            metrics={},
            baseline_metrics={},
        )

    train_idx, eval_idx = _split_indices(x.shape[0], seed, train_ratio)
    if eval_idx.size == 0:
        eval_idx = train_idx[-1:]
        train_idx = train_idx[:-1]

    train_x, eval_x = x[train_idx], x[eval_idx]
    train_x, eval_x = _standardize(train_x, eval_x)

    output_dim = 1
    if task_type == "multiclass":
        output_dim = int(np.max(y) + 1)
    elif task_type == "multilabel":
        output_dim = int(y.shape[1])

    model = nn.Linear(train_x.shape[1], output_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_x_t = torch.tensor(train_x, dtype=torch.float32, device=device)
    eval_x_t = torch.tensor(eval_x, dtype=torch.float32, device=device)

    if task_type == "regression":
        train_y_t = torch.tensor(y[train_idx], dtype=torch.float32, device=device).view(-1, 1)
        eval_y_t = torch.tensor(y[eval_idx], dtype=torch.float32, device=device)
        baseline_metrics = _majority_baseline(y[train_idx].astype(np.float32), y[eval_idx].astype(np.float32), task_type)
        loss_fn = lambda pred, target: F.mse_loss(pred, target)
    elif task_type == "binary":
        train_y_t = torch.tensor(y[train_idx], dtype=torch.float32, device=device).view(-1, 1)
        eval_y_t = torch.tensor(y[eval_idx], dtype=torch.float32, device=device)
        baseline_metrics = _majority_baseline(y[train_idx].astype(np.float32), y[eval_idx].astype(np.float32), task_type)
        loss_fn = lambda pred, target: F.binary_cross_entropy_with_logits(pred, target)
    elif task_type == "multilabel":
        train_y_t = torch.tensor(y[train_idx], dtype=torch.float32, device=device)
        eval_y_t = torch.tensor(y[eval_idx], dtype=torch.float32, device=device)
        baseline_metrics = _majority_baseline(y[train_idx].astype(np.float32), y[eval_idx].astype(np.float32), task_type)
        loss_fn = lambda pred, target: F.binary_cross_entropy_with_logits(pred, target)
    else:
        train_y_t = torch.tensor(y[train_idx], dtype=torch.long, device=device)
        eval_y_t = torch.tensor(y[eval_idx], dtype=torch.long, device=device)
        baseline_metrics = _majority_baseline(y[train_idx].astype(np.int64), y[eval_idx].astype(np.int64), task_type)
        loss_fn = lambda pred, target: F.cross_entropy(pred, target)

    model.train()
    for _ in range(int(epochs)):
        optimizer.zero_grad()
        pred = model(train_x_t)
        loss = loss_fn(pred, train_y_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        eval_pred = model(eval_x_t)
    metrics = _evaluate_predictions(eval_pred, eval_y_t, task_type)
    return ProbeResult(
        task_type=task_type,
        epochs=int(epochs),
        input_dim=int(x.shape[1]),
        output_dim=int(output_dim),
        metrics=metrics,
        baseline_metrics=baseline_metrics,
    )


def compare_feature_sets(
    feature_sets: Mapping[str, np.ndarray],
    targets: np.ndarray,
    task_type: str,
    *,
    seed: int = 0,
    epochs: int = 200,
    device: str = "cpu",
) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    for name, features in feature_sets.items():
        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2 or features.shape[0] != len(targets):
            results[name] = {
                "task_type": task_type,
                "epochs": 0,
                "input_dim": int(features.shape[1]) if features.ndim == 2 else 0,
                "output_dim": 0,
                "metrics": {},
                "baseline_metrics": {},
            }
            continue
        probe_result = train_linear_probe(
            features=features,
            targets=targets,
            task_type=task_type,
            seed=seed,
            epochs=epochs,
            device=device,
        )
        results[name] = {
            "task_type": probe_result.task_type,
            "epochs": probe_result.epochs,
            "input_dim": probe_result.input_dim,
            "output_dim": probe_result.output_dim,
            "metrics": probe_result.metrics,
            "baseline_metrics": probe_result.baseline_metrics,
        }
    return results

