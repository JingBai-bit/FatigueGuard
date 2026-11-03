"""Supervised fine-tuning and testing for FatigueGuard."""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from dataset.loader import FeatureData, FeatureDataset
from .utils import (
    class_weights,
    cpu_state,
    evaluate,
    inputs_for_mode,
    load_pretrained_weights,
    make_loader,
    new_model,
    seed_everything,
)


def train_epoch(
    model: torch.nn.Module,
    dataset: FeatureDataset,
    indices: np.ndarray,
    mode: str,
    optimizer: torch.optim.Optimizer,
    weights: torch.Tensor,
    args: argparse.Namespace,
    seed: int,
) -> float:
    model.train()
    total = 0.0
    loader = make_loader(dataset, indices, args.batch_size, True, seed)
    for eeg, eog, target in loader:
        target = target.to(args.device, non_blocking=True)
        logits = model(**inputs_for_mode(eeg, eog, mode, args.device))
        loss = F.cross_entropy(logits, target, weight=weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total += float(loss.detach()) * len(target)
    return total / max(len(indices), 1)


def select_finetune_epoch(
    data: FeatureData,
    dataset: FeatureDataset,
    pretrained: dict[str, torch.Tensor] | None,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    mode: str,
    args: argparse.Namespace,
    seed: int,
    prefix: str,
) -> tuple[int, dict[str, float], dict[str, torch.Tensor]]:
    """Fine-tune with validation-based checkpoint selection."""

    seed_everything(seed)
    model = new_model(data, args, mode=mode)
    load_pretrained_weights(model, pretrained)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.finetune_lr, weight_decay=args.weight_decay
    )
    weights = class_weights(dataset, train_indices, args.device)
    best_epoch = 0
    best_score = -float("inf")
    best_metrics: dict[str, float] = {}
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    for epoch in range(1, args.finetune_epochs + 1):
        loss = train_epoch(
            model, dataset, train_indices, mode, optimizer, weights, args, seed + epoch
        )
        metrics, _, _ = evaluate(model, dataset, validation_indices, mode, args)
        score = metrics[args.selection_metric]
        if not np.isfinite(score):
            score = -float("inf")
        print(
            f"{prefix} stage=finetune mode={mode} "
            f"epoch={epoch:03d}/{args.finetune_epochs} loss={loss:.5f} "
            f"validation={json.dumps(metrics, allow_nan=True)}",
            flush=True,
        )
        if score > best_score + 1e-8:
            best_epoch = epoch
            best_score = score
            best_metrics = metrics
            best_state = cpu_state(model)
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError(f"No valid validation checkpoint: {prefix} mode={mode}")
    return best_epoch, best_metrics, best_state


def fit_fixed_epochs(
    data: FeatureData,
    dataset: FeatureDataset,
    pretrained: dict[str, torch.Tensor] | None,
    indices: np.ndarray,
    epochs: int,
    mode: str,
    args: argparse.Namespace,
    seed: int,
    prefix: str,
) -> dict[str, torch.Tensor]:
    """Refit on all non-test data using the selected epoch count."""

    seed_everything(seed)
    model = new_model(data, args, mode=mode)
    load_pretrained_weights(model, pretrained)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.finetune_lr, weight_decay=args.weight_decay
    )
    weights = class_weights(dataset, indices, args.device)
    for epoch in range(1, epochs + 1):
        loss = train_epoch(
            model, dataset, indices, mode, optimizer, weights, args, seed + epoch
        )
        print(
            f"{prefix} stage=refit mode={mode} "
            f"epoch={epoch:03d}/{epochs} loss={loss:.5f}",
            flush=True,
        )
    return cpu_state(model)


def evaluate_state(
    data: FeatureData,
    dataset: FeatureDataset,
    state: dict[str, torch.Tensor],
    indices: np.ndarray,
    mode: str,
    args: argparse.Namespace,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    model = new_model(data, args, mode=mode)
    model.load_state_dict(state)
    return evaluate(model, dataset, indices, mode, args)
