"""Shared runtime utilities used by both FatigueGuard training stages."""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Subset

from dataset.loader import FeatureData, FeatureDataset
from model.fatigueguard import FatigueGuard
from model.pretrain import FatigueGuardPretrain
from model.config import FatigueGuardConfig


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cpu_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }


def new_model(
    data: FeatureData,
    args: argparse.Namespace,
    mode: str = "multi",
    pretraining: bool = False,
) -> torch.nn.Module:
    config = FatigueGuardConfig(
        hidden_size=args.hidden_size,
        intermediate_size=args.feedforward_size,
        num_attention_heads=args.attention_heads,
        num_hidden_layers=args.expert_layers + 1,
        hidden_dropout_prob=args.dropout,
        attention_probs_dropout_prob=args.dropout,
        disable_shared_expert=args.experiment == "no_shared_expert",
        disable_gate=args.experiment == "no_gate",
        disable_fusion_transformer=args.experiment == "no_fusion_transformer",
    )
    options = dict(
        config=config,
        device=torch.device(args.device),
        eeg_input_dim=data.eeg.shape[-1],
        eog_input_dim=data.eog.shape[-1],
        seq_length=data.sequence_length,
    )
    if pretraining:
        return FatigueGuardPretrain(
            **options,
            loss_weights={
                name: float(args.experiment != f"no_{name}")
                for name in ("con", "mat", "gen")
            },
        ).to(args.device)
    return FatigueGuard(
        **options,
        mode=mode,
        n_class=2,
    ).to(args.device)


def load_pretrained_weights(model: FatigueGuard, state: dict | None) -> None:
    """Transfer the encoder, input tokens, and matching gate; initialize the head."""
    if state is None:
        return
    target = model.state_dict()
    for key in target:
        if key.startswith("cls_head."):
            continue
        source = "itm_fusion.weight" if key == "cls_fusion.weight" else key
        if source not in state or state[source].shape != target[key].shape:
            raise ValueError(f"Missing or incompatible pretrained tensor: {source}")
        target[key] = state[source]
    model.load_state_dict(target)


def make_loader(
    dataset: FeatureDataset,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        Subset(dataset, indices.tolist()),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def inputs_for_mode(
    eeg: torch.Tensor, eog: torch.Tensor, mode: str, device: str
) -> dict[str, torch.Tensor | None]:
    return {
        "eeg": eeg.to(device, non_blocking=True) if mode != "eog" else None,
        "eog": eog.to(device, non_blocking=True) if mode != "eeg" else None,
    }


def class_weights(
    dataset: FeatureDataset, indices: np.ndarray, device: str
) -> torch.Tensor:
    labels = dataset.labels[indices].numpy()
    counts = np.bincount(labels, minlength=2).astype(np.float32)
    weights = counts.sum() / np.maximum(2.0 * counts, 1.0)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def compute_metrics(labels: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    prediction = (probability >= 0.5).astype(np.int64)
    result = {
        "acc": float(accuracy_score(labels, prediction)),
        "f1": float(f1_score(labels, prediction, zero_division=0)),
    }
    result["auroc"] = (
        float(roc_auc_score(labels, probability))
        if len(np.unique(labels)) == 2
        else float("nan")
    )
    return result


@torch.no_grad()
def evaluate(
    model: FatigueGuard,
    dataset: FeatureDataset,
    indices: np.ndarray,
    mode: str,
    args: argparse.Namespace,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    loader = make_loader(dataset, indices, args.batch_size, False, args.seed)
    for eeg, eog, target in loader:
        logits = model(**inputs_for_mode(eeg, eog, mode, args.device))
        probability = logits.softmax(dim=-1)[:, 1]
        labels.append(target.numpy())
        probabilities.append(probability.cpu().numpy())
    y = np.concatenate(labels)
    p = np.concatenate(probabilities)
    return compute_metrics(y, p), y, p
