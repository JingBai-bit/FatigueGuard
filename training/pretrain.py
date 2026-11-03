"""Multimodal representation pretraining for FatigueGuard."""

from __future__ import annotations

import argparse

import numpy as np
import torch

from dataset.loader import FeatureData, FeatureDataset
from .utils import cpu_state, make_loader, new_model, seed_everything


def pretrain(
    data: FeatureData,
    dataset: FeatureDataset,
    indices: np.ndarray,
    args: argparse.Namespace,
    seed: int,
    prefix: str,
) -> dict[str, torch.Tensor] | None:
    """Optimize contrastive, matching, and cross-modal generation tasks."""

    seed_everything(seed)
    if args.experiment == "scratch":
        return None
    if len(indices) < 2:
        raise ValueError("Pretraining requires at least two training sequences")
    model = new_model(data, args, pretraining=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.pretrain_lr, weight_decay=args.weight_decay
    )
    loader = make_loader(dataset, indices, args.batch_size, True, seed)
    for epoch in range(1, args.pretrain_epochs + 1):
        model.train()
        totals = np.zeros(4, dtype=np.float64)
        count = 0
        for eeg, eog, _ in loader:
            if len(eeg) < 2:
                continue
            eeg = eeg.to(args.device, non_blocking=True)
            eog = eog.to(args.device, non_blocking=True)
            loss, contrastive, matching, generation = model(eeg, eog)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            with torch.no_grad():
                model.temp.clamp_(0.01, 0.5)
            batch = len(eeg)
            totals += batch * np.asarray(
                [
                    float(loss.detach()),
                    float(contrastive.detach()),
                    float(matching.detach()),
                    float(generation.detach()),
                ]
            )
            count += batch
        values = totals / max(count, 1)
        print(
            f"{prefix} stage=pretrain epoch={epoch:03d}/{args.pretrain_epochs} "
            f"loss={values[0]:.5f} con={values[1]:.5f} "
            f"match={values[2]:.5f} gen={values[3]:.5f}",
            flush=True,
        )
    return cpu_state(model)
