from __future__ import annotations

import argparse
import json
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import Blip2QFormerConfig

from models.finetune import FinetuneModel
from models.pretrain import PretrainModel
from utils.data import FeatureDataset
from utils.metrics import metrics, selection_score


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def model_config(
    sequence_length: int,
    output_hidden_states: bool,
    eeg_input_dim: int,
    eog_input_dim: int,
) -> Blip2QFormerConfig:
    return Blip2QFormerConfig(
        if_DDP=False,
        seq_length=sequence_length,
        eeg_input_dim=eeg_input_dim,
        eye_input_dim=eog_input_dim,
        chunk_size_feed_forward=0,
        attention_probs_dropout_prob=0.1,
        output_hidden_states=output_hidden_states,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        hidden_size=64,
        intermediate_size=256,
        num_attention_heads=4,
        num_hidden_layers=4,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        model_type="blip_2_qformer",
    )


def make_loader(dataset: FeatureDataset, indices: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    subset = torch.utils.data.Subset(dataset, indices.tolist())
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)


def pretrain(
    dataset: FeatureDataset,
    train_indices: np.ndarray,
    args: argparse.Namespace,
    fold_dir: Path,
    fold_prefix: str = "",
) -> dict[str, torch.Tensor]:
    config = model_config(
        args.sequence_length,
        output_hidden_states=True,
        eeg_input_dim=dataset.eeg.shape[-1],
        eog_input_dim=dataset.eog.shape[-1],
    )
    model = PretrainModel(
        config,
        model_device=torch.device(args.device),
        seq_length=args.sequence_length,
        eeg_input_dim=dataset.eeg.shape[-1],
        eye_input_dim=dataset.eog.shape[-1],
    ).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.pretrain_lr, weight_decay=args.weight_decay
    )
    loader = make_loader(dataset, train_indices, args.batch_size, shuffle=True)
    model.train()
    for epoch in range(args.pretrain_epochs):
        totals = np.zeros(4, dtype=np.float64)
        count = 0
        for eeg, eog, _ in loader:
            if len(eeg) < 2:
                continue
            eeg, eog = eeg.to(args.device, non_blocking=True), eog.to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            losses = model(eeg, eog)
            losses[0].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            with torch.no_grad():
                model.temp.clamp_(0.01, 0.5)
            totals += np.asarray([float(loss.detach()) for loss in losses]) * len(eeg)
            count += len(eeg)
        values = totals / max(count, 1)
        print(
            f"pretrain epoch={epoch + 1:03d}/{args.pretrain_epochs} "
            f"loss={values[0]:.4f} con={values[1]:.4f} mat={values[2]:.4f} gen={values[3]:.4f}",
            flush=True,
        )
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    if args.save_checkpoints:
        torch.save({"model": state}, fold_dir / f"{fold_prefix}pretrain.pt")
    del model
    torch.cuda.empty_cache()
    return state


def load_pretrained_weights(model: FinetuneModel, state: dict[str, torch.Tensor]) -> tuple[int, int]:
    target = model.state_dict()
    loaded = 0
    backbone_loaded = 0
    for key in list(target):
        candidates = [key]
        if key == "cls_fusion.weight":
            candidates.insert(0, "itm_fusion.weight")
        for source_key in candidates:
            if source_key in state and state[source_key].shape == target[key].shape:
                target[key] = state[source_key]
                loaded += 1
                backbone_loaded += int(key.startswith("backbone."))
                break
    model.load_state_dict(target)
    if backbone_loaded == 0:
        raise RuntimeError("No pretrained encoder weights were transferred")
    return loaded, backbone_loaded


@torch.no_grad()
def predict(model: FinetuneModel, loader: DataLoader, mode: str, device: str):
    model.eval()
    all_y, all_prob = [], []
    for eeg, eog, y in loader:
        eeg, eog = eeg.to(device, non_blocking=True), eog.to(device, non_blocking=True)
        if mode == "multi":
            logits = model(eeg=eeg, eye=eog)
        elif mode == "eeg":
            logits = model(eeg=eeg, eye=None)
        else:
            logits = model(eeg=None, eye=eog)
        all_y.append(y.numpy())
        all_prob.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(all_y), np.concatenate(all_prob)


def build_finetune_model(
    mode: str,
    dataset: FeatureDataset,
    args: argparse.Namespace,
) -> FinetuneModel:
    config = model_config(
        args.sequence_length,
        output_hidden_states=False,
        eeg_input_dim=dataset.eeg.shape[-1],
        eog_input_dim=dataset.eog.shape[-1],
    )
    model_mode = "eye" if mode == "eog" else mode
    model = FinetuneModel(
        config,
        model_device=torch.device(args.device),
        seq_length=args.sequence_length,
        eeg_input_dim=dataset.eeg.shape[-1],
        eye_input_dim=dataset.eog.shape[-1],
        n_class=2,
        mode=model_mode,
    ).to(args.device)
    return model


def new_finetune_model(
    mode: str,
    pretrained_state: dict[str, torch.Tensor],
    dataset: FeatureDataset,
    args: argparse.Namespace,
) -> FinetuneModel:
    model = build_finetune_model(mode, dataset, args)
    loaded, backbone_loaded = load_pretrained_weights(model, pretrained_state)
    print(
        f"{mode}: transferred {loaded} tensors "
        f"({backbone_loaded} backbone tensors)",
        flush=True,
    )
    return model


def finetune_class_weights(
    dataset: FeatureDataset, train_indices: np.ndarray, device: str
) -> torch.Tensor:
    train_labels = dataset.y[train_indices].numpy()
    counts = np.bincount(train_labels, minlength=2).astype(np.float32)
    weights = counts.sum() / np.maximum(counts * 2.0, 1.0)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_finetune_epoch(
    model: FinetuneModel,
    mode: str,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    class_weights: torch.Tensor,
    device: str,
) -> float:
    model.train()
    total_loss, count = 0.0, 0
    for eeg, eog, y in loader:
        eeg, eog, y = eeg.to(device), eog.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        if mode == "multi":
            logits = model(eeg=eeg, eye=eog)
        elif mode == "eeg":
            logits = model(eeg=eeg, eye=None)
        else:
            logits = model(eeg=None, eye=eog)
        loss = F.cross_entropy(logits, y, weight=class_weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += float(loss.detach()) * len(y)
        count += len(y)
    return total_loss / max(count, 1)


def select_finetune_epoch(
    mode: str,
    pretrained_state: dict[str, torch.Tensor],
    dataset: FeatureDataset,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    args: argparse.Namespace,
    log_prefix: str,
) -> tuple[int, float, dict[str, float], dict[str, torch.Tensor]]:
    model = new_finetune_model(mode, pretrained_state, dataset, args)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.finetune_lr, weight_decay=args.weight_decay
    )
    train_loader = make_loader(dataset, train_indices, args.batch_size, shuffle=True)
    val_loader = make_loader(dataset, val_indices, args.batch_size, shuffle=False)
    class_weights = finetune_class_weights(dataset, train_indices, args.device)

    best_score = -float("inf")
    best_state = None
    best_epoch = 0
    best_metrics: dict[str, float] = {}
    stale = 0
    for epoch in range(1, args.finetune_epochs + 1):
        loss = train_finetune_epoch(
            model, mode, train_loader, optimizer, class_weights, args.device
        )
        val_y, val_prob = predict(model, val_loader, mode, args.device)
        val_result = metrics(val_y, val_prob)
        score = selection_score(val_result, args.selection_metric)
        print(
            f"{log_prefix} select mode={mode} epoch={epoch:03d}/{args.finetune_epochs} "
            f"loss={loss:.4f} val={json.dumps(val_result, allow_nan=True)} "
            f"selection={score:.6f}",
            flush=True,
        )
        if score > best_score + 1e-5:
            best_score = score
            best_epoch = epoch
            best_metrics = val_result
            best_state = deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                print(f"{log_prefix} select early_stop mode={mode} epoch={epoch}", flush=True)
                break

    if best_state is None:
        raise RuntimeError(f"No fine-tuning checkpoint was produced for {mode}")
    del model
    torch.cuda.empty_cache()
    return best_epoch, best_score, best_metrics, best_state


def fit_fixed_epochs(
    mode: str,
    pretrained_state: dict[str, torch.Tensor],
    dataset: FeatureDataset,
    train_indices: np.ndarray,
    epochs: int,
    args: argparse.Namespace,
    log_prefix: str,
) -> dict[str, torch.Tensor]:
    model = new_finetune_model(mode, pretrained_state, dataset, args)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.finetune_lr, weight_decay=args.weight_decay
    )
    train_loader = make_loader(dataset, train_indices, args.batch_size, shuffle=True)
    class_weights = finetune_class_weights(dataset, train_indices, args.device)
    for epoch in range(1, epochs + 1):
        loss = train_finetune_epoch(
            model, mode, train_loader, optimizer, class_weights, args.device
        )
        print(
            f"{log_prefix} refit mode={mode} epoch={epoch:03d}/{epochs} loss={loss:.4f}",
            flush=True,
        )
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    del model
    torch.cuda.empty_cache()
    return state


def evaluate_finetune_state(
    mode: str,
    state: dict[str, torch.Tensor],
    dataset: FeatureDataset,
    test_indices: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    model = build_finetune_model(mode, dataset, args)
    model.load_state_dict(state)
    test_loader = make_loader(dataset, test_indices, args.batch_size, shuffle=False)
    test_y, test_prob = predict(model, test_loader, mode, args.device)
    result = metrics(test_y, test_prob)
    del model
    torch.cuda.empty_cache()
    return result, test_y, test_prob
