#!/usr/bin/env python3
"""Train and evaluate FatigueGuard from a unified precomputed feature file."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from dataset.loader import (
    FeatureDataset,
    indices_for,
    load_feature_file,
    parse_folds,
    scale_features,
    temporal_fold_assignment,
)
from training.pretrain import pretrain
from training.finetune import evaluate_state, fit_fixed_epochs, select_finetune_epoch
from training.utils import compute_metrics, seed_everything
from model.config import EXPERIMENTS


MODES = ("multi", "eeg", "eog")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, required=True, help="Unified NPZ feature file"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="New or empty output directory (default: outputs/<timestamp>)",
    )
    parser.add_argument("--experiment", choices=EXPERIMENTS, default="full")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate feature arrays and temporal splits without training",
    )
    parser.add_argument(
        "--folds", default="all", help="all or comma-separated fold IDs"
    )
    parser.add_argument("--modalities", default="multi,eeg,eog")
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--validation-offset", type=int, default=2)
    parser.add_argument("--purge-boundary-windows", type=int, default=0)
    parser.add_argument("--pretrain-epochs", type=int, default=200)
    parser.add_argument("--finetune-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--pretrain-lr", type=float, default=1e-4)
    parser.add_argument("--finetune-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument(
        "--selection-metric", choices=("acc", "f1", "auroc"), default="acc"
    )
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--feedforward-size", type=int, default=256)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--expert-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--refit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Refit on all non-test folds",
    )
    parser.add_argument(
        "--save-checkpoints", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    if args.finetune_epochs < 1 or args.pretrain_epochs < (
        0 if args.experiment == "scratch" else 1
    ):
        parser.error(
            "Epoch counts must be positive (scratch permits --pretrain-epochs 0)"
        )
    if args.num_folds < 3 or args.validation_offset % args.num_folds == 0:
        parser.error(
            "Use at least 3 folds and a validation offset that differs from the test fold"
        )
    if args.purge_boundary_windows < 0 or args.batch_size < 2 or args.patience < 1:
        parser.error("Require purge >= 0, batch-size >= 2, and patience >= 1")
    if (
        min(
            args.hidden_size,
            args.feedforward_size,
            args.attention_heads,
            args.expert_layers,
        )
        < 1
    ):
        parser.error("Model dimensions and layer counts must be positive")
    if args.hidden_size % args.attention_heads or not 0 <= args.dropout < 1:
        parser.error(
            "hidden-size must be divisible by attention-heads; dropout must be in [0, 1)"
        )
    if not (args.pretrain_lr > 0 and args.finetune_lr > 0 and args.weight_decay >= 0):
        parser.error("Learning rates must be positive and weight-decay non-negative")
    modes = [value.strip() for value in args.modalities.split(",")]
    if (
        not modes
        or any(mode not in MODES for mode in modes)
        or len(set(modes)) != len(modes)
    ):
        parser.error(f"modalities must be a nonempty, unique selection from {MODES}")
    if args.output_dir is None:
        args.output_dir = (
            Path(__file__).resolve().parent
            / "outputs"
            / datetime.now().strftime("%Y%m%dT%H%M%S%f")
        )
    return args


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(
    fold_rows: list[dict],
    predictions: list[dict],
    modes: list[str],
    subjects: list[int],
) -> tuple[dict, list[dict]]:
    subject_rows: list[dict] = []
    reported: dict[str, dict] = {}
    for mode in modes:
        for subject in subjects:
            rows = [
                row
                for row in fold_rows
                if row["mode"] == mode and row["subject"] == subject
            ]
            if not rows:
                continue
            subject_rows.append(
                {
                    "subject": subject,
                    "mode": mode,
                    "n_folds": len(rows),
                    **{
                        metric: float(np.nanmean([row[metric] for row in rows]))
                        for metric in ("acc", "f1", "auroc")
                    },
                }
            )
        mode_subjects = [row for row in subject_rows if row["mode"] == mode]
        mode_predictions = [row for row in predictions if row["mode"] == mode]
        y = np.asarray([row["target"] for row in mode_predictions], dtype=np.int64)
        p = np.asarray([row["probability"] for row in mode_predictions], dtype=float)
        pooled = compute_metrics(y, p)
        reported[mode] = {}
        for metric in ("acc", "f1", "auroc"):
            values = np.asarray([row[metric] for row in mode_subjects], dtype=float)
            values = values[np.isfinite(values)]
            reported[mode][metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "n": int(len(values)),
            }
        reported[mode]["pooled_f1"] = float(pooled["f1"])
        reported[mode]["pooled_auroc"] = float(pooled["auroc"])
    return reported, subject_rows


def run(args: argparse.Namespace) -> None:
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu")
    seed_everything(args.seed)
    data = load_feature_file(args.data)
    assignment = temporal_fold_assignment(
        data, args.num_folds, args.purge_boundary_windows
    )
    subjects = sorted(np.unique(data.subjects).tolist())
    folds = parse_folds(args.folds, args.num_folds)
    modes = [value.strip() for value in args.modalities.split(",") if value.strip()]
    if any(mode not in MODES for mode in modes):
        raise ValueError(f"modalities must be selected from {MODES}")
    if data.eeg.shape[-1] < 2 or data.eog.shape[-1] < 2:
        raise ValueError(
            "Each modality needs at least two features for generation normalization"
        )
    for fold in folds:
        validation_fold = ((fold - 1 + args.validation_offset) % args.num_folds) + 1
        for subject in subjects:
            validation_indices = indices_for(
                data, assignment, subject, {validation_fold}
            )
            if (
                args.selection_metric == "auroc"
                and len(np.unique(data.labels[validation_indices])) < 2
            ):
                raise ValueError(
                    f"AUROC selection needs both validation classes: subject={subject} fold={validation_fold}; use --selection-metric acc"
                )
    print(
        f"Validated {len(data.labels)} sequences; {int((assignment < 0).sum())} boundary sequences excluded",
        flush=True,
    )
    if args.validate_only:
        return

    configuration = {
        **{
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "dataset_subjects": subjects,
        "selected_folds": folds,
        "feature_metadata": data.metadata,
        "feature_shapes": {"eeg": list(data.eeg.shape), "eog": list(data.eog.shape)},
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(configuration, indent=2, allow_nan=True) + "\n"
    )
    write_csv(
        args.output_dir / "fold_assignments.csv",
        [
            {
                "sample": index,
                "subject": int(data.subjects[index]),
                "session": int(data.sessions[index]),
                "sequence_start": int(data.sequence_start[index]),
                "fold": int(assignment[index]),
            }
            for index in range(len(data.labels))
        ],
        ["sample", "subject", "session", "sequence_start", "fold"],
    )

    print(
        f"data={args.data.resolve()} samples={len(data.labels)} subjects={subjects} "
        f"folds={folds} modalities={modes} refit={args.refit}",
        flush=True,
    )
    all_folds = set(range(1, args.num_folds + 1))
    fold_rows: list[dict] = []
    predictions: list[dict] = []

    for test_fold in folds:
        fold_seed = args.seed + test_fold * 100_000
        validation_fold = (
            (test_fold - 1 + args.validation_offset) % args.num_folds
        ) + 1
        train_folds = all_folds - {test_fold, validation_fold}
        non_test_folds = all_folds - {test_fold}
        selection_global = indices_for(data, assignment, subjects, train_folds)
        eeg, eog = scale_features(data, selection_global)
        selection_dataset = FeatureDataset(eeg, eog, data.labels)
        selection_pretrained = pretrain(
            data,
            selection_dataset,
            selection_global,
            args,
            fold_seed,
            f"fold={test_fold}",
        )

        selected: dict[tuple[int, str], tuple[int, dict, dict]] = {}
        for subject in subjects:
            train_indices = indices_for(data, assignment, subject, train_folds)
            validation_indices = indices_for(
                data, assignment, subject, {validation_fold}
            )
            for mode_index, mode in enumerate(modes):
                selected[(subject, mode)] = select_finetune_epoch(
                    data,
                    selection_dataset,
                    selection_pretrained,
                    train_indices,
                    validation_indices,
                    mode,
                    args,
                    fold_seed + subject * 100 + mode_index,
                    f"fold={test_fold} subject={subject}",
                )

        if args.refit:
            final_global = indices_for(data, assignment, subjects, non_test_folds)
            final_eeg, final_eog = scale_features(data, final_global)
            final_dataset = FeatureDataset(final_eeg, final_eog, data.labels)
            final_pretrained = pretrain(
                data,
                final_dataset,
                final_global,
                args,
                fold_seed + 50_000,
                f"fold={test_fold} final",
            )
        else:
            final_dataset = selection_dataset
            final_pretrained = selection_pretrained

        if args.save_checkpoints and final_pretrained is not None:
            torch.save(
                {"model": final_pretrained},
                args.output_dir / f"fold_{test_fold:02d}_pretrained.pt",
            )

        for subject in subjects:
            test_indices = indices_for(data, assignment, subject, {test_fold})
            refit_indices = indices_for(
                data,
                assignment,
                subject,
                non_test_folds if args.refit else train_folds,
            )
            for mode_index, mode in enumerate(modes):
                best_epoch, validation_metrics, selected_state = selected[
                    (subject, mode)
                ]
                if args.refit:
                    final_state = fit_fixed_epochs(
                        data,
                        final_dataset,
                        final_pretrained,
                        refit_indices,
                        best_epoch,
                        mode,
                        args,
                        fold_seed + subject * 100 + mode_index,
                        f"fold={test_fold} subject={subject}",
                    )
                else:
                    final_state = selected_state
                metrics, labels, probability = evaluate_state(
                    data, final_dataset, final_state, test_indices, mode, args
                )
                row = {
                    "fold": test_fold,
                    "validation_fold": validation_fold,
                    "subject": subject,
                    "mode": mode,
                    "best_epoch": best_epoch,
                    "refit": args.refit,
                    **metrics,
                }
                fold_rows.append(row)
                for sample_index, target, score in zip(
                    test_indices, labels.tolist(), probability.tolist()
                ):
                    predictions.append(
                        {
                            "fold": test_fold,
                            "subject": subject,
                            "mode": mode,
                            "sample": int(sample_index),
                            "target": int(target),
                            "probability": float(score),
                        }
                    )
                print(f"test {json.dumps(row, allow_nan=True)}", flush=True)
                if args.save_checkpoints:
                    torch.save(
                        {
                            "model": final_state,
                            "best_epoch": best_epoch,
                            "validation_metrics": validation_metrics,
                        },
                        args.output_dir
                        / f"fold_{test_fold:02d}_subject_{subject}_{mode}.pt",
                    )

    reported, subject_rows = summarize(fold_rows, predictions, modes, subjects)
    write_csv(
        args.output_dir / "per_subject_fold_metrics.csv",
        fold_rows,
        [
            "fold",
            "validation_fold",
            "subject",
            "mode",
            "best_epoch",
            "refit",
            "acc",
            "f1",
            "auroc",
        ],
    )
    write_csv(
        args.output_dir / "subject_summary.csv",
        subject_rows,
        ["subject", "mode", "n_folds", "acc", "f1", "auroc"],
    )
    write_csv(
        args.output_dir / "oof_predictions.csv",
        predictions,
        ["fold", "subject", "mode", "sample", "target", "probability"],
    )
    summary = {
        "method": "FatigueGuard",
        "protocol": "subject-dependent temporal cross-validation",
        "metric_protocol": (
            "average test-fold metrics within each subject, then report "
            "mean and sample standard deviation across subjects"
        ),
        "configuration": configuration,
        "per_subject_fold": fold_rows,
        "subject_summary": subject_rows,
        "reported_metrics": reported,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True) + "\n"
    )
    print(f"reported_metrics {json.dumps(reported, indent=2, allow_nan=True)}")


class Tee:
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log

    def write(self, text):
        self.terminal.write(text)
        self.log.write(text)
        return len(text)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def main() -> None:
    args = parse_args()
    if args.validate_only:
        run(args)
        return
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(
            f"Output directory must be empty to avoid overwriting a run: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "train.log").open(
        "w", encoding="utf-8", buffering=1
    ) as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(
            Tee(sys.stderr, log)
        ):
            run(args)


if __name__ == "__main__":
    main()
