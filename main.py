from __future__ import annotations

import argparse
from pathlib import Path

import torch

from train.trainer import run_subject_cv
from utils.data import load_preprocessed_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Synchronized within-subject temporal K-fold CV using a preprocessed "
            "EEG/EOG NPZ file"
        )
    )
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reuse-pretrain-dir", type=Path)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--folds", default="all")
    parser.add_argument("--validation-offset", type=int, default=2)
    parser.add_argument("--purge-boundary-windows", type=int, default=0)
    parser.add_argument("--modalities", default="multi,eeg,eog")
    parser.add_argument(
        "--selection-metric",
        choices=("auroc", "f1", "acc", "composite"),
        default="acc",
    )
    parser.add_argument("--finetune-fraction", type=float, default=1.0)
    parser.add_argument("--pretrain-epochs", type=int, default=100)
    parser.add_argument("--finetune-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--pretrain-lr", type=float, default=1e-4)
    parser.add_argument("--finetune-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--refit-non-test", action="store_true")
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument(
        "--check-data-only",
        action="store_true",
        help="validate the input and temporal folds without training",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_preprocessed_data(args.input_file)
    args.sequence_length = int(data["eeg"].shape[1])
    run_subject_cv(args, data, dataset_name=args.dataset_name)


if __name__ == "__main__":
    main()
