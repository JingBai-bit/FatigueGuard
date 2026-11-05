from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from .engine import (
    evaluate_finetune_state,
    fit_fixed_epochs,
    pretrain,
    seed_everything,
    select_finetune_epoch,
)
from utils.data import (
    FeatureDataset,
    requested_values,
    scale_fold,
    stratified_fraction_indices,
)
from utils.metrics import metrics


MODES = ("multi", "eeg", "eog")
METRICS = ("acc", "f1", "auroc")


def json_ready_args(args: argparse.Namespace) -> dict:
    return {
        key: str(value.resolve()) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def validate_args(args: argparse.Namespace) -> None:
    if args.num_folds < 3:
        raise ValueError("--num-folds must be at least 3")
    if args.validation_offset % args.num_folds == 0:
        raise ValueError("--validation-offset must not select the test fold")
    if args.purge_boundary_windows < 0:
        raise ValueError("--purge-boundary-windows must be non-negative")
    if args.sequence_length < 1:
        raise ValueError("The input sequence length must be positive")
    if args.pretrain_epochs < 1 or args.finetune_epochs < 1:
        raise ValueError("Training epochs must be positive")
    if args.patience < 1 or args.batch_size < 1:
        raise ValueError("Patience and batch size must be positive")
    if not 0.0 < args.finetune_fraction <= 1.0:
        raise ValueError("--finetune-fraction must be in (0, 1]")


def synchronized_temporal_folds(
    data: dict[str, np.ndarray], num_folds: int, sequence_length: int, purge: int
) -> tuple[np.ndarray, list[dict]]:
    """Assign complete sequences to contiguous folds within each recording."""
    assignment = np.full(len(data["y"]), -1, dtype=np.int64)
    records: list[dict] = []
    pairs = sorted(set(zip(data["subject"].tolist(), data["session"].tolist())))
    for subject, session in pairs:
        indices = np.flatnonzero(
            (data["subject"] == subject) & (data["session"] == session)
        )
        starts = data["sequence_start"][indices].astype(int)
        if not len(indices):
            continue
        if "record_n_windows" in data:
            lengths = np.unique(data["record_n_windows"][indices]).astype(int)
            if len(lengths) != 1:
                raise ValueError(
                    f"subject={subject} session={session} has inconsistent "
                    f"record_n_windows={lengths.tolist()}"
                )
            total_windows = int(lengths[0])
        else:
            total_windows = int(starts.max()) + sequence_length
        if total_windows < num_folds:
            raise ValueError(
                f"subject={subject} session={session} has {total_windows} base "
                f"windows, fewer than num_folds={num_folds}"
            )
        boundaries = np.linspace(0, total_windows, num_folds + 1, dtype=int)
        kept_counts: list[int] = []
        for fold_zero in range(num_folds):
            raw_lo = int(boundaries[fold_zero])
            raw_hi = int(boundaries[fold_zero + 1])
            lo = raw_lo + (purge if fold_zero > 0 else 0)
            hi = raw_hi - (purge if fold_zero < num_folds - 1 else 0)
            keep = indices[(starts >= lo) & (starts + sequence_length <= hi)]
            assignment[keep] = fold_zero + 1
            kept_counts.append(int(len(keep)))
        if min(kept_counts) == 0:
            raise ValueError(
                f"subject={subject} session={session} has an empty temporal fold; "
                "reduce NUM_FOLDS, sequence length, or boundary purge"
            )
        records.append(
            {
                "subject": int(subject),
                "session": int(session),
                "base_windows": total_windows,
                "fold_counts": kept_counts,
                "discarded_cross_boundary_or_purged": int(
                    len(indices) - sum(kept_counts)
                ),
            }
        )
    return assignment, records


def fold_indices(
    assignment: np.ndarray,
    subjects: np.ndarray,
    selected_subjects: list[int],
    folds: set[int],
) -> np.ndarray:
    return np.flatnonzero(
        np.isin(subjects, np.asarray(selected_subjects, dtype=np.int64))
        & np.isin(assignment, np.asarray(sorted(folds), dtype=np.int64))
    )


def subject_fold_indices(
    assignment: np.ndarray, subjects: np.ndarray, subject: int, folds: set[int]
) -> np.ndarray:
    return np.flatnonzero(
        (subjects == subject)
        & np.isin(assignment, np.asarray(sorted(folds), dtype=np.int64))
    )


def mean_std(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {
            "mean": float("nan"), "std": float("nan"), "n": 0,
            "mean_std": "nan ± nan",
        }
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    return {
        "mean": mean, "std": std, "n": int(len(array)),
        "mean_std": f"{mean:.4f} ± {std:.4f}",
    }


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def protocol_validation_report(
    assignment: np.ndarray,
    subjects: np.ndarray,
    y: np.ndarray,
    num_folds: int,
    validation_offset: int,
    require_two_class_validation_test: bool = False,
) -> tuple[list[int], dict[int, list[str]]]:
    """Find subjects suitable for every outer round of the requested protocol."""
    available = sorted(np.unique(subjects).astype(int).tolist())
    all_folds = set(range(1, num_folds + 1))
    valid: list[int] = []
    invalid: dict[int, list[str]] = {}
    for subject in available:
        reasons: list[str] = []
        subject_indices = np.flatnonzero((subjects == subject) & (assignment > 0))
        if len(np.unique(y[subject_indices])) < 2:
            reasons.append("all_assigned_sequences_have_one_class")
        for test_fold in range(1, num_folds + 1):
            validation_fold = (
                (test_fold - 1 + validation_offset) % num_folds
            ) + 1
            training_folds = all_folds - {test_fold, validation_fold}
            train_indices = subject_fold_indices(
                assignment, subjects, subject, training_folds
            )
            validation_indices = subject_fold_indices(
                assignment, subjects, subject, {validation_fold}
            )
            test_indices = subject_fold_indices(
                assignment, subjects, subject, {test_fold}
            )
            if not len(train_indices):
                reasons.append(f"outer_fold_{test_fold}:empty_training")
            elif len(np.unique(y[train_indices])) < 2:
                reasons.append(f"outer_fold_{test_fold}:single_class_training")
            if not len(validation_indices):
                reasons.append(f"outer_fold_{test_fold}:empty_validation")
            elif (
                require_two_class_validation_test
                and len(np.unique(y[validation_indices])) < 2
            ):
                reasons.append(
                    f"outer_fold_{test_fold}:single_class_validation"
                )
            if not len(test_indices):
                reasons.append(f"outer_fold_{test_fold}:empty_test")
            elif (
                require_two_class_validation_test
                and len(np.unique(y[test_indices])) < 2
            ):
                reasons.append(f"outer_fold_{test_fold}:single_class_test")
        if reasons:
            invalid[subject] = reasons
        else:
            valid.append(subject)
    return valid, invalid


def build_oof_summaries(
    predictions: list[dict], modes: list[str], selected_subjects: list[int]
) -> tuple[dict, list[dict], list[dict]]:
    """Aggregate selected folds before calculating each subject's metrics.

    This avoids averaging undefined per-fold AUROCs when a temporal fold happens
    to contain only one class. With all five folds, each valid subject's OOF
    collection contains both classes.
    """
    subject_rows: list[dict] = []
    fold_rows: list[dict] = []
    reported: dict[str, dict] = {}
    for mode in modes:
        mode_predictions = [row for row in predictions if row["mode"] == mode]
        for subject in selected_subjects:
            current = [row for row in mode_predictions if row["subject"] == subject]
            if not current:
                continue
            target = np.asarray([row["target"] for row in current], dtype=np.int64)
            probability = np.asarray(
                [row["probability"] for row in current], dtype=np.float64
            )
            subject_rows.append(
                {
                    "subject": subject,
                    "mode": mode,
                    "folds": ",".join(sorted({row["fold"] for row in current})),
                    "n_folds": len({row["fold"] for row in current}),
                    "n_samples": len(target),
                    "class_0_n": int(np.sum(target == 0)),
                    "class_1_n": int(np.sum(target == 1)),
                    **metrics(target, probability),
                }
            )
        for fold in sorted({row["fold"] for row in mode_predictions}):
            current = [row for row in mode_predictions if row["fold"] == fold]
            target = np.asarray([row["target"] for row in current], dtype=np.int64)
            probability = np.asarray(
                [row["probability"] for row in current], dtype=np.float64
            )
            fold_rows.append(
                {
                    "fold": fold,
                    "mode": mode,
                    "n_samples": len(target),
                    **metrics(target, probability),
                }
            )
        current_subjects = [row for row in subject_rows if row["mode"] == mode]
        all_target = np.asarray(
            [row["target"] for row in mode_predictions], dtype=np.int64
        )
        all_probability = np.asarray(
            [row["probability"] for row in mode_predictions], dtype=np.float64
        )
        pooled = metrics(all_target, all_probability)
        reported[mode] = {
            metric: {
                **mean_std([row[metric] for row in current_subjects]),
                "scope": "mean_std_across_subject_oof_metrics",
                "subjects": [row["subject"] for row in current_subjects],
            }
            for metric in METRICS
        }
        reported[mode]["pooled_f1"] = {
            "value": pooled["f1"],
            "scope": "pooled_selected_test_folds",
            "n_samples": len(all_target),
        }
        reported[mode]["pooled_auroc"] = {
            "value": pooled["auroc"],
            "scope": "pooled_selected_test_folds",
            "n_samples": len(all_target),
        }
    return reported, subject_rows, fold_rows


def build_fold_mean_summaries(
    rows: list[dict],
    predictions: list[dict],
    modes: list[str],
    selected_subjects: list[int],
) -> tuple[dict, list[dict], list[dict]]:
    """Report legacy subject-CV metrics: fold mean, then subject mean±SD."""

    def finite_mean(values: list[float]) -> float:
        array = np.asarray(values, dtype=np.float64)
        array = array[np.isfinite(array)]
        return float(array.mean()) if len(array) else float("nan")

    subject_rows: list[dict] = []
    fold_rows: list[dict] = []
    reported: dict[str, dict] = {}
    for mode in modes:
        mode_rows = [row for row in rows if row["mode"] == mode]
        mode_predictions = [row for row in predictions if row["mode"] == mode]
        for subject in selected_subjects:
            current = [row for row in mode_rows if row["subject"] == subject]
            if not current:
                continue
            subject_rows.append(
                {
                    "subject": subject,
                    "mode": mode,
                    "n_folds": len(current),
                    **{
                        f"{metric}_fold_mean": finite_mean(
                            [row[metric] for row in current]
                        )
                        for metric in METRICS
                    },
                }
            )
        for fold in sorted({row["fold"] for row in mode_predictions}):
            current = [row for row in mode_predictions if row["fold"] == fold]
            target = np.asarray([row["target"] for row in current], dtype=np.int64)
            probability = np.asarray(
                [row["probability"] for row in current], dtype=np.float64
            )
            fold_rows.append(
                {
                    "fold": fold,
                    "mode": mode,
                    "n_samples": len(target),
                    **metrics(target, probability),
                }
            )
        current_subjects = [row for row in subject_rows if row["mode"] == mode]
        all_target = np.asarray(
            [row["target"] for row in mode_predictions], dtype=np.int64
        )
        all_probability = np.asarray(
            [row["probability"] for row in mode_predictions], dtype=np.float64
        )
        pooled = metrics(all_target, all_probability)
        reported[mode] = {
            metric: {
                **mean_std(
                    [row[f"{metric}_fold_mean"] for row in current_subjects]
                ),
                "scope": "subject_macro_of_within_subject_fold_means",
                "subjects": [row["subject"] for row in current_subjects],
            }
            for metric in METRICS
        }
        reported[mode]["pooled_f1"] = {
            "value": pooled["f1"],
            "scope": "pooled_selected_test_folds",
            "n_samples": len(all_target),
        }
        reported[mode]["pooled_auroc"] = {
            "value": pooled["auroc"],
            "scope": "pooled_selected_test_folds",
            "n_samples": len(all_target),
        }
    return reported, subject_rows, fold_rows


def run_subject_cv(
    args: argparse.Namespace,
    data: dict[str, np.ndarray],
    dataset_name: str,
    *,
    assignment_override: np.ndarray | None = None,
    record_folds_override: list[dict] | None = None,
    protocol: str | None = None,
    require_two_class_validation_test: bool = False,
    split_metadata: dict | None = None,
    assignment_field: str = "temporal_fold",
) -> dict:
    validate_args(args)
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "config.json").open("w") as handle:
        json.dump(json_ready_args(args), handle, indent=2)

    subjects, y = data["subject"], data["y"]
    if (assignment_override is None) != (record_folds_override is None):
        raise ValueError(
            "assignment_override and record_folds_override must be provided together"
        )
    if assignment_override is None:
        assignment, record_folds = synchronized_temporal_folds(
            data, args.num_folds, args.sequence_length,
            args.purge_boundary_windows,
        )
    else:
        assignment = np.asarray(assignment_override, dtype=np.int64)
        if assignment.shape != (len(y),):
            raise ValueError(
                f"assignment_override shape={assignment.shape} does not match "
                f"n_samples={len(y)}"
            )
        record_folds = record_folds_override
    valid_subjects, invalid_subjects = protocol_validation_report(
        assignment,
        subjects,
        y,
        args.num_folds,
        args.validation_offset,
        require_two_class_validation_test=require_two_class_validation_test,
    )
    selected_subjects = sorted(np.unique(subjects).astype(int).tolist())
    modes = requested_values(args.modalities, list(MODES))
    selected_outer_folds = requested_values(
        args.folds, list(range(1, args.num_folds + 1))
    )
    assigned_selected = np.flatnonzero(
        np.isin(subjects, np.asarray(selected_subjects)) & (assignment > 0)
    )
    split_description = {
        "protocol": protocol or (
            "synchronized subject-dependent temporal K-fold CV with global "
            "multimodal pretraining and subject-specific fine-tuning"
        ),
        "subject_selection_rule": (
            "all subjects contained in the preprocessed input file; subject "
            "selection is performed before creating the NPZ"
        ),
        "input_subjects": selected_subjects,
        "protocol_valid_subjects": valid_subjects,
        "protocol_invalid_subjects": {
            str(key): value for key, value in invalid_subjects.items()
        },
        "selected_subjects": selected_subjects,
        "num_folds": args.num_folds,
        "validation_offset": args.validation_offset,
        "selected_outer_folds": selected_outer_folds,
        "record_fold_counts": [
            row for row in record_folds if row["subject"] in set(selected_subjects)
        ],
        "assigned_sequences": int(len(assigned_selected)),
        "discarded_selected_sequences": int(
            np.sum(np.isin(subjects, np.asarray(selected_subjects)) & (assignment < 0))
        ),
    }
    if split_metadata:
        split_description.update(split_metadata)
    with (args.output_dir / "fold_assignments.json").open("w") as handle:
        json.dump(split_description, handle, indent=2)
    print(
        f"dataset={dataset_name} protocol={split_description['protocol']} "
        f"input_subjects={selected_subjects} "
        f"num_folds={args.num_folds} selected_outer_folds={selected_outer_folds} "
        f"modalities={modes} assigned_sequences={len(assigned_selected)}",
        flush=True,
    )
    if invalid_subjects:
        raise ValueError(
            "The preprocessed input contains subjects that cannot complete the "
            "requested temporal cross-validation protocol. Filter them while "
            "creating the NPZ. Details: "
            f"{json.dumps(invalid_subjects)}"
        )
    if args.check_data_only:
        return {"config": json_ready_args(args), "split": split_description}

    result_rows: list[dict] = []
    predictions: list[dict] = []
    mode_offsets = {mode: index * 10000 for index, mode in enumerate(MODES, start=1)}
    all_fold_numbers = set(range(1, args.num_folds + 1))

    for test_fold in selected_outer_folds:
        val_fold = ((test_fold - 1 + args.validation_offset) % args.num_folds) + 1
        train_folds = all_fold_numbers - {test_fold, val_fold}
        non_test_folds = all_fold_numbers - {test_fold}
        fold_name = f"fold_{test_fold:02d}"
        fold_seed = args.seed + test_fold * 100000
        selection_global = fold_indices(
            assignment, subjects, selected_subjects, train_folds
        )
        final_global = fold_indices(
            assignment, subjects, selected_subjects, non_test_folds
        )
        test_global = fold_indices(
            assignment, subjects, selected_subjects, {test_fold}
        )
        print(
            f"outer={fold_name} train_folds={sorted(train_folds)} "
            f"validation_fold={val_fold} test_fold={test_fold} "
            f"selection_train_n={len(selection_global)} "
            f"final_train_n={len(final_global)} test_n={len(test_global)}",
            flush=True,
        )

        eeg_select, eog_select = scale_fold(data, selection_global)
        selection_dataset = FeatureDataset(eeg_select, eog_select, y)
        seed_everything(fold_seed)
        selection_pretrained = None
        if args.experiment != "scratch":
            reuse_path = (
                args.reuse_pretrain_dir / f"{fold_name}_selection_pretrain.pt"
                if args.reuse_pretrain_dir else None
            )
            if reuse_path is not None:
                if not reuse_path.exists():
                    raise FileNotFoundError(
                        f"Missing reused pre-training checkpoint: {reuse_path}"
                    )
                selection_pretrained = torch.load(
                    reuse_path, map_location="cpu", weights_only=True
                )["model"]
                print(f"loaded reused pretrain: {reuse_path}", flush=True)
            else:
                selection_pretrained = pretrain(
                    selection_dataset,
                    selection_global,
                    args,
                    args.output_dir,
                    f"{fold_name}_selection_",
                )

        selected_epochs: dict[tuple[int, str], tuple] = {}
        for subject in selected_subjects:
            subject_train_full = subject_fold_indices(
                assignment, subjects, subject, train_folds
            )
            subject_train = stratified_fraction_indices(
                subject_train_full, subjects, y, args.finetune_fraction
            )
            subject_validation = subject_fold_indices(
                assignment, subjects, subject, {val_fold}
            )
            if not len(subject_train) or not len(subject_validation):
                raise RuntimeError(
                    f"{fold_name} subject={subject} has empty fine-tuning or validation data"
                )
            for mode in modes:
                seed_everything(fold_seed + subject * 100 + mode_offsets[mode])
                selected_epochs[(subject, mode)] = select_finetune_epoch(
                    mode,
                    selection_pretrained,
                    selection_dataset,
                    subject_train,
                    subject_validation,
                    args,
                    f"outer={fold_name} subject={subject}",
                )

        if args.refit_non_test:
            eeg_final, eog_final = scale_fold(data, final_global)
            final_dataset = FeatureDataset(eeg_final, eog_final, y)
            seed_everything(fold_seed + 50000)
            final_pretrained = None
            if args.experiment != "scratch":
                reuse_path = (
                    args.reuse_pretrain_dir / f"{fold_name}_pretrain.pt"
                    if args.reuse_pretrain_dir else None
                )
                if reuse_path is not None:
                    if not reuse_path.exists():
                        raise FileNotFoundError(
                            f"Missing reused pre-training checkpoint: {reuse_path}"
                        )
                    final_pretrained = torch.load(
                        reuse_path, map_location="cpu", weights_only=True
                    )["model"]
                    print(f"loaded reused pretrain: {reuse_path}", flush=True)
                else:
                    final_pretrained = pretrain(
                        final_dataset,
                        final_global,
                        args,
                        args.output_dir,
                        f"{fold_name}_",
                    )
        else:
            final_dataset = selection_dataset
            final_pretrained = selection_pretrained

        current_fold_rows: list[dict] = []
        for subject in selected_subjects:
            subject_selection_train = subject_fold_indices(
                assignment, subjects, subject, train_folds
            )
            subject_test = subject_fold_indices(
                assignment, subjects, subject, {test_fold}
            )
            refit_folds = non_test_folds if args.refit_non_test else train_folds
            subject_refit_full = subject_fold_indices(
                assignment, subjects, subject, refit_folds
            )
            subject_refit = stratified_fraction_indices(
                subject_refit_full, subjects, y, args.finetune_fraction
            )
            for mode in modes:
                best_epoch, best_score, best_val_metrics, selected_state = (
                    selected_epochs[(subject, mode)]
                )
                seed_everything(fold_seed + subject * 100 + mode_offsets[mode])
                final_state = (
                    fit_fixed_epochs(
                        mode,
                        final_pretrained,
                        final_dataset,
                        subject_refit,
                        best_epoch,
                        args,
                        f"outer={fold_name} subject={subject}",
                    )
                    if args.refit_non_test
                    else selected_state
                )
                test_metrics, test_y, test_probability = evaluate_finetune_state(
                    mode, final_state, final_dataset, subject_test, args
                )
                row = {
                    "fold": fold_name,
                    "test_fold": test_fold,
                    "validation_fold": val_fold,
                    "subject": subject,
                    "mode": mode,
                    "selection_train_n": int(len(subject_selection_train)),
                    "validation_n": int(
                        len(subject_fold_indices(
                            assignment, subjects, subject, {val_fold}
                        ))
                    ),
                    "final_train_n": int(len(subject_refit)),
                    "test_n": int(len(subject_test)),
                    "test_class_counts": np.bincount(test_y, minlength=2).tolist(),
                    "best_epoch": best_epoch,
                    "best_validation_score": best_score,
                    "best_validation_metrics": best_val_metrics,
                    "refit_non_test": args.refit_non_test,
                    "auroc_defined": bool(len(np.unique(test_y)) == 2),
                    **test_metrics,
                }
                result_rows.append(row)
                current_fold_rows.append(row)
                print(f"test {json.dumps(row, allow_nan=True)}", flush=True)
                if args.save_checkpoints:
                    torch.save(
                        {
                            "model": final_state,
                            "mode": mode,
                            "subject": subject,
                            "fold": fold_name,
                            "best_epoch": best_epoch,
                            "refit_non_test": args.refit_non_test,
                        },
                        args.output_dir / f"{fold_name}_subject_{subject:02d}_{mode}.pt",
                    )
                for index, target, probability in zip(
                    subject_test, test_y, test_probability
                ):
                    prediction = {
                        "fold": fold_name,
                        assignment_field: int(assignment[index]),
                        "subject": int(subjects[index]),
                        "session": int(data["session"][index]),
                        "sequence_start": int(data["sequence_start"][index]),
                        "mode": mode,
                        "target": int(target),
                        "probability": float(probability),
                    }
                    if "group_id" in data:
                        prediction["group_id"] = int(data["group_id"][index])
                    predictions.append(prediction)
        with (args.output_dir / f"{fold_name}_metrics.json").open("w") as handle:
            json.dump(current_fold_rows, handle, indent=2, allow_nan=True)
        del selection_dataset
        if args.refit_non_test:
            del final_dataset
        torch.cuda.empty_cache()

    reported, subject_summary, fold_summary = build_fold_mean_summaries(
        result_rows, predictions, modes, selected_subjects
    )
    write_csv(
        args.output_dir / "per_subject_fold_metrics.csv",
        [
            {
                **row,
                "test_class_counts": json.dumps(row["test_class_counts"]),
                "best_validation_metrics": json.dumps(
                    row["best_validation_metrics"], allow_nan=True
                ),
            }
            for row in result_rows
        ],
        [
            "fold", "test_fold", "validation_fold", "subject", "mode",
            "selection_train_n", "validation_n", "final_train_n", "test_n",
            "test_class_counts", "best_epoch", "best_validation_score",
            "best_validation_metrics", "refit_non_test", "auroc_defined",
            "acc", "f1", "auroc",
        ],
    )
    write_csv(
        args.output_dir / "subject_summary.csv",
        subject_summary,
        [
            "subject", "mode", "n_folds", "acc_fold_mean",
            "f1_fold_mean", "auroc_fold_mean",
        ],
    )
    write_csv(
        args.output_dir / "fold_pooled_metrics.csv",
        fold_summary,
        ["fold", "mode", "n_samples", "acc", "f1", "auroc"],
    )
    write_csv(
        args.output_dir / "oof_predictions.csv",
        predictions,
        [
            "fold", assignment_field, "subject", "session", "sequence_start",
            *(["group_id"] if "group_id" in data else []),
            "mode", "target", "probability",
        ],
    )
    reported_rows = []
    for mode, mode_values in reported.items():
        for metric_name, values in mode_values.items():
            reported_rows.append(
                {
                    "mode": mode,
                    "metric": metric_name,
                    "scope": values.get("scope"),
                    "mean": values.get("mean"),
                    "std": values.get("std"),
                    "n": values.get("n"),
                    "mean_std": values.get("mean_std"),
                    "value": values.get("value"),
                    "n_samples": values.get("n_samples"),
                }
            )
    write_csv(
        args.output_dir / "reported_metrics.csv",
        reported_rows,
        [
            "mode", "metric", "scope", "mean", "std", "n", "mean_std",
            "value", "n_samples",
        ],
    )
    summary = {
        "dataset": dataset_name,
        "protocol": split_description["protocol"],
        "config": json_ready_args(args),
        "split": split_description,
        "metric_protocol": {
            "acc_f1_auroc": (
                "calculate each subject/test-fold metric, average selected "
                "outer folds within each subject, then report subject "
                "mean±sample std"
            ),
            "pooled_f1_auroc": (
                "calculate metrics after pooling all selected subjects and test folds"
            ),
        },
        "per_subject_fold": result_rows,
        "subject_summary": subject_summary,
        "fold_pooled_metrics": fold_summary,
        "reported_metrics": reported,
    }
    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, allow_nan=True)
    print(f"reported metrics {json.dumps(reported, indent=2, allow_nan=True)}", flush=True)
    return summary
