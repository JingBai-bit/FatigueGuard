"""Unified feature-file interface and temporal cross-validation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


REQUIRED_ALIASES = {
    "eeg": ("eeg",),
    "eog": ("eog",),
    "labels": ("labels", "y"),
    "subjects": ("subjects", "subject"),
    "sessions": ("sessions", "session"),
    "sequence_start": ("sequence_start", "start"),
}


def _read_array(archive: np.lib.npyio.NpzFile, logical_name: str) -> np.ndarray:
    for key in REQUIRED_ALIASES[logical_name]:
        if key in archive:
            return np.asarray(archive[key])
    accepted = ", ".join(REQUIRED_ALIASES[logical_name])
    raise ValueError(f"Missing '{logical_name}' array; accepted keys: {accepted}")


def _integer_vector(archive, name: str) -> np.ndarray:
    values = _read_array(archive, name)
    if values.ndim != 1 or values.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a one-dimensional integer array")
    if values.size and (values.min() < 0 or values.max() > np.iinfo(np.int64).max):
        raise ValueError(f"{name} must contain non-negative int64 values")
    return values.astype(np.int64, copy=False)


@dataclass(frozen=True)
class FeatureData:
    eeg: np.ndarray
    eog: np.ndarray
    labels: np.ndarray
    subjects: np.ndarray
    sessions: np.ndarray
    sequence_start: np.ndarray
    metadata: dict[str, object]
    record_n_windows: np.ndarray | None = None

    @property
    def sequence_length(self) -> int:
        return int(self.eeg.shape[1])


def load_feature_file(path: str | Path) -> FeatureData:
    """Load a dataset-independent NPZ feature file.

    Required logical arrays are EEG, EOG, binary labels, subject IDs, session
    IDs, and chronological sequence starts.  Both singular legacy keys such as
    ``subject``/``session`` and plural public-interface keys are accepted.
    """

    feature_path = Path(path).expanduser().resolve()
    if not feature_path.is_file():
        raise FileNotFoundError(f"Feature file does not exist: {feature_path}")

    with np.load(feature_path, allow_pickle=False) as archive:
        eeg = _read_array(archive, "eeg").astype(np.float32, copy=False)
        eog = _read_array(archive, "eog").astype(np.float32, copy=False)
        labels = _integer_vector(archive, "labels")
        subjects = _integer_vector(archive, "subjects")
        sessions = _integer_vector(archive, "sessions")
        starts = _integer_vector(archive, "sequence_start")
        record_lengths = None
        if "record_n_windows" in archive:
            record_lengths = np.asarray(archive["record_n_windows"])
            if record_lengths.ndim != 1 or record_lengths.dtype.kind not in "iu":
                raise ValueError(
                    "record_n_windows must be a one-dimensional integer array"
                )
        metadata: dict[str, object] = {"source": str(feature_path)}
        for key in archive.files:
            if key not in (
                "dataset",
                "window_seconds",
                "sequence_length",
                "sequence_stride",
                "label_threshold",
            ):
                continue
            value = np.asarray(archive[key])
            if value.ndim == 0 and value.dtype.kind in "biufU":
                metadata[key] = value.item()

    if eeg.ndim != 3 or eog.ndim != 3:
        raise ValueError("EEG and EOG must have shape [samples, sequence, features]")
    n = len(labels)
    if n == 0 or min(eeg.shape) == 0 or min(eog.shape) == 0:
        raise ValueError(
            "Feature arrays must have nonempty sample, sequence, and feature dimensions"
        )
    if any(len(array) != n for array in (eeg, eog, subjects, sessions, starts)):
        raise ValueError("All arrays must contain the same number of samples")
    if eeg.shape[:2] != eog.shape[:2]:
        raise ValueError("EEG and EOG must share sample and sequence dimensions")
    unique_labels = set(np.unique(labels).tolist())
    if not unique_labels.issubset({0, 1}):
        raise ValueError(f"Labels must be binary 0/1; found {sorted(unique_labels)}")
    if not np.isfinite(eeg).all() or not np.isfinite(eog).all():
        raise ValueError("Feature arrays contain NaN or infinite values")
    if "sequence_length" in metadata and metadata["sequence_length"] != eeg.shape[1]:
        raise ValueError("sequence_length metadata does not match feature arrays")
    if record_lengths is not None:
        if len(record_lengths) != n or np.any(record_lengths < starts + eeg.shape[1]):
            raise ValueError("record_n_windows must cover every complete sequence")

    return FeatureData(
        eeg, eog, labels, subjects, sessions, starts, metadata, record_lengths
    )


def temporal_fold_assignment(
    data: FeatureData, num_folds: int, purge: int = 0
) -> np.ndarray:
    """Assign whole sequences to window intervals; discard boundary overlaps (-1)."""

    if num_folds < 3:
        raise ValueError("num_folds must be at least 3")
    if purge < 0:
        raise ValueError("purge must be non-negative")
    assignment = np.full(len(data.labels), -1, dtype=np.int64)
    pairs = np.unique(np.column_stack([data.subjects, data.sessions]), axis=0)
    for subject, session in pairs:
        indices = np.flatnonzero(
            (data.subjects == subject) & (data.sessions == session)
        )
        starts = data.sequence_start[indices]
        if len(np.unique(starts)) != len(starts):
            raise ValueError(
                f"Duplicate sequence starts: subject={subject} session={session}"
            )
        total_windows = int(starts.max()) + data.sequence_length
        if data.record_n_windows is not None:
            lengths = np.unique(data.record_n_windows[indices])
            if len(lengths) != 1:
                raise ValueError(
                    f"Inconsistent recording length: subject={subject} session={session}"
                )
            total_windows = int(lengths[0])
        boundaries = np.linspace(0, total_windows, num_folds + 1, dtype=int)
        for fold in range(num_folds):
            lo = boundaries[fold] + (purge if fold > 0 else 0)
            hi = boundaries[fold + 1] - (purge if fold < num_folds - 1 else 0)
            keep = indices[(starts >= lo) & (starts + data.sequence_length <= hi)]
            if len(keep) == 0:
                raise ValueError(
                    f"Empty temporal fold {fold + 1}: subject={subject} session={session}; "
                    "reduce num_folds or purge, or supply longer sessions"
                )
            assignment[keep] = fold + 1
    return assignment


def parse_folds(spec: str, num_folds: int) -> list[int]:
    if spec.strip().lower() == "all":
        return list(range(1, num_folds + 1))
    selected = [int(value.strip()) for value in spec.split(",") if value.strip()]
    invalid = [fold for fold in selected if fold < 1 or fold > num_folds]
    if invalid:
        raise ValueError(f"Fold IDs must be between 1 and {num_folds}: {invalid}")
    if not selected:
        raise ValueError("No folds selected")
    return sorted(set(selected))


def indices_for(
    data: FeatureData,
    assignment: np.ndarray,
    subjects: list[int] | int,
    folds: set[int],
) -> np.ndarray:
    subject_values = [subjects] if isinstance(subjects, int) else subjects
    return np.flatnonzero(
        np.isin(data.subjects, np.asarray(subject_values))
        & np.isin(assignment, np.asarray(sorted(folds)))
    )


def scale_features(
    data: FeatureData, fit_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fit feature-wise scalers on training data only."""

    if len(fit_indices) == 0:
        raise ValueError("Cannot fit feature scalers on an empty training split")
    eeg_scaler = StandardScaler().fit(
        data.eeg[fit_indices].reshape(-1, data.eeg.shape[-1])
    )
    eog_scaler = StandardScaler().fit(
        data.eog[fit_indices].reshape(-1, data.eog.shape[-1])
    )
    eeg = eeg_scaler.transform(data.eeg.reshape(-1, data.eeg.shape[-1])).reshape(
        data.eeg.shape
    )
    eog = eog_scaler.transform(data.eog.reshape(-1, data.eog.shape[-1])).reshape(
        data.eog.shape
    )
    return eeg.astype(np.float32), eog.astype(np.float32)


class FeatureDataset(Dataset):
    def __init__(self, eeg: np.ndarray, eog: np.ndarray, labels: np.ndarray):
        self.eeg = torch.from_numpy(eeg)
        self.eog = torch.from_numpy(eog)
        self.labels = torch.from_numpy(labels.astype(np.int64, copy=False))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.eeg[index], self.eog[index], self.labels[index]
