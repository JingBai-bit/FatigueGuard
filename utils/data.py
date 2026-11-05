from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


REQUIRED_FIELDS = ("eeg", "eog", "y", "subject", "session", "sequence_start")
OPTIONAL_SAMPLE_FIELDS = ("record_n_windows", "group_id")


class FeatureDataset(Dataset):
    def __init__(self, eeg: np.ndarray, eog: np.ndarray, y: np.ndarray):
        self.eeg = torch.from_numpy(eeg.astype(np.float32))
        self.eog = torch.from_numpy(eog.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        return self.eeg[index], self.eog[index], self.y[index]


def _integer_vector(name: str, value: np.ndarray, n_samples: int) -> np.ndarray:
    value = np.asarray(value)
    if value.shape != (n_samples,):
        raise ValueError(
            f"NPZ field {name!r} must have shape ({n_samples},), got {value.shape}"
        )
    if not np.issubdtype(value.dtype, np.number) or not np.all(np.isfinite(value)):
        raise ValueError(f"NPZ field {name!r} must contain finite numeric values")
    converted = value.astype(np.int64)
    if not np.array_equal(value, converted):
        raise ValueError(f"NPZ field {name!r} must contain integers")
    return converted


def load_preprocessed_data(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Preprocessed input file does not exist: {path}")
    if path.suffix.lower() != ".npz":
        raise ValueError("The preprocessed input must be an .npz file")
    with np.load(path, allow_pickle=False) as archive:
        missing = [name for name in REQUIRED_FIELDS if name not in archive.files]
        if missing:
            raise ValueError(
                f"Missing required NPZ fields: {missing}; required={REQUIRED_FIELDS}"
            )
        data = {name: np.asarray(archive[name]) for name in REQUIRED_FIELDS}
        for name in OPTIONAL_SAMPLE_FIELDS:
            if name in archive.files:
                data[name] = np.asarray(archive[name])
        stored_sequence_length = (
            int(np.asarray(archive["sequence_length"]).item())
            if "sequence_length" in archive.files
            else None
        )

    for name in ("eeg", "eog"):
        if data[name].ndim == 2:
            data[name] = data[name][:, None, :]
        if data[name].ndim != 3:
            raise ValueError(
                f"NPZ field {name!r} must be [samples, sequence, features], "
                f"got shape={data[name].shape}"
            )
        if not np.issubdtype(data[name].dtype, np.number):
            raise ValueError(f"NPZ field {name!r} must be numeric")
        if not np.all(np.isfinite(data[name])):
            raise ValueError(f"NPZ field {name!r} contains NaN or infinity")
        data[name] = data[name].astype(np.float32, copy=False)

    n_samples = len(data["eeg"])
    if n_samples == 0:
        raise ValueError("The preprocessed input contains no samples")
    if len(data["eog"]) != n_samples:
        raise ValueError("EEG and EOG sample counts do not match")
    if data["eeg"].shape[1] != data["eog"].shape[1]:
        raise ValueError("EEG and EOG sequence lengths do not match")
    sequence_length = int(data["eeg"].shape[1])
    if stored_sequence_length is not None and stored_sequence_length != sequence_length:
        raise ValueError(
            f"Stored sequence_length={stored_sequence_length} does not match "
            f"the feature arrays ({sequence_length})"
        )

    for name in ("y", "subject", "session", "sequence_start"):
        data[name] = _integer_vector(name, data[name], n_samples)
    for name in OPTIONAL_SAMPLE_FIELDS:
        if name in data:
            data[name] = _integer_vector(name, data[name], n_samples)
    labels = np.unique(data["y"])
    if not set(labels.tolist()).issubset({0, 1}) or len(labels) != 2:
        raise ValueError(
            f"NPZ field 'y' must use both binary labels 0 and 1; got {labels.tolist()}"
        )
    if np.any(data["sequence_start"] < 0):
        raise ValueError("NPZ field 'sequence_start' must be non-negative")
    if "record_n_windows" in data and np.any(data["record_n_windows"] <= 0):
        raise ValueError("NPZ field 'record_n_windows' must be positive")

    print(
        f"loaded preprocessed data: {path.resolve()} samples={n_samples} "
        f"sequence_length={sequence_length} eeg_dim={data['eeg'].shape[-1]} "
        f"eog_dim={data['eog'].shape[-1]} "
        f"subjects={sorted(np.unique(data['subject']).tolist())}",
        flush=True,
    )
    return data


def scale_fold(
    data: dict[str, np.ndarray], fit_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    eeg, eog = data["eeg"], data["eog"]
    eeg_scaler = StandardScaler().fit(eeg[fit_indices].reshape(-1, eeg.shape[-1]))
    eog_scaler = StandardScaler().fit(eog[fit_indices].reshape(-1, eog.shape[-1]))
    eeg_scaled = eeg_scaler.transform(eeg.reshape(-1, eeg.shape[-1])).reshape(
        eeg.shape
    ).astype(np.float32)
    eog_scaled = eog_scaler.transform(eog.reshape(-1, eog.shape[-1])).reshape(
        eog.shape
    ).astype(np.float32)
    return eeg_scaled, eog_scaled


def requested_values(value: str, available: list[int] | list[str]):
    if value == "all":
        return available
    requested = [item.strip() for item in value.split(",") if item.strip()]
    if available and isinstance(available[0], int):
        requested = [int(item) for item in requested]
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(f"Unknown values {unknown}; available values are {available}")
    return requested


def stratified_fraction_indices(
    indices: np.ndarray,
    subjects: np.ndarray,
    y: np.ndarray,
    fraction: float,
) -> np.ndarray:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("--finetune-fraction must be in (0, 1]")
    if fraction == 1.0:
        return indices
    selected: list[int] = []
    for subject in np.unique(subjects[indices]):
        subject_indices = indices[subjects[indices] == subject]
        for label in np.unique(y[subject_indices]):
            class_indices = subject_indices[y[subject_indices] == label]
            count = max(1, int(round(len(class_indices) * fraction)))
            positions = np.linspace(
                0, len(class_indices) - 1, min(count, len(class_indices)), dtype=int
            )
            selected.extend(class_indices[positions].tolist())
    return np.asarray(sorted(set(selected)), dtype=np.int64)
