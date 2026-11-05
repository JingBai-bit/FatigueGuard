from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    prediction = (probability >= 0.5).astype(np.int64)
    result = {
        "acc": float(accuracy_score(y, prediction)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
    }
    result["auroc"] = (
        float(roc_auc_score(y, probability))
        if len(np.unique(y)) == 2
        else float("nan")
    )
    return result


def selection_score(result: dict[str, float], selection_metric: str) -> float:
    if selection_metric == "composite":
        auc = result["auroc"] if math.isfinite(result["auroc"]) else result["acc"]
        return float((auc + result["f1"] + result["acc"]) / 3.0)
    value = float(result[selection_metric])
    if not math.isfinite(value):
        print(
            f"validation {selection_metric} is undefined; falling back to ACC",
            flush=True,
        )
        return float(result["acc"])
    return value
