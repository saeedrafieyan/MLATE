from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             cohen_kappa_score, confusion_matrix, f1_score,
                             log_loss, matthews_corrcoef,
                             precision_recall_fscore_support, roc_auc_score)

from mlate import config as cfg

N_BOOTSTRAP = 1000


def scores(y_true, y_pred, proba=None, labels=None) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0)[0]),
        "macro_recall": float(precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0)[1]),
        "weighted_precision": float(precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0)[0]),
        "weighted_recall": float(precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0)[1]),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro",
                                   zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted",
                                      zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "quadratic_kappa": float(cohen_kappa_score(y_true, y_pred,
                                                   weights="quadratic")),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "specificity_macro": _specificity(y_true, y_pred, labels),
    }
    out.update(_probability_scores(y_true, proba, labels))
    return out


def _specificity(y_true, y_pred, labels=None) -> float:
    labels = list(labels) if labels is not None else sorted(set(np.asarray(y_true)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    total = cm.sum()
    out = []
    for i in range(len(labels)):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        out.append(tn / (tn + fp) if (tn + fp) else np.nan)
    return float(np.nanmean(out))


def _probability_scores(y_true, proba, labels) -> dict:
    blank = {"roc_auc_macro_ovr": np.nan, "roc_auc_weighted_ovr": np.nan,
             "log_loss": np.nan, "brier_multiclass": np.nan}
    if proba is None or labels is None:
        return blank
    proba = np.asarray(proba, dtype=float)
    labels = list(labels)
    if proba.ndim != 2 or proba.shape[1] != len(labels) or np.isnan(proba).any():
        return blank

    rows = proba.sum(axis=1, keepdims=True)
    proba = np.divide(proba, np.where(rows == 0, 1.0, rows))

    present = sorted(set(np.asarray(y_true)))
    out = dict(blank)
    try:
        keep = [labels.index(c) for c in present]
        if len(present) >= 2:
            sub = proba[:, keep]
            sub = sub / np.where(sub.sum(axis=1, keepdims=True) == 0, 1.0,
                                 sub.sum(axis=1, keepdims=True))
            kw = dict(labels=present, multi_class="ovr")
            out["roc_auc_macro_ovr"] = float(
                roc_auc_score(y_true, sub, average="macro", **kw))
            out["roc_auc_weighted_ovr"] = float(
                roc_auc_score(y_true, sub, average="weighted", **kw))
    except Exception:
        pass
    try:
        out["log_loss"] = float(log_loss(y_true, proba, labels=labels))
    except Exception:
        pass
    try:
        onehot = np.zeros_like(proba)
        index = {c: i for i, c in enumerate(labels)}
        for r, c in enumerate(np.asarray(y_true)):
            onehot[r, index[c]] = 1.0
        out["brier_multiclass"] = float(((proba - onehot) ** 2).sum(axis=1).mean())
    except Exception:
        pass
    return out


def per_class(y_true, y_pred, labels) -> pd.DataFrame:
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    predicted = pd.Series(y_pred).value_counts()
    return pd.DataFrame({
        "class": labels, "precision": p, "recall": r, "f1": f,
        "support": s,
        "n_predicted": [int(predicted.get(c, 0)) for c in labels],
    })


def confusion(y_true, y_pred, labels) -> pd.DataFrame:
    m = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(m, index=[f"true_{c}" for c in labels],
                        columns=[f"pred_{c}" for c in labels])


def bootstrap_ci(y_true, y_pred, metric: str = "macro_f1",
                 n: int = N_BOOTSTRAP, alpha: float = 0.05,
                 seed: int = cfg.RANDOM_STATE) -> tuple[float, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rng = np.random.default_rng(seed)

    single = {
        "macro_f1": lambda t, p_: f1_score(t, p_, average="macro",
                                           zero_division=0),
        "weighted_f1": lambda t, p_: f1_score(t, p_, average="weighted",
                                              zero_division=0),
        "accuracy": accuracy_score,
        "balanced_accuracy": balanced_accuracy_score,
        "mcc": matthews_corrcoef,
        "kappa": cohen_kappa_score,
        "quadratic_kappa": lambda t, p_: cohen_kappa_score(t, p_,
                                                           weights="quadratic"),
    }.get(metric)
    if single is None:
        single = lambda t, p_: scores(t, p_)[metric]

    out = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        out.append(float(single(y_true[idx], y_pred[idx])))
    if not out:
        return (np.nan, np.nan)
    return (float(np.percentile(out, 100 * alpha / 2)),
            float(np.percentile(out, 100 * (1 - alpha / 2))))


TASK_LABELS = {"printability": [0, 1, 2, 3],
               "cell_response": [1, 2, 3, 4, 5],
               "cell_response_cellular": [2, 3, 4, 5]}


def score_predictions(preds: pd.DataFrame,
                      bootstrap: bool = True) -> pd.DataFrame:
    rows = []
    keys = ["model", "task", "protocol", "selection", "split"]
    keys = [k for k in keys if k in preds.columns]
    for values, g in preds.groupby(keys, sort=False):
        record = dict(zip(keys, values if isinstance(values, tuple)
                          else (values,)))
        labels = TASK_LABELS[record["task"]]
        pcols = [f"p_{c}" for c in labels]
        proba = None
        if all(c in g.columns for c in pcols):
            arr = g[pcols].to_numpy(dtype=float)
            proba = None if np.isnan(arr).any() else arr
        record["n_scored"] = int(len(g))
        record.update(scores(g["y_true"], g["y_pred"], proba, labels))
        if bootstrap and record.get("split") == "test":
            for metric in ("weighted_f1", "macro_f1"):
                lo, hi = bootstrap_ci(g["y_true"], g["y_pred"], metric)
                record[f"{metric}_lo"] = lo
                record[f"{metric}_hi"] = hi
        rows.append(record)
    return pd.DataFrame(rows)


def train_test_gap(board: pd.DataFrame,
                   metric: str = "weighted_f1") -> pd.DataFrame:
    keys = [k for k in ("model", "task", "protocol", "selection")
            if k in board.columns]
    wide = board.pivot_table(index=keys, columns="split", values=metric)
    if not {"train", "test"} <= set(wide.columns):
        return pd.DataFrame()
    wide = wide.reset_index()
    wide[f"{metric}_gap"] = wide["train"] - wide["test"]
    return wide.rename(columns={"train": f"{metric}_train",
                                "test": f"{metric}_test"})


def summarise(frame: pd.DataFrame, labels) -> pd.DataFrame:
    rows = []
    keys = ["model", "task", "protocol"]
    pcols = [f"p_{c}" for c in labels]

    def _proba(block):
        if not all(c in block.columns for c in pcols):
            return None
        arr = block[pcols].to_numpy(dtype=float)
        return None if np.isnan(arr).any() else arr

    for (model, task, protocol), g in frame.groupby(keys, sort=False):
        pooled = scores(g["y_true"], g["y_pred"], _proba(g), labels)
        by_fold = pd.DataFrame([
            scores(h["y_true"], h["y_pred"], _proba(h), labels)
            for _, h in g.groupby("fold")])
        lo, hi = bootstrap_ci(g["y_true"], g["y_pred"], "macro_f1")
        rows.append({
            "model": model, "task": task, "protocol": protocol,
            "n_scored": int(len(g)), "n_folds": int(g["fold"].nunique()),
            **pooled,
            "macro_f1_lo": lo, "macro_f1_hi": hi,
            "accuracy_sd": float(by_fold["accuracy"].std(ddof=1)),
            "macro_f1_sd": float(by_fold["macro_f1"].std(ddof=1)),
        })
    return pd.DataFrame(rows)
