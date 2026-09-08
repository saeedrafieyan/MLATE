from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from mlate import config as cfg

TABLES = cfg.step_dir("04_machine_learning", "tables")
BENCH = TABLES / "tuned_benchmark.xlsx"
PARAMS = TABLES / "tuned_best_params.xlsx"
OUT = TABLES / "TableS_ML_algorithm_detail.xlsx"
OUT_COMPACT = TABLES / "TableS_ML_algorithm_summary.xlsx"

TASKS = {"printability": "Printability", "cell_response": "Cell Response"}
PROTOCOL_LABEL = {"random": "random split", "doi": "study-grouped (DOI)"}
HOLDOUT = {"random": "random_holdout", "doi": "doi_holdout"}

SELECTIONS = ("weighted", "composed")

METRICS = [
    ("accuracy", "Accuracy"), ("balanced_accuracy", "Balanced accuracy"),
    ("weighted_precision", "Precision (weighted)"),
    ("weighted_recall", "Recall (weighted)"),
    ("weighted_f1", "F1 (weighted)"), ("macro_f1", "F1 (macro)"),
    ("mcc", "MCC"), ("kappa", "Kappa"),
    ("quadratic_kappa", "Quadratic kappa"),
    ("roc_auc_weighted_ovr", "AUC (weighted OvR)"),
    ("log_loss", "Log loss"), ("n_scored", "n test"),
]

COMPACT = ["Accuracy", "F1 (weighted)", "F1 (macro)", "MCC",
           "AUC (weighted OvR)"]


def pretty_params(raw) -> str:
    if not isinstance(raw, str) or not raw.strip():
        return ""
    try:
        params = json.loads(raw)
    except json.JSONDecodeError:
        return str(raw)
    parts = []
    for key, value in params.items():
        if isinstance(value, float):
            value = round(value, 5)
        parts.append(f"{key}={value}")
    return "; ".join(parts)


def build() -> dict[str, pd.DataFrame]:
    bench = pd.read_excel(BENCH)
    params = pd.read_excel(PARAMS)

    bench = bench[(bench["split"] == "test")
                  & (bench["selection"].isin(SELECTIONS))
                  & (~bench["model"].str.startswith("Dummy"))]

    sheets = {}
    for task, task_label in TASKS.items():
        rows = []
        for protocol in ("random", "doi"):
            b = bench[(bench["task"] == task)
                      & (bench["protocol"] == protocol)].copy()
            p = params[(params["task"] == task)
                       & (params["protocol"] == protocol)
                       & (params["fold"] == HOLDOUT[protocol])
                       & (params["selection"].isin(SELECTIONS))]
            lookup = dict(zip(p["model"], p["params"]))
            inner = dict(zip(p["model"], p["inner_score"]))

            b = b.sort_values("weighted_f1", ascending=False)
            for r in b.itertuples():
                row = {"Task": task_label,
                       "Model": r.model,
                       "Family": getattr(r, "family", ""),
                       "Validation protocol": PROTOCOL_LABEL[protocol]}
                for col, label in METRICS:
                    value = getattr(r, col, None)
                    row[label] = (round(float(value), 4)
                                  if pd.notna(value) else None)
                score = inner.get(r.model)
                row["Inner CV score"] = (round(float(score), 4)
                                         if pd.notna(score) else None)
                row["Selected hyperparameters"] = (
                    pretty_params(lookup.get(r.model))
                    or "meta-ensemble of Random Forest, XGBoost and "
                       "logistic regression")
                rows.append(row)
        sheets[task_label] = pd.DataFrame(rows)
    return sheets


def main() -> None:
    sheets = build()

    with pd.ExcelWriter(OUT) as xl:
        for name, frame in sheets.items():
            frame.to_excel(xl, sheet_name=name, index=False)

    keep = ["Task", "Model", "Family", "Validation protocol", *COMPACT,
            "Selected hyperparameters"]
    with pd.ExcelWriter(OUT_COMPACT) as xl:
        for name, frame in sheets.items():
            frame[keep].to_excel(xl, sheet_name=name, index=False)
    for name, frame in sheets.items():
        n_params = (frame["Selected hyperparameters"].str.contains("=")).sum()
        print(f"{name:15s} {len(frame):>4} rows, "
              f"{frame['Model'].nunique():>3} models, "
              f"{n_params} with tuned hyperparameters")
    print(f"\nfull    -> {OUT}")
    print(f"summary -> {OUT_COMPACT}")


if __name__ == "__main__":
    main()
