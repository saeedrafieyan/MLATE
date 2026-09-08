from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from mlate import config as cfg

TABLES = cfg.step_dir("05_deep_learning", "tables")
DL_BENCH = TABLES / "dl_benchmark.xlsx"
DL_PARAMS = TABLES / "dl_best_params.xlsx"
FOUNDATION = TABLES / "foundation_models.xlsx"

OUT = TABLES / "TableS_DL_model_detail.xlsx"
OUT_COMPACT = TABLES / "TableS_DL_model_summary.xlsx"

TASKS = {"printability": "Printability", "cell_response": "Cell Response"}
PROTOCOL_LABEL = {"random": "random split", "doi": "study-grouped (DOI)"}
HOLDOUT = {"random": "random_holdout", "doi": "doi_holdout"}

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


def foundation_config(model: str) -> str:
    checkpoint = ("tabpfn-v2.6-classifier-v2.6_default.ckpt"
                  if model.startswith("TabPFN") else "TabICL v2 (auto-download)")
    tuning = ("; inference tuning: temperature calibration and per-class "
              "decision thresholds against F1" if model.endswith("(thinking)")
              else "")
    return (f"zero-shot, no hyperparameter search; checkpoint={checkpoint}; "
            f"n_estimators=8{tuning}")


def build() -> dict[str, pd.DataFrame]:
    dl = pd.read_excel(DL_BENCH)
    params = pd.read_excel(DL_PARAMS)
    fnd = pd.read_excel(FOUNDATION, sheet_name="pooled")

    dl = dl[(dl["split"] == "test") & (dl["selection"] == "weighted")]
    fnd = fnd[(fnd["split"] == "test") & (fnd["selection"] == "zero_shot")]

    sheets = {}
    for task, task_label in TASKS.items():
        rows = []
        for protocol in ("random", "doi"):
            p = params[(params["task"] == task)
                       & (params["protocol"] == protocol)
                       & (params["fold"] == HOLDOUT[protocol])
                       & (params["selection"] == "weighted")]
            lookup = dict(zip(p["model"], p["params"]))
            inner = dict(zip(p["model"], p["inner_score"]))
            epoch = dict(zip(p["model"], p["best_epoch"]))

            block = pd.concat([
                dl[(dl["task"] == task) & (dl["protocol"] == protocol)]
                .assign(kind="deep learning"),
                fnd[(fnd["task"] == task) & (fnd["protocol"] == protocol)]
                .assign(kind="foundation model"),
            ], ignore_index=True).sort_values("weighted_f1", ascending=False)

            for r in block.itertuples():
                row = {"Task": task_label,
                       "Model": r.model,
                       "Type": r.kind,
                       "Validation protocol": PROTOCOL_LABEL[protocol]}
                for col, label in METRICS:
                    value = getattr(r, col, None)
                    row[label] = (round(float(value), 4)
                                  if pd.notna(value) else None)
                if r.kind == "foundation model":
                    row["Inner CV score"] = None
                    row["Selected epoch"] = None
                    row["Configuration"] = foundation_config(r.model)
                else:
                    score, best = inner.get(r.model), epoch.get(r.model)
                    row["Inner CV score"] = (round(float(score), 4)
                                             if pd.notna(score) else None)
                    row["Selected epoch"] = (int(best) if pd.notna(best)
                                             else None)
                    row["Configuration"] = pretty_params(lookup.get(r.model))
                rows.append(row)
        sheets[task_label] = pd.DataFrame(rows)
    return sheets


def main() -> None:
    sheets = build()

    with pd.ExcelWriter(OUT) as xl:
        for name, frame in sheets.items():
            frame.to_excel(xl, sheet_name=name, index=False)

    keep = ["Task", "Model", "Type", "Validation protocol", *COMPACT,
            "Configuration"]
    with pd.ExcelWriter(OUT_COMPACT) as xl:
        for name, frame in sheets.items():
            frame[keep].to_excel(xl, sheet_name=name, index=False)

    for name, frame in sheets.items():
        kinds = frame["Type"].value_counts().to_dict()
        print(f"{name:15s} {len(frame):>3} rows, "
              f"{frame['Model'].nunique():>2} models  {kinds}")
    print(f"\nfull    -> {OUT}")
    print(f"summary -> {OUT_COMPACT}")


if __name__ == "__main__":
    main()
