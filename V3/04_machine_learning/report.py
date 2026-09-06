"""
Score the tuned benchmark
=========================

    python 04_machine_learning/report.py

Reads the stored predictions written by tuning.py and produces every table the
manuscript needs. Nothing is refitted here, so the scoring can be revised
without touching the compute that produced the predictions.

Sheets written to tuned_benchmark.xlsx:

  leaderboard     one row per model x task x protocol x selection x split,
                  the full sixteen-metric panel, with bootstrap intervals on
                  the test rows
  test            the same, filtered to split == "test" and the primary
                  selection - the table the main text quotes
  overfitting     train minus test, the diagnostic that separates a model that
                  could not learn from one that learned study structure
  protocol_gap    random versus study-grouped for the same model
  per_class       precision, recall, F1 and support per class for the leaders
  confusion       confusion matrices for the same

Training rows are retained but never presented as results; every sheet that
carries them labels the split explicitly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from mlate import config as cfg
from mlate import evaluation as ev
from mlate import models as zoo
from mlate import style as ms

TABLES = cfg.step_dir("04_machine_learning", "tables")
PREDS = TABLES / "predictions_tuned"
PRIMARY = "weighted"          # the selection the main text reports


def load(tasks: list[str] | None = None) -> pd.DataFrame:
    frames = []
    for path in sorted(PREDS.glob("*.parquet")):
        task = path.stem.split("__")[0]
        if tasks and task not in tasks:
            continue
        part = pd.read_parquet(path)
        part["task"] = task
        frames.append(part)
    if not frames:
        raise SystemExit(f"no predictions in {PREDS}; run tuning.py first")
    return pd.concat(frames, ignore_index=True)


def detail(preds: pd.DataFrame, board: pd.DataFrame, top: int = 3):
    """Per-class tables and confusion matrices for the leading models."""
    per_class, confusion = [], []
    test = board[(board["split"] == "test")]

    # The set of models to detail is chosen per TASK, not per task-and-protocol,
    # and every chosen model is then detailed under BOTH protocols. Selecting
    # separately per protocol produces a table where the random-split winner
    # has no grouped rows, which makes the obvious figure - one model, both
    # protocols, side by side - impossible to draw.
    wanted: dict[str, list[str]] = {}
    for task, g in test.groupby("task"):
        cell = g[g["selection"].isin([PRIMARY, "untuned", "composed"])]
        names = []
        for protocol, h in cell.groupby("protocol"):
            names += (h[~h["model"].str.startswith("Dummy")]
                      .nlargest(top, "weighted_f1")["model"].tolist())
        names.append("Dummy (majority)")
        wanted[task] = list(dict.fromkeys(names))

    for (task, protocol), g in test.groupby(["task", "protocol"]):
        for model in wanted.get(task, []):
            rows = preds[(preds["task"] == task)
                         & (preds["protocol"] == protocol)
                         & (preds["split"] == "test")
                         & (preds["model"] == model)]
            rows = rows[rows["selection"].isin([PRIMARY, "untuned",
                                                "composed"])]
            if rows.empty:
                continue
            labels = ev.TASK_LABELS[task]
            pc = ev.per_class(rows["y_true"], rows["y_pred"], labels)
            for col, val in (("protocol", protocol), ("task", task),
                             ("model", model)):
                pc.insert(0, col, val)
            per_class.append(pc)

            cm = ev.confusion(rows["y_true"], rows["y_pred"], labels)
            cm = cm.reset_index(names="truth")
            for col, val in (("protocol", protocol), ("task", task),
                             ("model", model)):
                cm.insert(0, col, val)
            confusion.append(cm)
    return (pd.concat(per_class, ignore_index=True) if per_class
            else pd.DataFrame(),
            pd.concat(confusion, ignore_index=True) if confusion
            else pd.DataFrame())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", nargs="*", default=None,
                    help="score only these tasks; default is every task with "
                         "stored predictions")
    ap.add_argument("--tag", default="",
                    help="suffix for the output workbook, so a separate "
                         "analysis does not overwrite the main one "
                         "(e.g. --tag _cellular)")
    args = ap.parse_args()

    preds = load(args.tasks)
    print(f"{len(preds):,} predictions | {preds['model'].nunique()} models")

    board = ev.score_predictions(preds)
    board["family"] = board["model"].map(
        lambda m: zoo.REGISTRY[m].family if m in zoo.REGISTRY else "?")

    gap = ev.train_test_gap(board)
    test = board[(board["split"] == "test")
                 & (board["selection"].isin([PRIMARY, "untuned",
                                             "composed"]))].copy()

    protocol_gap = test.pivot_table(
        index=["task", "model"], columns="protocol",
        values=["weighted_f1", "macro_f1", "quadratic_kappa"])
    protocol_gap.columns = [f"{m}_{p}" for m, p in protocol_gap.columns]
    for metric in ("weighted_f1", "macro_f1", "quadratic_kappa"):
        a, b = f"{metric}_random", f"{metric}_doi"
        if a in protocol_gap and b in protocol_gap:
            protocol_gap[f"{metric}_drop"] = protocol_gap[a] - protocol_gap[b]
    protocol_gap = protocol_gap.reset_index()

    pc, cm = detail(preds, board)

    out = TABLES / f"tuned_benchmark{args.tag}.xlsx"
    with pd.ExcelWriter(out) as xl:
        board.round(4).to_excel(xl, sheet_name="leaderboard", index=False)
        test.round(4).to_excel(xl, sheet_name="test", index=False)
        gap.round(4).to_excel(xl, sheet_name="overfitting", index=False)
        protocol_gap.round(4).to_excel(xl, sheet_name="protocol_gap",
                                       index=False)
        if len(pc):
            pc.round(4).to_excel(xl, sheet_name="per_class", index=False)
        if len(cm):
            cm.to_excel(xl, sheet_name="confusion", index=False)

    pd.set_option("display.width", 240)
    cols = ms.metric_columns()
    for task in sorted(test["task"].unique()):
        for protocol in ("random", "doi"):
            cell = test[(test["task"] == task)
                        & (test["protocol"] == protocol)]
            if cell.empty:
                continue
            show = cell.nlargest(8, "weighted_f1")
            base = cell[cell["model"] == "Dummy (majority)"]
            show = pd.concat([show, base]).drop_duplicates("model")
            view = show[["model"] + cols].copy()
            view.columns = ["model"] + ms.metric_headers()
            print(f"\n=== {task} / {protocol} (test) ===")
            print(view.round(3).to_string(index=False))

    print(f"\n-> {TABLES / 'tuned_benchmark.xlsx'}")


if __name__ == "__main__":
    main()
