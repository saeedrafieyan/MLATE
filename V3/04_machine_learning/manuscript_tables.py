"""
Publication tables for the supervised section
=============================================

    python 04_machine_learning/manuscript_tables.py

Emits Tables S11-S14 in the form the manuscript refers to, with the metric
column order fixed by figure_detail.METRIC_ORDER so that the tables and the
figures cannot disagree about how metrics are arranged.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from mlate import config as cfg

TABLES = cfg.step_dir("04_machine_learning", "tables")
BENCH = TABLES / "model_benchmark.xlsx"

# Import the sibling module by path: the folder is not a package.
_spec = importlib.util.spec_from_file_location(
    "figure_detail", Path(__file__).resolve().parent / "figure_detail.py")
_fd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fd)

PRETTY = dict(_fd.METRIC_ORDER)
EXTRA = {"log_loss": "Log loss", "brier_multiclass": "Brier",
         "macro_f1_lo": "Macro F1 lo", "macro_f1_hi": "Macro F1 hi"}
TASK_NAME = {"printability": "Printability",
             "cell_response": "Cell Response (5 classes)",
             "cell_response_cellular": "Cell Response (cellular only)"}
PROTO_NAME = {"random": "Random", "doi": "Study-grouped (DOI)",
              "tissue": "Leave-one-tissue-out"}


def main() -> None:
    board = pd.read_excel(BENCH, sheet_name="leaderboard")
    order = [k for k, _ in _fd.METRIC_ORDER] + list(EXTRA)

    s11 = board[["task", "protocol", "model", "family", "n_scored"] + order].copy()
    s11["task"] = s11["task"].map(TASK_NAME)
    s11["protocol"] = s11["protocol"].map(PROTO_NAME)
    s11 = s11.rename(columns={**PRETTY, **EXTRA, "task": "Task",
                              "protocol": "Protocol", "model": "Model",
                              "family": "Family", "n_scored": "n"})
    for c in s11.columns:
        if s11[c].dtype.kind == "f":
            s11[c] = s11[c].round(3)

    s12 = pd.read_excel(BENCH, sheet_name="per_class")
    conf = pd.read_excel(BENCH, sheet_name="confusion")
    s13 = pd.read_excel(BENCH, sheet_name="protocol_gap")
    s14 = pd.read_excel(BENCH, sheet_name="unseen_material")

    # One workbook per protocol, each carrying all 33 models against all three
    # tasks, plus a combined workbook. The manuscript refers to the protocols
    # separately, so they are separate files rather than sheets a reader has to
    # find.
    files = {"random": "TableS11_random_split.xlsx",
             "doi": "TableS12_study_grouped.xlsx",
             "tissue": "TableS13_leave_one_tissue_out.xlsx"}
    for proto, fname in files.items():
        part = s11[s11["Protocol"] == PROTO_NAME[proto]]
        if part.empty:
            continue
        with pd.ExcelWriter(TABLES / fname) as xl:
            (part.sort_values(["Task", "Macro F1"], ascending=[True, False])
             .to_excel(xl, sheet_name="all_models", index=False))
            for task in part["Task"].unique():
                (part[part["Task"] == task]
                 .sort_values("Macro F1", ascending=False)
                 .drop(columns=["Task", "Protocol"])
                 .to_excel(xl, sheet_name=task[:28].replace("(", "")
                           .replace(")", "").strip(), index=False))
        print(f"  {fname:44s} {len(part):3d} rows "
              f"({part['Model'].nunique()} models x {part['Task'].nunique()} tasks)")

    with pd.ExcelWriter(TABLES / "TableS11_classification_benchmark.xlsx") as xl:
        s11.to_excel(xl, sheet_name="all_models", index=False)
        for proto, label in PROTO_NAME.items():
            (s11[s11["Protocol"] == label]
             .sort_values(["Task", "Macro F1"], ascending=[True, False])
             .to_excel(xl, sheet_name=proto[:28], index=False))
    with pd.ExcelWriter(TABLES / "TableS16_per_class.xlsx") as xl:
        s12.round(3).to_excel(xl, sheet_name="per_class", index=False)
        conf.to_excel(xl, sheet_name="confusion", index=False)
    s13.round(3).to_excel(TABLES / "TableS14_protocol_gap.xlsx", index=False)
    s14.round(3).to_excel(TABLES / "TableS15_unseen_materials.xlsx", index=False)

    print(f"Table S11  {len(s11):4d} rows  ({s11['Model'].nunique()} models x "
          f"{s11['Task'].nunique()} tasks x {s11['Protocol'].nunique()} protocols)")
    print(f"Table S12  {len(s12):4d} per-class rows, {len(conf)} confusion rows")
    print(f"Table S13  {len(s13):4d} rows")
    print(f"Table S14  {len(s14):4d} rows")
    print(f"\n-> {TABLES}")


if __name__ == "__main__":
    main()
