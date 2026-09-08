from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from mlate import config as cfg
from mlate import deep
from mlate import panels
from mlate import style as ms

warnings.filterwarnings("ignore")

TABLES = cfg.step_dir("05_deep_learning", "tables")
DL = TABLES / "dl_benchmark.xlsx"
FOUNDATION = TABLES / "foundation_models.xlsx"

TASK_TITLE = {
    "printability": "Printability",
    "cell_response": "Cell Response",
    "cell_response_cellular": "Cell Response, cellular only",
}
TASK_DETAIL = {
    "printability": "4 ordinal classes; 2,116 train / 530 test (random), "
                    "2,072 / 574 (study-grouped)",
    "cell_response": "5 ordinal classes; 2,116 train / 530 test (random), "
                     "2,072 / 574 (study-grouped)",
    "cell_response_cellular":
        "4 ordinal classes, class 1 (no cells) excluded; 821 train / 206 test "
        "(random), 804 / 223 (study-grouped)",
}
PRIMARY = ("weighted", "zero_shot")

FOUNDATION_FAMILY = "foundation"


def load(dl_name: str, fnd_name: str) -> pd.DataFrame:
    frames = []
    dl_path = TABLES / dl_name
    fnd_path = TABLES / fnd_name
    if dl_path.exists():
        d = pd.read_excel(dl_path)
        d["family"] = d["model"].map(
            lambda m: deep.REGISTRY[m].family if m in deep.REGISTRY
            else "neural")
        frames.append(d)
    if fnd_path.exists():
        f = pd.read_excel(fnd_path, sheet_name="pooled")
        f["family"] = FOUNDATION_FAMILY
        frames.append(f)
    if not frames:
        raise SystemExit("no step-05 results; run tuning_dl.py and "
                         "foundation_models.py first")
    board = pd.concat(frames, ignore_index=True)
    board = board[board["split"] == "test"]
    return board[board["selection"].isin(PRIMARY)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dl", default="dl_benchmark.xlsx")
    ap.add_argument("--foundation", default="foundation_models.xlsx")
    ap.add_argument("--tasks", nargs="*",
                    default=["printability", "cell_response"])
    args = ap.parse_args()

    ms.apply()
    board = load(args.dl, args.foundation)
    print(f"{board['model'].nunique()} models | "
          f"{sorted(board['family'].unique())}")

    for task in args.tasks:
        cell = board[board["task"] == task]
        if cell.empty:
            continue
        families = cell.drop_duplicates("model").set_index("model")["family"]
        fig = panels.benchmark_panel(
            cell, families=families, annotate_heat=True, row_height=0.30,
            title=f"{TASK_TITLE[task]} - deep learning and foundation models",
            subtitle=TASK_DETAIL[task])
        paths = ms.save(fig, f"fig7_dl_foundation_{task}",
                        step="05_deep_learning")
        print(f"  {task:16s} -> {', '.join(p.name for p in paths)}")


if __name__ == "__main__":
    main()
