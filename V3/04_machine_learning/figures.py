from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from mlate import config as cfg
from mlate import panels
from mlate import style as ms

warnings.filterwarnings("ignore")

TABLES = cfg.step_dir("04_machine_learning", "tables")
BENCH = TABLES / "tuned_benchmark.xlsx"
PRIMARY = ("weighted", "untuned", "composed")

TASK_TITLE = {"printability": "Printability",
              "cell_response": "Cell Response",
              "cell_response_cellular": "Cell Response, cellular only"}
TASK_DETAIL = {
    "printability": "4 ordinal classes; 2,116 train / 530 test (random), "
                    "2,072 / 574 (study-grouped, 45 held-out studies)",
    "cell_response": "5 ordinal classes; 2,116 train / 530 test (random), "
                     "2,072 / 574 (study-grouped, 45 held-out studies)",
    "cell_response_cellular":
        "4 ordinal classes, class 1 (no cells) excluded; 821 train / 206 test "
        "(random), 804 / 223 (study-grouped, 40 held-out studies)",
}
BASELINE_MODEL = "Dummy (majority)"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench", default="tuned_benchmark.xlsx",
                    help="workbook to read, relative to the tables directory")
    ap.add_argument("--tasks", nargs="*",
                    default=["printability", "cell_response"])
    args = ap.parse_args()

    ms.apply()
    bench = TABLES / args.bench
    if not bench.exists():
        raise SystemExit(f"{bench} not found; run report.py first")
    board = pd.read_excel(bench, sheet_name="leaderboard")
    board = board[(board["split"] == "test")
                  & (board["selection"].isin(PRIMARY))]

    for task in args.tasks:
        cell = board[board["task"] == task]
        if cell.empty:
            continue
        base = {r.protocol: r.weighted_f1
                for r in cell[cell["model"] == BASELINE_MODEL].itertuples()}
        ranked = cell[~cell["model"].str.startswith("Dummy")]
        families = ranked.drop_duplicates("model").set_index("model")["family"]

        fig = panels.benchmark_panel(
            ranked, families=families, baselines=base,
            annotate_heat=True, row_height=0.185,
            title=f"{TASK_TITLE[task]} - conventional machine learning",
            subtitle=TASK_DETAIL[task])
        paths = ms.save(fig, f"fig5_ml_benchmark_{task}",
                        step="04_machine_learning")
        print(f"  {task:16s} {ranked['model'].nunique():2d} models -> "
              f"{', '.join(p.name for p in paths)}")


if __name__ == "__main__":
    main()
