from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from mlate import config as cfg
from mlate import style as ms

TABLES = cfg.step_dir("04_machine_learning", "tables")
BENCH = TABLES / "model_benchmark.xlsx"

TASK_TITLE = {
    "printability": "Printability\n(4 classes, n = 2,646)",
    "cell_response": "Cell Response\n(5 classes, n = 2,646)",
    "cell_response_cellular": "Cell Response, cellular\n(4 classes, n = 1,027)",
}
PROTOCOL = {
    "random": ("figS9_all_models_random", "Random 5-fold cross-validation",
               "within-corpus interpolation; 97% of test samples share a "
               "publication with the training set"),
    "doi": ("figS10_all_models_doi", "Study-grouped 5-fold cross-validation",
            "whole publications held out; the estimate that applies to a new "
            "laboratory"),
    "tissue": ("figS11_all_models_tissue", "Leave-one-tissue-out",
               "unseen tissue and unseen laboratory simultaneously"),
}

FAMILY_COLOURS = {
    "baseline": ms.SLATE,
    "linear": ms.NAVY,
    "discriminant": ms.PLUM,
    "naive_bayes": ms.TEAL,
    "neighbours": ms.SAGE,
    "svm": ms.GOLD,
    "tree": ms.CLAY,
    "bagging": ms.RUST,
    "boosting": "#1C5E72",
    "neural": "#8E6FA8",
    "meta": "#5C7A54",
}


def _short(name: str) -> str:
    return (name.replace("Logistic Regression", "Logistic Reg.")
                .replace("Discriminant", "Discrim.")
                .replace("Gradient Boosting", "Grad. Boost.")
                .replace("Naive Bayes", "NB")
                .replace("k-Nearest Neighbours", "k-NN")
                .replace(" (RF+XGB+LR -> LR)", " (stack)")
                .replace(" (RF+XGB+LR)", " (vote)")
                .replace("Dummy (majority)", "Dummy — majority")
                .replace("Dummy (stratified)", "Dummy — stratified"))


def draw(board: pd.DataFrame, protocol: str) -> None:
    cell = board[board["protocol"] == protocol]
    if cell.empty:
        return
    stem, title, subtitle = PROTOCOL[protocol]
    tasks = [t for t in TASK_TITLE if t in set(cell["task"])]

    order = (cell.groupby("model")["macro_f1"].mean()
             .sort_values(ascending=True).index.tolist())
    family = board.drop_duplicates("model").set_index("model")["family"]

    fig = ms.plt.figure(figsize=(ms.WIDTHS["double"], 8.6))
    gs = fig.add_gridspec(1, len(tasks), wspace=0.10)
    y = np.arange(len(order))

    for i, task in enumerate(tasks):
        ax = fig.add_subplot(gs[0, i])
        g = cell[cell["task"] == task].set_index("model")
        vals = g["macro_f1"].reindex(order).to_numpy(dtype=float)
        colours = [FAMILY_COLOURS.get(family.get(m, "?"), ms.SLATE)
                   for m in order]
        ax.barh(y, vals, color=colours, height=0.72)

        base = g["macro_f1"].get("Dummy (majority)", np.nan)
        if np.isfinite(base):
            ax.axvline(float(base), color=ms.RUST, lw=1.0, ls="--", zorder=3)

        hi = float(np.nanmax(vals))
        for yi, v in zip(y, vals):
            if np.isfinite(v):
                ax.text(v + hi * 0.02, yi, f"{v:.3f}", va="center", size=4.6,
                        color=ms.MUTED)
        ax.set_xlim(0, hi * 1.20)
        ax.set_ylim(-0.7, len(order) - 0.3)
        ax.set_yticks(y)
        if i == 0:
            ax.set_yticklabels([_short(m) for m in order], size=5.4)
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("macro F1", size=7)
        ms.grid_axis(ax, "x")
        ax.set_title(TASK_TITLE[task], size=6.8, color=ms.MUTED, pad=6)

    handles = [ms.plt.Line2D([0], [0], marker="s", ls="", ms=4.4,
                             markerfacecolor=c, markeredgecolor="none",
                             label=f.replace("_", " "))
               for f, c in FAMILY_COLOURS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=5.8,
               frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(title, size=9.5, y=0.985)
    fig.text(0.5, 0.962, subtitle, ha="center", size=6.4, color=ms.MUTED)
    fig.subplots_adjust(top=0.93, bottom=0.075, left=0.20, right=0.985)

    paths = ms.save(fig, stem, step="04_machine_learning")
    print(f"  {protocol:7s} -> {paths[0].name}")


def main() -> None:
    ms.apply()
    board = pd.read_excel(BENCH, sheet_name="leaderboard")
    for protocol in PROTOCOL:
        draw(board, protocol)


if __name__ == "__main__":
    main()
