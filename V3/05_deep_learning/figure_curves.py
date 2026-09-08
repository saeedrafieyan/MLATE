from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from mlate import config as cfg
from mlate import style as ms

warnings.filterwarnings("ignore")

TABLES = cfg.step_dir("05_deep_learning", "tables")
TUNING = TABLES / "tuning"
PRIMARY = "weighted"
TASK_TITLE = {"printability": "Printability",
              "cell_response": "Cell Response"}
ORDER = ("MLP", "ResNet", "1D_CNN", "FT_Transformer", "TabNet_Lite",
         "NODE_Lite")


def load() -> pd.DataFrame:
    files = sorted(TUNING.glob("*/history/*.parquet"))
    if not files:
        raise SystemExit(f"no epoch histories under {TUNING}; run "
                         f"tuning_dl.py first")
    hist = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return hist[hist["selection"] == PRIMARY]


def main() -> None:
    ms.apply()
    hist = load()
    print(f"{len(hist):,} epoch records | "
          f"{hist['model'].nunique()} architectures")

    for task in ("printability", "cell_response"):
        block = hist[hist["task"] == task]
        if block.empty:
            continue
        models = [m for m in ORDER if m in set(block["model"])]
        fig, axes = ms.plt.subplots(
            2, 3, figsize=(ms.WIDTHS["double"], 4.3), sharex=False)
        for ax, model in zip(axes.ravel(), models):
            for proto in ("random", "doi"):
                cur = block[(block["model"] == model)
                            & (block["protocol"] == proto)]
                if cur.empty:
                    continue
                colour = ms.PROTOCOL_COLORS[proto]
                ax.plot(cur["epoch"], cur["train_loss"], color=colour,
                        lw=1.0, zorder=3)
                ax.plot(cur["epoch"], cur["val_loss"], color=colour, lw=1.0,
                        ls=(0, (2.5, 1.6)), zorder=3)
                best = cur.loc[cur["val_macro_f1"].idxmax()]
                ax.scatter([best["epoch"]], [best["val_loss"]], s=13,
                           facecolor="white", edgecolor=colour, lw=0.9,
                           zorder=5)
            ax.set_title(ms.short(model), size=7, color=ms.TEXT, pad=4)
            ax.set_xlabel("epoch", size=6.5)
            ax.set_ylabel("loss", size=6.5)
            ax.tick_params(labelsize=5.8)
            ms.grid_axis(ax, "y")
            ms.despine(ax)
        for ax in axes.ravel()[len(models):]:
            ax.set_axis_off()

        handles = [
            ms.plt.Line2D([], [], color=ms.PROTOCOL_COLORS["random"], lw=1.0,
                          label="random split"),
            ms.plt.Line2D([], [], color=ms.PROTOCOL_COLORS["doi"], lw=1.0,
                          label="study-grouped (DOI)"),
            ms.plt.Line2D([], [], color=ms.MUTED, lw=1.0, label="training"),
            ms.plt.Line2D([], [], color=ms.MUTED, lw=1.0,
                          ls=(0, (2.5, 1.6)), label="validation"),
            ms.plt.Line2D([], [], marker="o", ls="none", markersize=3.6,
                          markerfacecolor="white", markeredgecolor=ms.MUTED,
                          label="selected epoch"),
        ]
        fig.legend(handles=handles, loc="lower center",
                   bbox_to_anchor=(0.5, 0.002), ncol=5, frameon=False,
                   fontsize=6.2, handletextpad=0.5, columnspacing=1.5)
        fig.suptitle(f"{TASK_TITLE[task]} - training and validation loss",
                     size=9.5, y=0.995, va="top")
        fig.subplots_adjust(left=0.075, right=0.985, top=0.885, bottom=0.145,
                            hspace=0.62, wspace=0.30)
        paths = ms.save(fig, f"figS14_dl_curves_{task}",
                        step="05_deep_learning")
        print(f"  {task:16s} {len(models)} panels -> "
              f"{', '.join(p.name for p in paths)}")


if __name__ == "__main__":
    main()
