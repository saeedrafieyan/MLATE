"""
Which clustering algorithm, and why (Figure S16)
================================================

    python 03_clustering/figure_algorithms.py

The algorithm and the number of clusters were chosen together, not in sequence:
selecting k inside an algorithm picked on other grounds would beg the question,
since a different algorithm can prefer a different k. Every criterion was
therefore computed for all five fixed-k algorithms over k = 2..20 - 95
configurations - and the winner is the single best cell of that grid.

  A  prediction strength against k, one line per algorithm. This is the
     criterion that decided the analysis. Only four cells in the grid reach the
     0.80 threshold and three of those are the trivial two-way split; the
     reported partition, bisecting k-means at k = 4, is the only configuration
     at k > 3 anywhere to exceed 0.75, and is a local maximum within its own
     algorithm. The circle marks the grid's single highest value, k-means at
     k = 3, which is NOT the reported partition - select_k.py and the Methods
     give the reasons.
  B  PAC against k. Lower is better. It is computed from an independent
     procedure - consensus over resamples rather than prediction on held-out
     halves - so agreement between the two panels is not one statistic
     restated.
  C  the verdict as a ranking: each algorithm's best prediction strength
     anywhere in its own sweep.
  D  HDBSCAN, which is reported but not ranked against the others. It selects
     its own number of clusters and assigns low-density points to a noise
     class, so its high silhouette is not comparable: it achieves 0.63-0.68
     while discarding 41-79% of the corpus.
"""

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

TABLES = cfg.step_dir("03_clustering", "tables")
PS_THRESHOLD = 0.80
N_ROWS = 2646


def main() -> None:
    ms.apply()
    sel = pd.read_excel(TABLES / "select_k_raw.xlsx")
    sweep = pd.read_excel(TABLES / "clustering_sweep.xlsx", sheet_name="sweep")
    hdb = sweep[sweep["algorithm"] == "HDBSCAN"].sort_values("k")

    order = (sel.groupby("algorithm")["prediction_strength"].max()
             .sort_values(ascending=False))
    winner = order.index[0]
    best = sel.loc[sel["prediction_strength"].idxmax()]
    colours = {a: ms.CATEGORICAL[i % len(ms.CATEGORICAL)]
               for i, a in enumerate(order.index)}

    fig = ms.plt.figure(figsize=(ms.WIDTHS["double"], 5.6))
    gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.34)

    def curves(ax, column, ylabel, better):
        for alg in order.index:
            g = sel[sel["algorithm"] == alg].sort_values("k")
            win = alg == winner
            ax.plot(g["k"], g[column], marker="o", ms=2.6,
                    lw=1.9 if win else 0.95, color=colours[alg],
                    alpha=1.0 if win else 0.55, label=alg,
                    zorder=4 if win else 2)
        ax.set_xlabel("number of clusters, $k$")
        ax.set_ylabel(f"{ylabel}  ({better})")
        ax.set_xticks(list(range(2, 21, 2)))
        ms.grid_axis(ax, "y")

    # ── A. the deciding criterion ───────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    curves(ax, "prediction_strength", "prediction strength", "higher better")
    ax.axhline(PS_THRESHOLD, color=ms.RUST, lw=0.9, ls=(0, (2, 2)), zorder=1)
    ax.text(20, PS_THRESHOLD - 0.03, "threshold 0.80", ha="right", va="top",
            fontsize=5.8, color=ms.RUST, path_effects=ms.halo(1.8))
    ax.scatter([best.k], [best.prediction_strength], s=60, facecolor="none",
               edgecolor=ms.TEXT, lw=1.2, zorder=6)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, fontsize=5.6)
    ms.panel_tag(ax, "A", dx=-0.17, dy=1.30)

    # ── B. the independent confirmation ─────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    curves(ax, "pac", "PAC", "lower better")
    ax.scatter([best.k], [best.pac], s=60, facecolor="none",
               edgecolor=ms.TEXT, lw=1.2, zorder=6)
    ms.panel_tag(ax, "B", dx=-0.17, dy=1.30)
    ax.set_title("an independent criterion, agreeing on the same cell",
                 size=6.4, color=ms.MUTED, pad=4)

    # ── C. the verdict ──────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    y = np.arange(len(order))
    ax.barh(y, order.to_numpy(), height=0.6,
            color=[colours[a] for a in order.index])
    for yi, alg in zip(y, order.index):
        row = sel.loc[sel[sel["algorithm"] == alg]
                      ["prediction_strength"].idxmax()]
        ax.text(order[alg] + 0.012, yi,
                f"{order[alg]:.3f}  (at $k$ = {int(row.k)})",
                va="center", size=5.6, color=ms.MUTED)
    ax.axvline(PS_THRESHOLD, color=ms.RUST, lw=0.9, ls=(0, (2, 2)), zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(order.index, size=6.0)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.18)
    ax.set_xlabel("best prediction strength anywhere in its sweep")
    ms.grid_axis(ax, "x")
    ms.panel_tag(ax, "C", dx=-0.52, dy=1.13)

    # ── D. HDBSCAN, reported but not ranked ─────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    noise = 100 * hdb["n_noise"] / N_ROWS
    ax.plot(hdb["k"], noise, marker="o", ms=3.4, lw=1.6, color=ms.RUST,
            label="% discarded as noise (left axis)")
    ax.set_xlabel("HDBSCAN minimum cluster size")
    ax.set_ylabel("% of samples discarded", color=ms.RUST)
    ax.set_ylim(0, 100)
    ax.tick_params(axis="y", colors=ms.RUST)
    for x, v, n in zip(hdb["k"], noise, hdb["n_clusters"]):
        ax.annotate(f"{int(n)} clusters", xy=(x, v), xytext=(0, -11),
                    textcoords="offset points", ha="center", fontsize=5.4,
                    color=ms.MUTED)
    twin = ax.twinx()
    twin.plot(hdb["k"], hdb["silhouette"], marker="s", ms=3.0, lw=1.4,
              color=ms.NAVY, ls=(0, (3, 2)),
              label="silhouette (right axis)")
    twin.set_ylabel("silhouette", color=ms.NAVY)
    twin.tick_params(axis="y", colors=ms.NAVY)
    twin.set_ylim(0, 1.0)
    # Two axes with two series needs a key: colouring the axis labels tells a
    # reader which scale belongs to which line, but not what either line is.
    handles = ax.get_legend_handles_labels()[0] +         twin.get_legend_handles_labels()[0]
    labels_d = ax.get_legend_handles_labels()[1] +         twin.get_legend_handles_labels()[1]
    ax.legend(handles, labels_d, loc="lower right", fontsize=5.6,
              frameon=True, framealpha=0.9, edgecolor="none")
    ms.grid_axis(ax, "y")
    ms.panel_tag(ax, "D", dx=-0.20, dy=1.13)
    ax.set_title("high silhouette bought by discarding most of the corpus",
                 size=6.4, color=ms.MUTED, pad=4)

    fig.suptitle("Algorithm and cluster count were selected together, over "
                 "95 configurations", x=0.008, ha="left", fontsize=9.6,
                 y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    paths = ms.save(fig, "figS16_algorithm_selection", step="03_clustering")

    print(f"winner: {winner} k={int(best.k)}  "
          f"PS {best.prediction_strength:.3f}  PAC {best.pac:.3f}")
    print(order.round(3).to_string())
    print("  " + ", ".join(p.name for p in paths))


if __name__ == "__main__":
    main()
