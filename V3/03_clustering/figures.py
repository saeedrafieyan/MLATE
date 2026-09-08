from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd

from mlate import clustering as cl
from mlate import config as cfg
from mlate import style as ms

TABLES = cfg.step_dir("03_clustering", "tables")
MODELS = cfg.step_dir("03_clustering", "models")
MATRIX = cfg.step_dir("02_preprocessing", "tables") / "feature_matrix.parquet"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="", help="partition suffix, e.g. _k4")
    ap.add_argument("--name", default="fig4_clustering",
                    help="output figure name")
    args = ap.parse_args()
    tag = args.tag

    ms.apply()
    matrix = pd.read_parquet(MATRIX)
    sweep = pd.read_excel(TABLES / f"clustering_sweep{tag}.xlsx", sheet_name="sweep")
    profiles = pd.read_excel(TABLES / f"cluster_profiles{tag}.xlsx",
                             sheet_name="summary")
    markers = pd.read_excel(TABLES / f"cluster_profiles{tag}.xlsx",
                            sheet_name="enriched_materials")
    dist = pd.read_excel(TABLES / f"cluster_profiles{tag}.xlsx",
                         sheet_name="target_distribution")
    labels = pd.read_parquet(TABLES / f"cluster_assignments{tag}.parquet")["cluster"]
    bundle = joblib.load(MODELS / f"clustering{tag}.pkl")

    X = matrix.to_numpy(dtype=float)
    if bundle.get("space") == "pca":
        X, _ = cl.reduce(X)
    clusters = sorted(labels.unique())
    palette = {c: ms.CATEGORICAL[i % len(ms.CATEGORICAL)]
               for i, c in enumerate(clusters)}

    fig = ms.plt.figure(figsize=(ms.WIDTHS["double"], 7.0))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.34,
                          height_ratios=[1.0, 1.15])

    ax = fig.add_subplot(gs[0, 0])
    sel_path = TABLES / "select_k_raw.xlsx"
    sel = pd.read_excel(sel_path)
    sel = sel[sel["algorithm"] == bundle["algorithm"]].sort_values("k")

    ax.plot(sel["k"], sel["prediction_strength"], marker="o", ms=3.0, lw=1.7,
            color=ms.NAVY, label="prediction strength", zorder=3)
    ax.plot(sel["k"], 1 - sel["pac"], marker="s", ms=2.8, lw=1.4,
            color=ms.TEAL, label="1 - PAC (consensus)", zorder=3)
    ax.plot(sel["k"], sel["silhouette"], lw=1.1, ls=(0, (4, 2.5)),
            color=ms.MUTED, label="silhouette (no optimum)", zorder=2)
    ax.axhline(0.80, color=ms.RUST, lw=0.9, ls=(0, (2, 2)), zorder=1)
    ax.text(sel["k"].max(), 0.775, "threshold 0.80", ha="right", va="top",
            fontsize=5.8, color=ms.RUST, path_effects=ms.halo(1.8))

    row = sel[sel["k"] == bundle["k"]].iloc[0]
    ax.scatter([bundle["k"]], [row["prediction_strength"]], s=58,
               facecolor="none", edgecolor=ms.TEXT, lw=1.2, zorder=5)
    ax.annotate(f"$k$ = {bundle['k']}", xy=(bundle["k"],
                                            row["prediction_strength"]),
                xytext=(bundle["k"] + 1.4, row["prediction_strength"] + 0.06),
                fontsize=6.6, color=ms.TEXT,
                path_effects=ms.halo(1.8))
    ax.set_xlabel("number of clusters, $k$")
    ax.set_ylabel("criterion value")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(list(sel["k"])[::2])
    ms.grid_axis(ax, "y")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, fontsize=5.8)
    ms.panel_tag(ax, "A", dx=-0.16, dy=1.16)

    ax = fig.add_subplot(gs[0, 1])
    emb = cl.reference_plane(X)
    for c in clusters:
        m = (labels == c).to_numpy()
        ax.scatter(emb[m, 0], emb[m, 1], s=5.0, lw=0, alpha=0.70,
                   color=palette[c], label=f"C{c}")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), ncol=1,
              fontsize=5.4, markerscale=2.6, handletextpad=0.3,
              borderpad=0.2, labelspacing=0.28)
    ax.set_xlabel("discriminant 1")
    ax.set_ylabel("discriminant 2")
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)
    ms.panel_tag(ax, "B", dx=-0.10, dy=1.16)
    ax.set_title(f"{bundle['algorithm']}, k = {bundle['k']}", size=7.5,
                 color=ms.MUTED, pad=5)

    ax = fig.add_subplot(gs[1, 0])
    keep = (markers.sort_values("enrichment", ascending=False)
                   .groupby("cluster").head(3)["biomaterial"].unique())
    prevalence = pd.read_excel(TABLES / f"cluster_profiles{tag}.xlsx",
                               sheet_name="prevalence_matrix"
                               ).set_index("biomaterial")
    prevalence.columns = [int(c) for c in prevalence.columns]
    grid = prevalence.reindex(keep)[sorted(prevalence.columns)]
    v = grid.to_numpy(dtype=float)
    im = ax.imshow(v, cmap=ms.SEQ_TEAL, aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(grid.shape[1]))
    ax.set_xticklabels([f"C{c}" for c in grid.columns], size=6)
    ax.set_yticks(range(len(grid)))
    ax.set_yticklabels([b[:26] for b in grid.index], size=5.4)
    ax.grid(False)
    ms.despine(ax, keep=())
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            if np.isnan(v[i, j]):
                continue
            ax.text(j, i, f"{v[i, j]:.0f}", ha="center", va="center", size=4.8,
                    color="white" if v[i, j] > 55 else ms.TEXT)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("% of cluster containing", size=6)
    cb.ax.tick_params(labelsize=5.5)
    cb.outline.set_visible(False)
    ms.panel_tag(ax, "C", dx=-0.30, dy=1.08)

    ax = fig.add_subplot(gs[1, 1])
    pcols = [c for c in dist.columns if c.startswith("printability_")]
    d = dist.set_index("cluster")[pcols]
    d = 100 * d.div(d.sum(axis=1), axis=0)
    bottom = np.zeros(len(d))
    shades = [ms.RUST, ms.GOLD, ms.SAGE, ms.TEAL]
    for i, col in enumerate(pcols):
        w = d[col].to_numpy(dtype=float)
        ax.bar(range(len(d)), w, bottom=bottom, width=0.66,
               color=shades[i % len(shades)],
               label=f"class {col.split('_')[-1]}")
        for j, (wi, bi) in enumerate(zip(w, bottom)):
            if wi >= 9:
                ax.text(j, bi + wi / 2, f"{wi:.0f}", ha="center", va="center",
                        size=5.2, color="white")
        bottom += w
    ax.set_xticks(range(len(d)))
    ax.set_xticklabels([f"C{c}" for c in d.index], size=6.2)
    ax.set_ylabel("% of cluster")
    ax.set_ylim(0, 100)
    ms.grid_axis(ax, "y")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=4, fontsize=5.8,
              title=None)
    ms.panel_tag(ax, "D", dx=-0.14, dy=1.13)

    paths = ms.save(fig, args.name, step="03_clustering")
    print("written:", *[p.name for p in paths])


if __name__ == "__main__":
    main()
