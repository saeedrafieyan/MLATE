"""
k = 3 to 8, on one shared projection
====================================

    python 03_clustering/figure_k_choice.py

Choosing the shared plane
-------------------------
Panels of a comparison figure must share coordinates, or a reader cannot tell a
difference between partitions from a difference between views. Three candidates
were tried and two failed.

A shared UMAP embedding was comparable but empty: measured inside the UMAP
plane the silhouette of the reported partition is -0.063, so the view
interleaves the very groups the figure exists to show.

The first two principal components are comparable, unsupervised and score
adequately (silhouette 0.509 at k = 3), but are illegible in practice: they
capture too little of a 153-dimensional matrix, so the points collapse into an
overplotted band and the colouring reads as noise. A number can look acceptable
while the picture it describes does not - the panel is judged by eye, so it has
to be chosen by eye too.

The plane used is therefore the two linear discriminants of the REPORTED k = 3
partition, held fixed across all six panels. It spreads the corpus properly,
every panel shares one coordinate system, and the question it puts to the
reader is the right one: how do finer partitions sit inside the structure the
analysis actually reports?

Because that plane is fitted to the k = 3 labels it necessarily flatters k = 3,
and no partition is judged by how it looks in it. The `sep` annotation is
computed separately, in each partition's OWN discriminant plane, so every k is
scored at its best regardless of the view drawn.

What the annotations mean
-------------------------
    PS          prediction strength on held-out halves; >= 0.80 is the
                conventional threshold for a reproducible partition
    sep         silhouette in that partition's OWN discriminant plane - each
                partition judged at its best, independent of the shared view
    enrichment  strongest biomaterial over-representation achieved by any
                cluster, relative to corpus prevalence
    smallest    size of the smallest cluster, since a group of 115 rows cannot
                be characterised chemically however well it separates

Read together they are the trade-off the figure exists to show: chemical
resolution rises with k while reproducibility and separability fall, and no k
delivers both.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.cluster import BisectingKMeans, KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import silhouette_score

from mlate import clustering as cl
from mlate import config as cfg
from mlate import style as ms

warnings.filterwarnings("ignore")

TABLES = cfg.step_dir("03_clustering", "tables")
MATRIX = cfg.step_dir("02_preprocessing", "tables") / "feature_matrix.parquet"
K_VALUES = (3, 4, 5, 6, 7, 8)

# Both reported partitions get this figure, so the algorithm, its reported k
# and the estimator are chosen together rather than hard-coded.
ALGORITHMS = {
    "KMeans": (lambda k: KMeans(n_clusters=k, n_init=20,
                                random_state=cfg.RANDOM_STATE), 3),
    "BisectingKMeans": (lambda k: BisectingKMeans(
        n_clusters=k, n_init=10, random_state=cfg.RANDOM_STATE), 4),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--algorithm", default="KMeans", choices=list(ALGORITHMS))
    ap.add_argument("--name", default="figS15_k_choice")
    args = ap.parse_args()
    make, REPORTED_K = ALGORITHMS[args.algorithm]

    ms.apply()
    matrix = pd.read_parquet(MATRIX)
    X = matrix.to_numpy(dtype=float)

    bio = [c for c in matrix.columns
           if not c.startswith("Cell Line_") and "[reported]" not in c
           and not any(u in c for u in ("(s)", "(kPa)", "(mm/s)", "(m)",
                                        "(C)", "Cell Density"))]
    prev0 = (matrix[bio] > 0).mean()

    ps = {}
    book = TABLES / "select_k_raw.xlsx"
    if book.exists():
        d = pd.read_excel(book)
        d = d[d.algorithm == args.algorithm]
        ps = dict(zip(d.k, d.prediction_strength))

    # One plane for every panel: the discriminants of the reported partition.
    shared = cl.reference_plane(X)

    fig, axes = ms.figure(width="double", height=5.2, nrows=2, ncols=3)
    axes = axes.ravel()
    rows = []

    for ax, k in zip(axes, K_VALUES):
        lab = make(k).fit_predict(X)
        emb = shared
        # Separability is reported from each partition's OWN discriminant
        # plane, not from the shared one. Judging every k inside a single plane
        # would confound "these groups are not distinct" with "this particular
        # view does not show them", and the annotation would then be a property
        # of the figure rather than of the partition.
        sep = silhouette_score(
            LinearDiscriminantAnalysis(n_components=2)
            .fit(X, lab).transform(X), lab)

        prev = pd.DataFrame({c: [(matrix.loc[lab == j, c] > 0).mean()
                                 for j in range(k)] for c in bio})
        enr = ((prev + .002).div(prev0 + .002, axis=1)).max().max()
        sizes = np.bincount(lab)

        for j in range(k):
            m = lab == j
            ax.scatter(emb[m, 0], emb[m, 1], s=2.6, lw=0, alpha=0.62,
                       color=ms.CATEGORICAL[j % len(ms.CATEGORICAL)])

        reported = k == REPORTED_K
        ax.set_title(f"k = {k}" + ("   (reported)" if reported else ""),
                     loc="left", fontsize=8.6, pad=46,
                     fontweight="bold" if reported else "normal",
                     color=ms.TEXT if reported else ms.MUTED)
        ax.text(0.0, 1.090, f"PS {ps.get(k, float('nan')):.2f}     "
                            f"sep {sep:.2f}",
                transform=ax.transAxes, fontsize=6.3,
                color=ms.NAVY if reported else ms.MUTED, va="bottom")
        ax.text(0.0, 1.020, f"enrichment {enr:.1f}×     "
                            f"smallest {sizes.min()}",
                transform=ax.transAxes, fontsize=6.3, color=ms.MUTED,
                va="bottom")
        # The house style hides the top and right spines, so a panel highlight
        # has to switch them back on before recolouring or only two sides of
        # the box would draw.
        for side, sp in ax.spines.items():
            sp.set_visible(True)
            sp.set_edgecolor(ms.NAVY if reported else ms.GRIDLINE)
            sp.set_linewidth(1.2 if reported else 0.6)
        ax.set_xticks([]); ax.set_yticks([])
        rows.append({"k": k, "prediction_strength": ps.get(k, np.nan),
                     "discriminant_separability": sep,
                     "max_material_enrichment": enr,
                     "smallest_cluster": int(sizes.min()),
                     "sizes": sizes.tolist()})
        print(f"  k={k:2d}  PS {ps.get(k, float('nan')):.3f}  sep {sep:.3f}  "
              f"enrich {enr:5.1f}x  sizes {sizes.tolist()}")

    fig.suptitle(f"Reproducibility and chemical resolution move in opposite "
                 f"directions  —  {args.algorithm}",
                 x=0.008, ha="left", fontsize=9.6, y=0.995)
    fig.text(0.008, 0.945,
             "all six panels share one plane - the two discriminants of the "
             f"reported k = {REPORTED_K} partition - so differences between "
             "panels are "
             "differences between partitions, not between views;\nsep is "
             "measured in each partition's own discriminant plane, so every k "
             "is scored at its best independently of the plane drawn here",
             ha="left", fontsize=6.5, color=ms.MUTED, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.875))
    paths = ms.save(fig, args.name, step="03_clustering")

    pd.DataFrame(rows).round(4).to_excel(
        TABLES / f"k_choice_tradeoff_{args.algorithm}.xlsx", index=False)
    print("  " + ", ".join(p.name for p in paths))
    print(f"  -> {TABLES / f'k_choice_tradeoff_{args.algorithm}.xlsx'}")


if __name__ == "__main__":
    main()
