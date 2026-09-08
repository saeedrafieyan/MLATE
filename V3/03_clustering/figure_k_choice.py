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

    shared = cl.reference_plane(X)

    fig, axes = ms.figure(width="double", height=5.2, nrows=2, ncols=3)
    axes = axes.ravel()
    rows = []

    for ax, k in zip(axes, K_VALUES):
        lab = make(k).fit_predict(X)
        emb = shared
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
