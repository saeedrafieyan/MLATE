from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from mlate import config as cfg
from mlate import style as ms

TABLES = cfg.step_dir("03_clustering", "tables")


def reporting_flag_ami(matrix, doi, k: int) -> tuple[float, float]:
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_mutual_info_score

    flags = [c for c in matrix.columns if "[reported]" in str(c)]
    out = []
    for cols in (matrix[flags], matrix.drop(columns=flags)):
        lab = KMeans(n_clusters=k, n_init=20,
                     random_state=cfg.RANDOM_STATE).fit_predict(
            cols.to_numpy(dtype=float))
        out.append(float(adjusted_mutual_info_score(doi, lab)))
    return out[0], out[1]


def main() -> None:
    ms.apply()
    curve = pd.read_excel(TABLES / "null_model_curve.xlsx")
    agree = pd.read_excel(TABLES / "clustering_diagnostics.xlsx",
                          sheet_name="external_agreement")
    sens = pd.read_excel(TABLES / "clustering_diagnostics.xlsx",
                         sheet_name="temperature_sensitivity")

    import joblib
    from mlate.dataset import load_dataset
    matrix = pd.read_parquet(cfg.step_dir("02_preprocessing", "tables")
                             / "feature_matrix.parquet")
    bundle = joblib.load(cfg.step_dir("03_clustering", "models")
                         / "clustering.pkl")
    doi = load_dataset()[0]["DOI"].astype(str).to_numpy()

    fig = ms.plt.figure(figsize=(ms.WIDTHS["double"], 5.6))
    gs = fig.add_gridspec(2, 2, hspace=0.52, wspace=0.40)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(curve["k"], curve["silhouette"], marker="o", ms=2.8, lw=1.5,
            color=ms.NAVY, label="observed")
    ax.plot(curve["k"], curve["silhouette_null"], marker="o", ms=2.4, lw=1.2,
            color=ms.SLATE, ls="--", label="column-permuted null")
    ax.fill_between(curve["k"], curve["silhouette_null"], curve["silhouette"],
                    color=ms.NAVY, alpha=0.10, lw=0)
    ax.set_xscale("log")
    ax.set_xticks(list(curve["k"]))
    ax.set_xticklabels([str(k) for k in curve["k"]], size=5.4)
    ax.minorticks_off()
    ax.set_xlabel("number of clusters, $k$")
    ax.set_ylabel("silhouette")
    ms.grid_axis(ax, "y")
    ax.legend(loc="upper left", fontsize=5.8)
    ms.panel_tag(ax, "A", dx=-0.17, dy=1.13)
    ax.set_title("structure is real, but has no natural $k$",
                 size=6.6, color=ms.MUTED, pad=4)

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(curve["k"], curve["AMI_DOI"], marker="o", ms=2.8, lw=1.5,
            color=ms.RUST, label="source publication (DOI)")
    ax.plot(curve["k"], curve["AMI_tissue"], marker="o", ms=2.8, lw=1.5,
            color=ms.TEAL, label="target tissue")

    flags_only, no_flags = reporting_flag_ami(matrix, doi, int(bundle["k"]))
    ax.scatter([bundle["k"]], [flags_only], s=30, marker="^", color=ms.RUST,
               zorder=5, label="DOI, the 7 reporting flags alone")
    ax.scatter([bundle["k"]], [no_flags], s=30, marker="v", color=ms.RUST,
               zorder=5, alpha=0.55, label="DOI, those flags removed")
    print(f"  reporting flags at k={bundle['k']}: alone {flags_only:.3f}, "
          f"removed {no_flags:.3f}")
    ax.set_xscale("log")
    ax.set_xticks(list(curve["k"]))
    ax.set_xticklabels([str(k) for k in curve["k"]], size=5.4)
    ax.minorticks_off()
    ax.set_xlabel("number of clusters, $k$")
    ax.set_ylabel("adjusted mutual information")
    ax.set_ylim(0, max(0.8, float(curve["AMI_DOI"].max()) * 1.18))
    ms.grid_axis(ax, "y")
    ax.legend(loc="upper left", fontsize=5.8)
    ms.panel_tag(ax, "B", dx=-0.17, dy=1.13)
    ax.set_title("finer partitions resolve publications, not tissues",
                 size=6.6, color=ms.MUTED, pad=4)

    ax = fig.add_subplot(gs[1, 0])
    metrics, colours = ["AMI", "ARI", "purity"], [ms.NAVY, ms.TEAL, ms.GOLD]
    y = np.arange(len(agree))
    h = 0.26
    for i, (metric, colour) in enumerate(zip(metrics, colours)):
        off = (i - 1) * h
        ax.barh(y + off, agree[metric], height=h, color=colour, label=metric)
        for yi, v in zip(y + off, agree[metric]):
            ax.text(v + 0.014, yi, f"{v:.2f}", va="center", size=5.0,
                    color=ms.MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{n}\n({c} categories)" for n, c in
                        zip(agree["external_label"], agree["n_categories"])],
                       size=5.4)
    ax.invert_yaxis()
    ax.set_xlim(0, min(1.05, float(agree[metrics].to_numpy().max()) * 1.32))
    ax.set_xlabel("agreement with the selected partition")
    ms.grid_axis(ax, "x")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=3, fontsize=5.8)
    ms.panel_tag(ax, "C", dx=-0.46, dy=1.15)

    ax = fig.add_subplot(gs[1, 1])
    y = np.arange(len(sens))
    colours = [ms.SLATE, ms.SLATE, ms.RUST]
    ax.barh(y, sens["ARI"], height=0.52, color=colours)
    for yi, row in zip(y, sens.itertuples()):
        edge = row.ARI_max if np.isfinite(row.ARI_max) else row.ARI
        ax.text(edge + 0.022, yi, f"{row.ARI:.2f}", va="center", size=5.8,
                color=ms.MUTED)
        if np.isfinite(row.ARI_min):
            ax.plot([row.ARI_min, row.ARI_max], [yi, yi], color=ms.TEXT,
                    lw=0.9, solid_capstyle="butt", zorder=4)
            for e in (row.ARI_min, row.ARI_max):
                ax.plot([e, e], [yi - 0.11, yi + 0.11], color=ms.TEXT, lw=0.9,
                        zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(sens["comparison"], size=5.6)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("ARI with the selected partition")
    ms.grid_axis(ax, "x")
    ms.panel_tag(ax, "D", dx=-0.52, dy=1.15)
    ax.set_title("bars show the range over repeats",
                 size=6.0, color=ms.MUTED, pad=4)

    paths = ms.save(fig, "figS8_clustering_diagnostics", step="03_clustering")
    print("written:", *[p.name for p in paths])


if __name__ == "__main__":
    main()
