"""
What is in each cluster
=======================

    python 03_clustering/characterise.py

Turns the partition into something a reader can name. For each cluster:

  composition    the biomaterials most over-represented relative to the corpus,
                 by prevalence ratio rather than raw prevalence - alginate is
                 common everywhere, so listing the most frequent materials would
                 return the same list for every cluster.
  context        tissue, cell line and cellular/acellular breakdown.
  process        median printing parameters, computed on reported values only so
                 that a cluster's median is not just the global fill value.
  outcome        Printability and Cell Response distributions.

Outcome columns are described here, never used to form the clusters.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from mlate import config as cfg
from mlate.dataset import load_dataset, load_taxonomy, modeling_tissue

TABLES = cfg.step_dir("03_clustering", "tables")
TOP_N = 6
MIN_PREVALENCE = 0.10      # ignore materials present in <10% of a cluster


def main() -> None:
    # The revision reports two partitions - k-means k=3 and the nested
    # bisecting k-means k=4 - so every artefact this script writes has to be
    # addressable. Without a tag the second profile silently overwrites the
    # first and the manuscript ends up quoting one partition's numbers under
    # the other's name.
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="",
                    help="suffix identifying the partition, e.g. _k4")
    args = ap.parse_args()
    tag = args.tag
    df, columns = load_dataset()
    assignments = pd.read_parquet(TABLES / f"cluster_assignments{tag}.parquet")
    df = df.join(assignments)
    tax = load_taxonomy().set_index("column")["material_class"]
    tissue = modeling_tissue(df)

    bio = df[columns.biomaterials].fillna(0) > 0
    overall = bio.mean()

    summary, markers = [], []
    for c, rows in df.groupby("cluster"):
        idx = rows.index
        present = bio.loc[idx].mean()

        # Enrichment against the corpus, restricted to materials that actually
        # appear in this cluster often enough to describe it.
        ratio = (present / overall.replace(0, np.nan)).replace(
            [np.inf, -np.inf], np.nan)
        ratio = ratio[present >= MIN_PREVALENCE].dropna().nlargest(TOP_N)
        for material, r in ratio.items():
            markers.append({
                "cluster": c, "biomaterial": material,
                "material_class": tax.get(material, "unassigned"),
                "prevalence_in_cluster": 100 * present[material],
                "prevalence_overall": 100 * overall[material],
                "enrichment": r,
            })

        cellular = rows[columns.cell_line] != cfg.ACELLULAR_TOKEN
        reported = {c_: rows[c_].dropna() for c_ in columns.print_params}
        summary.append({
            "cluster": c,
            "n": len(rows),
            "pct_of_corpus": 100 * len(rows) / len(df),
            "n_studies": rows["DOI"].nunique(),
            "top_tissue": tissue.loc[idx].value_counts().idxmax(),
            "pct_top_tissue": 100 * tissue.loc[idx].value_counts().iloc[0] / len(rows),
            "pct_cellular": 100 * cellular.mean(),
            "top_cell_line": (rows.loc[cellular, columns.cell_line]
                              .value_counts().idxmax() if cellular.any() else "-"),
            "top_materials": ", ".join(ratio.index[:3]),
            "median_printability": float(rows["Printability"].median()),
            "mean_printability": float(rows["Printability"].mean()),
            "median_cell_response": float(rows.loc[cellular, "Cell Response"].median())
                                    if cellular.any() else np.nan,
            **{f"median_{c_}": (float(reported[c_].median())
                                if len(reported[c_]) else np.nan)
               for c_ in columns.print_params},
            **{f"n_reported_{c_}": int(len(reported[c_]))
               for c_ in columns.print_params},
        })

    summary = pd.DataFrame(summary)
    markers = pd.DataFrame(markers)

    # Target distribution as counts, for the stacked bars in the figure.
    dist = (pd.crosstab(df["cluster"], df["Printability"])
            .rename(columns=lambda v: f"printability_{v}"))
    cell = df[df[columns.cell_line] != cfg.ACELLULAR_TOKEN]
    dist = dist.join(pd.crosstab(cell["cluster"], cell["Cell Response"])
                     .rename(columns=lambda v: f"cell_response_{v}")).fillna(0)

    # Prevalence of every material in every cluster, so a figure can show a
    # chosen material across all clusters instead of only where it ranked.
    prevalence = (bio.groupby(df["cluster"]).mean().T * 100)
    prevalence.index.name = "biomaterial"

    with pd.ExcelWriter(TABLES / f"cluster_profiles{tag}.xlsx") as xl:
        summary.to_excel(xl, sheet_name="summary", index=False)
        prevalence.reset_index().to_excel(xl, sheet_name="prevalence_matrix",
                                          index=False)
        markers.to_excel(xl, sheet_name="enriched_materials", index=False)
        dist.reset_index().to_excel(xl, sheet_name="target_distribution",
                                    index=False)

    pd.set_option("display.width", 250)
    cols = ["cluster", "n", "pct_of_corpus", "n_studies", "top_tissue",
            "pct_cellular", "top_materials", "mean_printability"]
    print(summary[cols].to_string(index=False,
                                  float_format=lambda v: f"{v:6.1f}"))
    print(f"\n-> {TABLES / 'cluster_profiles.xlsx'}")


if __name__ == "__main__":
    main()
