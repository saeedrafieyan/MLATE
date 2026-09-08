from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from mlate import config as cfg

TABLES = cfg.step_dir("03_clustering", "tables")


def table_s9(tag: str = "") -> tuple[pd.DataFrame, pd.DataFrame]:
    sweep = pd.read_excel(TABLES / f"clustering_sweep{tag}.xlsx", sheet_name="sweep")
    rename = {"algorithm": "Algorithm", "k": "k", "n_clusters": "Clusters",
              "silhouette": "SI", "davies_bouldin": "DBI",
              "calinski_harabasz": "CHI", "stability": "Stability (ARI)",
              "n_noise": "n_noise"}

    fixed = (sweep[(sweep["algorithm"] != "HDBSCAN") & (sweep["k"] <= 20)]
             .rename(columns=rename)
             [["Algorithm", "k", "SI", "DBI", "CHI", "Stability (ARI)"]]
             .sort_values(["Algorithm", "k"]))

    sel_path = TABLES / "select_k_raw.xlsx"
    if sel_path.exists():
        sel = (pd.read_excel(sel_path)
               .rename(columns={"algorithm": "Algorithm",
                                "prediction_strength": "PS", "pac": "PAC"})
               [["Algorithm", "k", "PS", "PAC"]])
        fixed = fixed.merge(sel, on=["Algorithm", "k"], how="left")
        fixed = fixed[["Algorithm", "k", "PS", "PAC", "SI", "DBI", "CHI",
                       "Stability (ARI)"]]

    for c, dp in [("SI", 3), ("DBI", 3), ("CHI", 1), ("Stability (ARI)", 3),
                  ("PS", 3), ("PAC", 3)]:
        if c in fixed.columns:
            fixed[c] = fixed[c].round(dp)

    hdb = (sweep[sweep["algorithm"] == "HDBSCAN"]
           .rename(columns={**rename, "k": "min_cluster_size"})
           [["min_cluster_size", "Clusters", "n_noise", "SI", "DBI", "CHI"]])
    hdb["% of samples as noise"] = (100 * hdb["n_noise"] / 2646).round(1)
    hdb["Stability (ARI)"] = "n/a"
    for c, dp in [("SI", 3), ("DBI", 3), ("CHI", 1)]:
        hdb[c] = hdb[c].round(dp)
    return fixed, hdb


def table_s10(tag: str = "") -> pd.DataFrame:
    prof = pd.read_excel(TABLES / f"cluster_profiles{tag}.xlsx", sheet_name="summary")
    conc = pd.read_excel(TABLES / f"clustering_diagnostics{tag}.xlsx",
                         sheet_name="study_concentration")
    mark = pd.read_excel(TABLES / f"cluster_profiles{tag}.xlsx",
                         sheet_name="enriched_materials")
    dist = pd.read_excel(TABLES / f"cluster_profiles{tag}.xlsx",
                         sheet_name="target_distribution").set_index("cluster")

    top3 = (mark.sort_values("enrichment", ascending=False)
                .groupby("cluster")
                .head(3)
                .groupby("cluster")
                .apply(lambda g: "; ".join(
                    f"{r.biomaterial.split(' (')[0]} "
                    f"({r.prevalence_in_cluster:.0f}% vs "
                    f"{r.prevalence_overall:.0f}%, {r.enrichment:.1f}x)"
                    for r in g.itertuples()), include_groups=False))

    pcols = [c for c in dist.columns if c.startswith("printability_")]
    pct3 = 100 * dist["printability_3"] / dist[pcols].sum(axis=1)

    out = pd.DataFrame({
        "Cluster": [f"C{c}" for c in prof["cluster"]],
        "n": prof["n"],
        "% of corpus": prof["pct_of_corpus"].round(1),
        "Studies": prof["n_studies"],
        "% from largest study": conc.set_index("cluster")
                                    .loc[prof["cluster"], "pct_from_largest_study"]
                                    .round(1).to_numpy(),
        "Top tissue": prof["top_tissue"],
        "% top tissue": prof["pct_top_tissue"].round(1),
        "% cellular": prof["pct_cellular"].round(1),
        "Top cell line": prof["top_cell_line"],
        "Three most enriched biomaterials": prof["cluster"].map(top3),
        "Median nozzle diameter (um)": prof["median_Nozzle Diameter (µm)"].round(0),
        "Median extrusion pressure (kPa)": prof["median_Extrusion Pressure (kPa)"].round(0),
        "Mean Printability": prof["mean_printability"].round(2),
        "% Printability class 3": pct3.loc[prof["cluster"]].round(1).to_numpy(),
        "Median Cell Response": prof["median_cell_response"],
    })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="", help="partition suffix, e.g. _k4")
    ap.add_argument("--s10", default="TableS10_cluster_composition.xlsx")
    ap.add_argument("--s9", default="TableS9_clustering_sweep.xlsx")
    args = ap.parse_args()
    tag = args.tag
    fixed, hdb = table_s9(tag)
    s10 = table_s10(tag)

    with pd.ExcelWriter(TABLES / args.s9) as xl:
        fixed.to_excel(xl, sheet_name="fixed_k_algorithms", index=False)
        hdb.to_excel(xl, sheet_name="HDBSCAN", index=False)
    s10.to_excel(TABLES / args.s10, index=False)

    pd.set_option("display.width", 320)
    pd.set_option("display.max_colwidth", 46)
    print("Table S9 - fixed-k algorithms (first 12 rows)")
    print(fixed.head(12).to_string(index=False))
    print("\nTable S9 - HDBSCAN")
    print(hdb.to_string(index=False))
    print("\nTable S10")
    print(s10[["Cluster", "n", "Studies", "% from largest study", "Top tissue",
               "% cellular", "Mean Printability",
               "% Printability class 3"]].to_string(index=False))
    print(f"\n-> {TABLES}")


if __name__ == "__main__":
    main()
