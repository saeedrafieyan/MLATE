"""
Is the clustering finding chemistry, or finding studies?
========================================================

    python 03_clustering/diagnose.py

Two questions that have to be answered before any cluster is interpreted.

1. Study identity. Study membership already turned out to carry most of the
   apparent signal in the imputation benchmark, and an unsupervised method has
   no protection against it at all: if a publication contributes 30 rows with
   one formulation family and one printing setup, those 30 rows will sit
   together and the algorithm will call it a cluster. Agreement between the
   partition and DOI is measured directly, against agreement with tissue and
   with material class for comparison.

2. Imputation artefacts. Substrate and syringe temperature are unreported in
   58% and 52% of samples and are filled with a single constant, so those two
   columns contain a large artificial spike at 22 C. If clusters are partly
   tracking "did this study report a temperature", dropping the columns will
   move the partition. The clustering is rerun without them and the two
   labellings compared.

Neither test can be passed or failed on its own - they produce numbers that
have to be reported alongside the clusters.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

from mlate import clustering as cl
from mlate import config as cfg
from mlate.dataset import load_dataset, load_taxonomy, modeling_tissue

TABLES = cfg.step_dir("03_clustering", "tables")
MODELS = cfg.step_dir("03_clustering", "models")
MATRIX = cfg.step_dir("02_preprocessing", "tables") / "feature_matrix.parquet"


def dominant_material_class(df, columns) -> pd.Series:
    """Label each sample by the class of the biomaterial it contains most of."""
    tax = load_taxonomy().set_index("column")["material_class"]
    bio = df[columns.biomaterials]
    top = bio.idxmax(axis=1)
    return top.map(tax).fillna("none").astype(str)


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
    matrix = pd.read_parquet(MATRIX)
    assignments = pd.read_parquet(TABLES / f"cluster_assignments{tag}.parquet")
    bundle = joblib.load(MODELS / f"clustering{tag}.pkl")
    labels = assignments["cluster"].to_numpy()
    df, columns = load_dataset()

    print(f"partition: {bundle['algorithm']} k={bundle['k']}")

    # ── 1. what external label does the partition agree with? ───────────────
    external = {
        "DOI (study)": df["DOI"].astype(str),
        "tissue": modeling_tissue(df).astype(str),
        "dominant material class": dominant_material_class(df, columns),
        "cellular vs acellular": (df[columns.cell_line]
                                  != cfg.ACELLULAR_TOKEN).map(
                                      {True: "cellular", False: "acellular"}),
    }
    rows = [{"external_label": name, "n_categories": ext.nunique(),
             **cl.agreement(labels, ext)} for name, ext in external.items()]
    agree = pd.DataFrame(rows)

    # How concentrated is each cluster in a single study?
    per_cluster = (pd.DataFrame({"cluster": labels,
                                 "DOI": df["DOI"].astype(str).to_numpy()})
                   .groupby("cluster")["DOI"]
                   .agg(n_rows="size", n_studies="nunique",
                        largest_study=lambda s: s.value_counts().iloc[0])
                   .reset_index())
    per_cluster["pct_from_largest_study"] = (
        100 * per_cluster["largest_study"] / per_cluster["n_rows"])

    # ── 2. do the constant-filled temperature columns drive the partition? ──
    # The test means nothing without a baseline. k-means at this k does not
    # return the same partition twice, so "removing the columns changed the
    # answer" has to be read against how much a reseed changes it, and against
    # removing the same number of arbitrary columns.
    make = cl.algorithms()[bundle["algorithm"]]
    k = bundle["k"]
    temp = [c for c in matrix.columns if "Temperature" in c]
    rng = np.random.default_rng(cfg.RANDOM_STATE)
    X_full = cl.reduce(matrix.to_numpy(dtype=float))[0]

    def against(cols_removed) -> tuple[float, float]:
        Xr, _ = cl.reduce(matrix.drop(columns=list(cols_removed)).to_numpy(float))
        other = cl.fit_predict(make(k), Xr)
        return (float(adjusted_rand_score(labels, other)),
                float(adjusted_mutual_info_score(labels, other)))

    reseeds = [float(adjusted_rand_score(
                   labels,
                   cl.fit_predict(cl.algorithms(seed)[bundle["algorithm"]](k),
                                  X_full)))
               for seed in (1, 7, 13, 99, 2024)]
    randoms = [against(rng.choice(matrix.columns, size=len(temp),
                                  replace=False))[0] for _ in range(5)]
    temp_ari, temp_ami = against(temp)

    sensitivity = pd.DataFrame([
        {"comparison": "reseed on the full matrix",
         "n_columns_removed": 0, "detail": "seeds 1, 7, 13, 99, 2024",
         "ARI": float(np.mean(reseeds)), "ARI_min": float(np.min(reseeds)),
         "ARI_max": float(np.max(reseeds)), "AMI": np.nan},
        {"comparison": f"remove {len(temp)} arbitrary columns",
         "n_columns_removed": len(temp), "detail": "5 random draws",
         "ARI": float(np.mean(randoms)), "ARI_min": float(np.min(randoms)),
         "ARI_max": float(np.max(randoms)), "AMI": np.nan},
        {"comparison": "remove the temperature columns",
         "n_columns_removed": len(temp), "detail": ", ".join(temp),
         "ARI": temp_ari, "ARI_min": np.nan, "ARI_max": np.nan,
         "AMI": temp_ami},
    ])

    with pd.ExcelWriter(TABLES / f"clustering_diagnostics{tag}.xlsx") as xl:
        agree.to_excel(xl, sheet_name="external_agreement", index=False)
        per_cluster.to_excel(xl, sheet_name="study_concentration", index=False)
        sensitivity.to_excel(xl, sheet_name="temperature_sensitivity",
                             index=False)

    pd.set_option("display.width", 200)
    print("\nagreement with external labels")
    print(agree.to_string(index=False, float_format=lambda v: f"{v:7.3f}"))
    print("\nstudy concentration per cluster")
    print(per_cluster.to_string(index=False,
                                float_format=lambda v: f"{v:7.1f}"))
    print(chr(10) + "sensitivity, against its own baselines")
    print(sensitivity[["comparison", "ARI", "ARI_min", "ARI_max"]]
          .to_string(index=False, float_format=lambda v: f"{v:6.3f}"))
    print(f"\n-> {TABLES / 'clustering_diagnostics.xlsx'}")


if __name__ == "__main__":
    main()
