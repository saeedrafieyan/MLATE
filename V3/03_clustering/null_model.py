from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score, silhouette_score

from mlate import clustering as cl
from mlate import config as cfg
from mlate.dataset import load_dataset, modeling_tissue

TABLES = cfg.step_dir("03_clustering", "tables")
MATRIX = cfg.step_dir("02_preprocessing", "tables") / "feature_matrix.parquet"

K_CURVE = (2, 3, 4, 6, 8, 10, 12, 15, 20, 25, 30, 40, 60)
N_NULL = 3


def main() -> None:
    matrix = pd.read_parquet(MATRIX)
    raw = matrix.to_numpy(dtype=float)
    X, _ = cl.reduce(raw)

    df, columns = load_dataset()
    doi = df["DOI"].astype(str).to_numpy()
    tissue = modeling_tissue(df).astype(str).to_numpy()

    rng = np.random.default_rng(cfg.RANDOM_STATE)
    nulls = []
    for _ in range(N_NULL):
        permuted = raw.copy()
        for j in range(permuted.shape[1]):
            rng.shuffle(permuted[:, j])
        nulls.append(cl.reduce(permuted)[0])

    rows = []
    for k in K_CURVE:
        labels = KMeans(k, n_init=10, random_state=cfg.RANDOM_STATE).fit_predict(X)
        null_scores = [silhouette_score(
            N, KMeans(k, n_init=10, random_state=cfg.RANDOM_STATE).fit_predict(N))
            for N in nulls]
        purity = (pd.DataFrame({"c": labels, "d": doi})
                  .groupby("c")["d"].agg(lambda s: s.value_counts().iloc[0])
                  .sum()) / len(labels)
        rows.append({
            "k": k,
            "silhouette": float(silhouette_score(X, labels)),
            "silhouette_null": float(np.mean(null_scores)),
            "gap_over_null": float(silhouette_score(X, labels) - np.mean(null_scores)),
            "AMI_DOI": float(adjusted_mutual_info_score(doi, labels)),
            "AMI_tissue": float(adjusted_mutual_info_score(tissue, labels)),
            "purity_DOI": float(purity),
        })
        print(f"  k={k:3d} done")

    curve = pd.DataFrame(rows)
    curve.to_excel(TABLES / "null_model_curve.xlsx", index=False)

    pd.set_option("display.width", 200)
    print()
    print(curve.to_string(index=False, float_format=lambda v: f"{v:7.3f}"))
    print(f"\n-> {TABLES / 'null_model_curve.xlsx'}")


if __name__ == "__main__":
    main()
