from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.cluster import HDBSCAN

from mlate import clustering as cl
from mlate import config as cfg
from mlate import resources

TABLES = cfg.step_dir("03_clustering", "tables")
MODELS = cfg.step_dir("03_clustering", "models")
MATRIX = cfg.step_dir("02_preprocessing", "tables") / "feature_matrix.parquet"

FINAL_ALGORITHM = "BisectingKMeans"
FINAL_K = 4

MIN_STABILITY = 0.60
K_FLOOR = 4
K_CEILING = 15


def load_matrix(drop_flags: bool) -> pd.DataFrame:
    matrix = pd.read_parquet(MATRIX)
    leaked = [c for c in matrix.columns
              if any(t.lower() in str(c).lower()
                     for t in ("printability", "cell response", "wssq"))]
    if leaked:
        raise SystemExit(f"outcome columns present in the clustering input: "
                         f"{leaked}")
    if drop_flags:
        flags = [c for c in matrix.columns if "[reported]" in str(c)]
        matrix = matrix.drop(columns=flags)
        print(f"dropped {len(flags)} reporting-indicator columns")
    return matrix


def _one(name: str, k: int, X: np.ndarray, space: str) -> dict:
    with resources.single_thread():
        make = cl.algorithms(space=space)[name]
        labels = cl.fit_predict(make(k), X)
        return {"algorithm": name, "k": k,
                "n_clusters": int(len(np.unique(labels))), "n_noise": 0,
                **cl.indices(X, labels),
                "stability": cl.stability(X, make, k)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--space", default="raw", choices=["raw", "pca"])
    ap.add_argument("--drop-reported-flags", action="store_true",
                    help="sensitivity arm: cluster on formulation and process "
                         "columns only, excluding the reporting indicators")
    ap.add_argument("--tag", default="", help="suffix for the output files")
    ap.add_argument("--algorithm", default=FINAL_ALGORITHM,
                    help="algorithm to fit (default: the reported one)")
    ap.add_argument("--k", type=int, default=FINAL_K,
                    help="number of clusters to fit (default: the reported "
                         "k=3, selected by prediction strength and PAC in "
                         "select_k.py; see this module's docstring)")
    args = ap.parse_args()

    resources.claim()
    matrix = load_matrix(args.drop_reported_flags)

    if args.space == "pca":
        X, pca = cl.reduce(matrix.to_numpy(dtype=float))
        print(f"\nmatrix {matrix.shape[0]} x {matrix.shape[1]} -> PCA "
              f"{X.shape[1]} components "
              f"({100 * pca.explained_variance_ratio_.sum():.1f}% variance)")
    else:
        X, pca = matrix.to_numpy(dtype=float), None
        print(f"\nmatrix {matrix.shape[0]} x {matrix.shape[1]} (raw space)")

    jobs = [(name, k) for name in cl.algorithms(space=args.space)
            for k in cl.K_RANGE]
    print(f"{len(jobs)} configurations x {cl.N_BOOTSTRAP + 1} fits "
          f"= {len(jobs) * (cl.N_BOOTSTRAP + 1):,} on {cfg.N_JOBS} workers")
    rows = Parallel(n_jobs=cfg.N_JOBS, backend="loky", batch_size=1,
                    verbose=1)(
        delayed(_one)(name, k, X, args.space) for name, k in jobs)

    for size in (15, 25, 40, 60, 100):
        labels = HDBSCAN(min_cluster_size=size).fit_predict(X)
        n = int(len(np.unique(labels[labels >= 0])))
        if n < 2:
            continue
        rows.append({"algorithm": "HDBSCAN", "k": size, "n_clusters": n,
                     "n_noise": int((labels < 0).sum()),
                     **cl.indices(X, labels), "stability": np.nan})
    print("  HDBSCAN              done")

    sweep = pd.DataFrame(rows)
    fixed_k = sweep[sweep["algorithm"] != "HDBSCAN"]
    eligible = fixed_k[(fixed_k["stability"] >= MIN_STABILITY)
                       & fixed_k["k"].between(K_FLOOR, K_CEILING)]
    if eligible.empty:
        print(f"\nno configuration met stability {MIN_STABILITY} within "
              f"k in [{K_FLOOR}, {K_CEILING}]; falling back to the most stable")
        eligible = fixed_k[fixed_k["k"].between(K_FLOOR, K_CEILING)].nlargest(
            1, "stability")
    eligible = eligible.sort_values(["silhouette", "k"],
                                    ascending=[False, True])
    best = eligible.iloc[0]

    chosen_alg = args.algorithm or best["algorithm"]
    chosen_k = int(args.k or best["k"])
    if args.algorithm or args.k:
        row = fixed_k[(fixed_k.algorithm == chosen_alg)
                      & (fixed_k.k == chosen_k)]
        print(f"\nOVERRIDE: rule selected {best['algorithm']} "
              f"k={int(best['k'])}; using {chosen_alg} k={chosen_k}")
        if not row.empty:
            r = row.iloc[0]
            print(f"  chosen: silhouette {r.silhouette:.3f}  "
                  f"DB {r.davies_bouldin:.3f}  "
                  f"CH {r.calinski_harabasz:.0f}  "
                  f"stability {r.stability:.3f}")

    make = cl.algorithms(space=args.space)[chosen_alg]
    model = make(chosen_k)
    labels = cl.fit_predict(model, X)

    tag = args.tag or ("" if not args.drop_reported_flags else "_noflags")
    tag += "" if args.space == "raw" else "_pca"

    pd.DataFrame({"cluster": labels}, index=matrix.index).to_parquet(
        TABLES / f"cluster_assignments{tag}.parquet")
    joblib.dump({"pca": pca, "model": model, "space": args.space,
                 "algorithm": chosen_alg, "k": chosen_k,
                 "rule_selected": f"{best['algorithm']} k={int(best['k'])}",
                 "feature_names": list(matrix.columns)},
                MODELS / f"clustering{tag}.pkl", compress=3)

    with pd.ExcelWriter(TABLES / f"clustering_sweep{tag}.xlsx") as xl:
        sweep.to_excel(xl, sheet_name="sweep", index=False)
        sweep.sort_values("silhouette", ascending=False).to_excel(
            xl, sheet_name="by_silhouette", index=False)
        eligible.to_excel(xl, sheet_name="eligible", index=False)

    print(f"\nselection rule: stability >= {MIN_STABILITY}, "
          f"k in [{K_FLOOR}, {K_CEILING}], highest silhouette")
    print(f"  {len(eligible)} eligible configurations")
    print(f"rule selected: {best['algorithm']} k={int(best['k'])}  "
          f"silhouette {best['silhouette']:.3f}  "
          f"stability {best['stability']:.3f}")
    print("runner-up: " + "  ".join(
        f"{r.algorithm} k={int(r.k)} sil={r.silhouette:.3f}"
        for r in eligible.iloc[1:4].itertuples()))
    print(f"cluster sizes: {np.bincount(labels[labels >= 0]).tolist()}")
    print(f"\nmodel  -> {MODELS / f'clustering{tag}.pkl'}")
    print(f"tables -> {TABLES}")


if __name__ == "__main__":
    main()
