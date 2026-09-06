"""
Cluster the formulation space
=============================

    python 03_clustering/run.py                    # the partition in the paper
    python 03_clustering/run.py --k 6              # explore another k
    python 03_clustering/run.py --space pca        # sensitivity
    python 03_clustering/run.py --drop-reported-flags   # sensitivity

Fits the reported partition - KMeans, k = 3, in the raw feature space - and
writes it, its model and the full algorithm x k sweep that justifies it.

What goes in
------------
The 153 columns the supervised models receive. The targets are already absent
from the feature matrix and the script asserts it rather than trusting the
pipeline: 130 biomaterial concentrations, the binary-encoded cell line, cell
density, seven printing parameters and seven indicators recording whether each
printing parameter was reported. Bibliographic fields, tissue annotation,
Printability and Cell Response never enter. They are used only afterwards, to
characterise and validate a partition they had no part in forming, which is
what makes the external-agreement results in diagnose.py meaningful rather than
circular.

The space is raw, not PCA. An earlier version reduced to 47 components first
and justified it by the sparse biomaterial block dominating Euclidean distance.
Measured rather than assumed, that claim is false: distance correlates with the
count of jointly-zero columns at -0.775 raw and -0.770 after PCA, so the
projection buys nothing it was meant to buy. `--space pca` reproduces the
reduced variant as a sensitivity check.

Where k = 3 comes from
----------------------
Not from silhouette. Silhouette rises monotonically from k = 2 to k = 60 on
this corpus, as does the gap statistic, so any rule of the form "highest
silhouette within k in [a, b]" returns the edge of its own window - which is how
earlier drafts of this analysis produced 12, then 15, then 2 from the same data.
Those were facts about the window, not about the corpus.

k = 3 comes from two criteria constructed to have an interior optimum, both
computed in select_k.py over five algorithms and k = 2..20:

  prediction strength   0.936   the highest of all 95 configurations tested,
                                and the only k > 2 clearing the conventional
                                0.80 threshold. It collapses to 0.576 at k = 4.
  PAC                   0.026   the lowest of all 95, rising to 0.178 at k = 4.

Bootstrap stability (0.961) agrees. Silhouette does not - it prefers k = 2 and
then every larger k - and that disagreement is reported rather than hidden,
because it is the reason the earlier drafts went wrong.

The honest caveat travels with the number: three clusters separate cellularity
and process regime but resolve little biomaterial structure. Chemically
distinct groups only appear at k >= 5, and those partitions do not survive on
held-out data (prediction strength 0.52 and below). figure_k_choice.py shows
that trade-off, and the manuscript reports it as a finding: this formulation
space is a continuum, not a set of modes.
"""

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

# The reported partition. Chosen by prediction strength and PAC in
# select_k.py, not by the silhouette-based sweep below, which is retained only
# because the submitted manuscript reported those indices and a reader needs
# them to compare.
#
# Note this is NOT the single highest prediction strength in the grid: k-means
# at k = 3 scores 0.936 against 0.794 here. It is not adopted because it splits
# a coherent group - 477 entirely cellular records defined by ionic
# alginate-CaCl2 crosslinking are divided 263/214 across two of its three
# clusters - while bisecting k-means at k = 4 keeps that group intact, nests
# exactly inside the robust cellular/acellular division, and separates
# cellularity better (purity 0.936 against 0.830). See the module docstring.
FINAL_ALGORITHM = "BisectingKMeans"
FINAL_K = 4

MIN_STABILITY = 0.60
K_FLOOR = 4
K_CEILING = 15


def load_matrix(drop_flags: bool) -> pd.DataFrame:
    """The clustering input, with the target-exclusion asserted, not assumed."""
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
    """
    One (algorithm, k) cell: the four indices plus resampling stability.

    Runs in a worker process, so every native thread pool is pinned to one
    thread. Setting the environment variables here would be too late - OpenMP
    reads them when its runtime initialises, long before this function is
    called - and without the pin 51 workers each spawning 51 BLAS threads
    oversubscribe the machine badly enough to make a parallel sweep slower than
    a serial one.
    """
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

    # HDBSCAN picks its own number of clusters, so it is swept over the
    # minimum cluster size and reported alongside rather than ranked against
    # the fixed-k methods.
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
