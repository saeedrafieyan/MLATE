"""
The optimal number of clusters, from criteria that can actually have an optimum
==============================================================================

    python 03_clustering/select_k.py --dry-run
    python 03_clustering/select_k.py

Why this file exists
--------------------
Earlier passes selected k with silhouette, or with rules built on top of it -
"highest silhouette above a stability floor, within k in [a, b]". On this corpus
silhouette rises monotonically from k = 2 to k = 60, and so does the gap
statistic. Any rule of that shape therefore returns the edge of whatever window
it was given: window [2,12] returns 12, window [4,15] returns 15, no window
returns 2. Those are three different answers to the question "what does the
window say", not three answers to "how many clusters are there". The fault is
the criterion, not the data.

Two criteria are constructed to have an interior optimum, and both are used here.

PREDICTION STRENGTH  (Tibshirani & Walther 2005)
    Split the rows in half. Cluster each half independently into k groups.
    Assign the test half to the training half's centroids, then ask, for each
    test cluster, what proportion of its within-cluster pairs are still
    co-assigned under the training centroids. Prediction strength is the
    WORST cluster's proportion - a partition is only as trustworthy as its
    least reproducible group.

    This cannot run away to large k. Splitting genuine structure into more
    pieces than it contains produces at least one group that the other half
    cannot reproduce, and the minimum collapses. The conventional decision rule
    is the largest k whose mean prediction strength reaches 0.80.

    The reported partition does not quite meet that bar and the departure is
    deliberate: bisecting k-means at k = 4 scores 0.794. It is adopted over the
    highest-scoring cell (k-means at k = 3, 0.936) because prediction strength
    measures REPRODUCIBILITY, not interpretability, and the k = 3 solution
    splits a coherent 477-record group - entirely cellular, defined by ionic
    alginate-CaCl2 crosslinking - 263/214 across two of its clusters. Within its
    own algorithm the reported configuration is a local maximum rather than a
    near miss: 0.860, 0.680, 0.794, 0.580 for k = 2..5. The full grid is written
    out so a reader can apply a different rule and see what it returns.

PROPORTION OF AMBIGUOUS CLUSTERING  (Monti et al. 2003; Senbabaoglu 2014)
    Resample the rows many times, cluster each resample, and record how often
    each pair of points lands in the same cluster given both were drawn. A
    consensus value near 0 or near 1 is a decided pair; anything between is
    ambiguous. PAC is the fraction of pairs falling in (0.1, 0.9), so LOWER is
    better, and it too degrades once k exceeds the structure present.

Both are computed for every algorithm, so the algorithm and k are chosen
together rather than k being chosen inside an algorithm picked for other
reasons. Silhouette, Davies-Bouldin, Calinski-Harabasz and bootstrap stability
are carried alongside for continuity with the submitted manuscript, but they do
not decide anything here.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import silhouette_score

from mlate import clustering as cl
from mlate import config as cfg
from mlate import resources

warnings.filterwarnings("ignore")

TABLES = cfg.step_dir("03_clustering", "tables")
MATRIX = cfg.step_dir("02_preprocessing", "tables") / "feature_matrix.parquet"

K_RANGE = range(2, 21)
PS_SPLITS = 20          # half-splits per (algorithm, k)
CONSENSUS_RUNS = 30     # resamples per (algorithm, k)
CONSENSUS_FRACTION = 0.80
PAC_LOW, PAC_HIGH = 0.10, 0.90
PS_THRESHOLD = 0.80


def _centroids(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return np.vstack([X[labels == c].mean(axis=0)
                      for c in np.unique(labels)])


def _assign(X: np.ndarray, centres: np.ndarray) -> np.ndarray:
    d = ((X[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
    return d.argmin(axis=1)


def prediction_strength(X: np.ndarray, make, k: int, n_splits: int,
                        seed: int) -> float:
    """
    Mean over splits of the worst cluster's preserved co-membership fraction.

    Comparing co-membership rather than labels is what makes this well defined:
    cluster numbering is arbitrary between two independent fits, so any
    label-matching scheme would need an assignment step and would inherit its
    arbitrariness. Pairs do not have that problem.
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    out = []
    for _ in range(n_splits):
        perm = rng.permutation(n)
        a, b = perm[: n // 2], perm[n // 2:]
        try:
            lab_a = cl.fit_predict(make(k), X[a])
            lab_b = cl.fit_predict(make(k), X[b])
        except Exception:
            continue
        if len(np.unique(lab_a)) < 2 or len(np.unique(lab_b)) < 2:
            continue
        # test points, assigned to the TRAINING half's centroids
        cross = _assign(X[b], _centroids(X[a], lab_a))
        worst = 1.0
        for c in np.unique(lab_b):
            idx = np.where(lab_b == c)[0]
            m = len(idx)
            if m < 2:
                continue
            same = cross[idx][:, None] == cross[idx][None, :]
            # off-diagonal pairs only
            preserved = (same.sum() - m) / (m * (m - 1))
            worst = min(worst, float(preserved))
        out.append(worst)
    return float(np.mean(out)) if out else np.nan


def pac(X: np.ndarray, make, k: int, n_runs: int, seed: int) -> float:
    """
    Proportion of pairs whose co-clustering frequency is ambiguous.

    The consensus matrix is accumulated as counts rather than stored per run,
    and only the scalar comes back, because a 2,646 x 2,646 float matrix per
    (algorithm, k) cell would not survive being held across a parallel sweep.
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    m = int(CONSENSUS_FRACTION * n)
    co = np.zeros((n, n), dtype=np.float32)
    seen = np.zeros((n, n), dtype=np.float32)
    for _ in range(n_runs):
        idx = rng.choice(n, m, replace=False)
        try:
            lab = cl.fit_predict(make(k), X[idx])
        except Exception:
            continue
        seen[np.ix_(idx, idx)] += 1
        same = lab[:, None] == lab[None, :]
        block = co[np.ix_(idx, idx)]
        block += same.astype(np.float32)
        co[np.ix_(idx, idx)] = block
    mask = seen > 0
    iu = np.triu_indices(n, k=1)
    valid = mask[iu]
    if not valid.any():
        return np.nan
    frac = co[iu][valid] / seen[iu][valid]
    return float(np.mean((frac > PAC_LOW) & (frac < PAC_HIGH)))


def one_cell(name: str, k: int, X: np.ndarray, space: str,
             ps_splits: int, runs: int) -> dict:
    with resources.single_thread():
        make = cl.algorithms(space=space)[name]
        labels = cl.fit_predict(make(k), X)
        row = {"algorithm": name, "k": k,
               "n_clusters": int(len(np.unique(labels)))}
        if row["n_clusters"] > 1:
            row["silhouette"] = float(silhouette_score(X, labels))
            row["stability"] = cl.stability(X, make, k)
        row["prediction_strength"] = prediction_strength(
            X, make, k, ps_splits, cfg.RANDOM_STATE)
        row["pac"] = pac(X, make, k, runs, cfg.RANDOM_STATE)
        return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--space", default="raw", choices=["raw", "pca"])
    ap.add_argument("--ps-splits", type=int, default=PS_SPLITS)
    ap.add_argument("--runs", type=int, default=CONSENSUS_RUNS)
    ap.add_argument("--drop-reported-flags", action="store_true",
                    help="exclude the 7 missingness indicators. They encode "
                         "which parameters a journal required rather than any "
                         "property of a scaffold, and they carry a strong "
                         "study signature, so whether the reproducible "
                         "structure survives without them is the question "
                         "this flag answers.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    resources.claim()
    matrix = pd.read_parquet(MATRIX)
    if args.drop_reported_flags:
        drop = [c for c in matrix.columns if "[reported]" in str(c)]
        matrix = matrix.drop(columns=drop)
        print(f"dropped {len(drop)} reporting-indicator columns")
    X = matrix.to_numpy(dtype=float)
    if args.space == "pca":
        X, _ = cl.reduce(X)
    print(f"\nmatrix {X.shape[0]} x {X.shape[1]} ({args.space} space)")

    jobs = [(a, k) for a in cl.algorithms(space=args.space) for k in K_RANGE]
    print(f"{len(jobs)} cells | prediction strength {args.ps_splits} splits, "
          f"consensus {args.runs} resamples")
    if args.dry_run:
        print("dry run")
        return

    rows = Parallel(n_jobs=min(cfg.N_JOBS, 32), backend="loky", batch_size=1,
                    verbose=1)(
        delayed(one_cell)(a, k, X, args.space, args.ps_splits, args.runs)
        for a, k in jobs)
    d = pd.DataFrame(rows)
    suffix = "_noflags" if args.drop_reported_flags else ""
    dest = TABLES / f"select_k_{args.space}{suffix}.xlsx"
    d.round(5).to_excel(dest, index=False)

    pd.set_option("display.width", 240)
    print("\nprediction strength (higher is better; >= 0.80 is the rule)")
    print(d.pivot_table(index="k", columns="algorithm",
                        values="prediction_strength").round(3).to_string())
    print("\nPAC, proportion of ambiguous pairs (LOWER is better)")
    print(d.pivot_table(index="k", columns="algorithm",
                        values="pac").round(3).to_string())

    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    ok = d[d.prediction_strength >= PS_THRESHOLD]
    if ok.empty:
        print(f"no (algorithm, k) reached prediction strength "
              f"{PS_THRESHOLD}; best was "
              f"{d.prediction_strength.max():.3f}")
    else:
        best = ok.sort_values(["k", "prediction_strength"],
                              ascending=[False, False]).iloc[0]
        print(f"largest k with prediction strength >= {PS_THRESHOLD}: "
              f"{best.algorithm} k={int(best.k)} "
              f"(PS {best.prediction_strength:.3f}, PAC {best.pac:.3f})")
    lowest = d.loc[d.pac.idxmin()]
    print(f"lowest PAC overall: {lowest.algorithm} k={int(lowest.k)} "
          f"(PAC {lowest.pac:.3f}, PS {lowest.prediction_strength:.3f})")
    print(f"\n-> {dest}")


if __name__ == "__main__":
    main()
