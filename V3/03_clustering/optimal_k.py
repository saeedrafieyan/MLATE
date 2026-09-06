"""
Does this corpus have a natural number of clusters?  (It does not.)
==================================================================

    python 03_clustering/optimal_k.py --dry-run
    python 03_clustering/optimal_k.py

This script establishes the negative result the selection rests on: no
compactness criterion turns over on this corpus, so no compactness criterion
can choose k. select_k.py then does the choosing, with criteria that can fail.

What is computed, and what it showed
------------------------------------
1. GAP STATISTIC (Tibshirani, Walther & Hastie 2001) with the one-standard-error
   rule. The standard principled selector, and explicitly constructed to have an
   interior optimum when one exists. It has none here: the gap rises at all 33
   steps from k = 2 to 60, in both the raw and the PCA space. Where the 1-SE rule
   fires on such a curve it is firing on noise - one increment falling below one
   standard error - so `gap_choice` reports the monotonicity check alongside the
   trigger and refuses the number.
2. GAUSSIAN-MIXTURE BIC. Penalises parameters directly and often has a clean
   minimum where compactness indices have none. Here its argmin sits at the edge
   of the swept range in the raw space, and in the PCA space is shallower than
   the curve's own step-to-step variation, so `bic_choice` rejects it too.
3. RAW vs PCA, settled by diagnostic rather than preference. 130 of the 153
   columns are biomaterial concentrations that are zero in ~95% of rows, so raw
   Euclidean distance may be dominated by shared ABSENCES. The Spearman
   correlation between pairwise distance and the number of jointly-zero columns
   measures exactly that - and it is -0.775 raw against -0.770 after PCA, so the
   projection does not decouple distance from sparsity and buys nothing it was
   meant to buy. The raw space, which the submitted manuscript also used, is
   primary.
4. HOPKINS, as a cluster-tendency check: 0.987 raw, 0.979 PCA, far from the 0.5
   expected of structureless data. Structure exists; a preferred NUMBER of
   groups does not.

Why the fallback in this file is not the reported partition
-----------------------------------------------------------
The fallback below - highest silhouette above a stability floor, within an
interpretability ceiling - is retained only to show what such a rule returns,
because rules of that shape are what the literature commonly uses. It is
degenerate on this corpus: silhouette rises monotonically, so the rule returns
the edge of whatever window it is given, and different windows returned 12, 15
and 2 from identical data. The reported partition comes from select_k.py
instead.
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
from sklearn.cluster import (AgglomerativeClustering, BisectingKMeans,
                             HDBSCAN, KMeans, MiniBatchKMeans)
from sklearn.decomposition import PCA
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)
from sklearn.mixture import GaussianMixture

from mlate import clustering as cl
from mlate import config as cfg
from mlate import resources

warnings.filterwarnings("ignore")

TABLES = cfg.step_dir("03_clustering", "tables")
MATRIX = cfg.step_dir("02_preprocessing", "tables") / "feature_matrix.parquet"

K_RANGE = range(2, 31)
TAIL_K = (35, 40, 45, 50, 60)      # KMeans only, to document the tail
GAP_B = 20                          # reference datasets per k
N_BOOT = 20
SILHOUETTE_SAMPLE = 2646            # whole corpus; no subsampling
INTERPRETABLE_CEILING = 12
STABILITY_FLOOR = 0.60


# ── spaces ───────────────────────────────────────────────────────────────────

def build_spaces(matrix: pd.DataFrame) -> dict:
    """The two candidate distance spaces, plus the diagnostics that judge them."""
    raw = matrix.to_numpy(dtype=float)
    pca = PCA(n_components=cl.PCA_VARIANCE, svd_solver="full",
              random_state=cfg.RANDOM_STATE)
    reduced = pca.fit_transform(raw)
    return {
        "raw": {"X": raw, "n_features": raw.shape[1], "pca": None},
        "pca": {"X": reduced, "n_features": reduced.shape[1], "pca": pca},
    }


def sparsity_coupling(raw: np.ndarray, X: np.ndarray, n_pairs: int = 40_000,
                      seed: int = cfg.RANDOM_STATE) -> float:
    """
    Spearman correlation between pairwise distance and shared-zero count.

    The question this answers: when two formulations are far apart in this
    space, is that because their chemistries differ, or merely because they
    omit different subsets of 130 mostly-absent materials? A strong negative
    correlation - more shared zeros, smaller distance - means the space is
    largely measuring co-absence.
    """
    from scipy.stats import spearmanr
    rng = np.random.default_rng(seed)
    n = len(raw)
    i = rng.integers(0, n, n_pairs)
    j = rng.integers(0, n, n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    zero = raw == 0
    shared = (zero[i] & zero[j]).sum(axis=1)
    dist = np.linalg.norm(X[i] - X[j], axis=1)
    return float(spearmanr(shared, dist).statistic)


def hopkins(X: np.ndarray, m_frac: float = 0.05,
            seed: int = cfg.RANDOM_STATE) -> float:
    """
    Hopkins statistic: does this data cluster at all?

    ~0.5 means indistinguishable from uniform (no cluster tendency); values
    approaching 1 indicate structure. Reported because a k-selection exercise
    on structureless data is meaningless regardless of which criterion wins.
    """
    from sklearn.neighbors import NearestNeighbors
    rng = np.random.default_rng(seed)
    n, d = X.shape
    m = max(10, int(m_frac * n))
    lo, hi = X.min(axis=0), X.max(axis=0)

    sample_idx = rng.choice(n, m, replace=False)
    nn = NearestNeighbors(n_neighbors=2).fit(X)
    # distance to the nearest OTHER real point
    w = nn.kneighbors(X[sample_idx], n_neighbors=2)[0][:, 1]
    # distance from a uniform point to the nearest real point
    uniform = rng.uniform(lo, hi, size=(m, d))
    u = nn.kneighbors(uniform, n_neighbors=1)[0][:, 0]
    return float(u.sum() / (u.sum() + w.sum()))


# ── algorithms ───────────────────────────────────────────────────────────────

def algorithms(space: str, seed: int = cfg.RANDOM_STATE) -> dict:
    """
    The four algorithms of the submitted manuscript plus the two added in
    re-analysis. MiniBatchKMeans is restored: it appears in the published
    Table S3, and dropping it would leave a published row with no counterpart.

    GaussianMixture uses full covariance in PCA space and DIAGONAL covariance
    in raw space. In 153 dimensions where 130 columns are zero in ~95% of rows,
    a full per-component covariance matrix is rank-deficient; it would either
    fail outright or be rescued by regularisation into a number that means
    nothing. The restriction is a real limitation of the raw space and is
    recorded as such rather than hidden.
    """
    cov = "full" if space == "pca" else "diag"
    return {
        "KMeans": lambda k: KMeans(n_clusters=k, n_init=20, random_state=seed),
        "MiniBatchKMeans": lambda k: MiniBatchKMeans(
            n_clusters=k, n_init=10, random_state=seed, batch_size=1024),
        "BisectingKMeans": lambda k: BisectingKMeans(
            n_clusters=k, n_init=10, random_state=seed),
        "AgglomerativeWard": lambda k: AgglomerativeClustering(
            n_clusters=k, linkage="ward"),
        "GaussianMixture": lambda k: GaussianMixture(
            n_components=k, covariance_type=cov, n_init=5, random_state=seed,
            reg_covar=1e-5),
    }


def within_dispersion(X: np.ndarray, labels: np.ndarray) -> float:
    """Tibshirani's W_k: pooled within-cluster sum of squares."""
    total = 0.0
    for c in np.unique(labels):
        pts = X[labels == c]
        if len(pts) > 1:
            total += ((pts - pts.mean(axis=0)) ** 2).sum()
    return float(total)


# ── criteria ─────────────────────────────────────────────────────────────────

def score_one(space: str, name: str, k: int, X: np.ndarray) -> dict:
    """Every k-dependent criterion for one (space, algorithm, k)."""
    make = algorithms(space)[name]
    with resources.single_thread():
        model = make(k)
        labels = cl.fit_predict(model, X)
        row = {"space": space, "algorithm": name, "k": k,
               "n_clusters": int(len(np.unique(labels)))}
        if row["n_clusters"] < 2:
            return row
        row.update({
            "silhouette": float(silhouette_score(X, labels)),
            "davies_bouldin": float(davies_bouldin_score(X, labels)),
            "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
            "within_dispersion": within_dispersion(X, labels),
        })
        if name == "GaussianMixture":
            row["bic"] = float(model.bic(X))
            row["aic"] = float(model.aic(X))
        if hasattr(model, "inertia_"):
            row["inertia"] = float(model.inertia_)
        row["stability"] = cl.stability(X, make, k, n_boot=N_BOOT)
    return row


def gap_one(space: str, k: int, X: np.ndarray, seed: int) -> dict:
    """
    One point of the gap curve: log W_k on the data against B uniform
    reference sets drawn over the bounding box of the same space.
    """
    make = algorithms(space)["KMeans"]
    rng = np.random.default_rng(seed + k)
    with resources.single_thread():
        obs = np.log(within_dispersion(X, cl.fit_predict(make(k), X)))
        lo, hi = X.min(axis=0), X.max(axis=0)
        refs = []
        for _ in range(GAP_B):
            ref = rng.uniform(lo, hi, size=X.shape)
            refs.append(np.log(within_dispersion(
                ref, cl.fit_predict(make(k), ref))))
    refs = np.asarray(refs)
    return {"space": space, "k": k, "log_Wk": obs,
            "log_Wk_ref_mean": float(refs.mean()),
            "gap": float(refs.mean() - obs),
            "sd": float(refs.std(ddof=1)),
            "s_k": float(refs.std(ddof=1) * np.sqrt(1 + 1 / GAP_B))}


def gap_choice(gap: pd.DataFrame) -> tuple[int | None, str]:
    """
    Tibshirani's 1-SE rule: smallest k with Gap(k) >= Gap(k+1) - s(k+1).

    Returns the triggering k AND a monotonicity verdict, because the rule alone
    is not safe to read as an optimum. It compares one increment against one
    standard error, so on a curve that never turns over it still fires wherever
    a single increment happens to fall below the noise band - which is a
    fluctuation, not a knee. On this corpus the PCA curve rises at every one of
    33 steps yet the rule fired at k=23 on an increment of 0.0014 against a mean
    of 0.033. Reporting that as "the optimal number of clusters is 23" would
    have been wrong, so the check travels with the number.
    """
    g = gap.sort_values("k").reset_index(drop=True)
    inc = g["gap"].diff().dropna()
    monotone = bool((inc > 0).all())
    hit = None
    for i in range(len(g) - 1):
        if g.loc[i, "gap"] >= g.loc[i + 1, "gap"] - g.loc[i + 1, "s_k"]:
            hit = int(g.loc[i, "k"])
            break
    if hit is not None and monotone:
        return None, (f"rule fired at k={hit} but the gap curve rises at all "
                      f"{len(inc)} steps: a fluctuation, not an optimum")
    if hit is None:
        return None, "no k satisfied the 1-SE rule"
    return hit, "interior optimum"


def bic_choice(b: pd.DataFrame) -> tuple[int | None, str]:
    """
    Interior BIC minimum, rejected when it is indistinguishable from noise.

    A minimum only means something if it is deeper than the curve's own
    step-to-step wobble. Here neighbouring k differ by more than the margin the
    "minimum" holds over them, so the argmin is not a model-selection signal.
    """
    b = b.sort_values("k").reset_index(drop=True)
    i = int(b["bic"].values.argmin())
    if i == 0 or i == len(b) - 1:
        return None, "argmin sits at the edge of the swept range"
    depth = float(b["bic"].values[[i - 1, i + 1]].min() - b["bic"].values[i])
    wobble = float(np.abs(np.diff(b["bic"].values)).mean())
    if depth < wobble:
        return None, (f"argmin k={int(b['k'].values[i])} is shallower "
                      f"({depth:,.0f}) than the mean step-to-step change "
                      f"({wobble:,.0f}): noise, not a minimum")
    return int(b["k"].values[i]), "interior minimum"


def knee(k_values, curve) -> int | None:
    """
    Elbow by maximum distance to the chord joining the curve's endpoints.

    The classical construction, implemented directly rather than pulled from a
    dependency: it always returns a point, so a value here is not evidence that
    an elbow exists - the curvature reported beside it is what says whether the
    bend is real.
    """
    k = np.asarray(k_values, dtype=float)
    y = np.asarray(curve, dtype=float)
    if len(k) < 3:
        return None
    k_n = (k - k.min()) / max(np.ptp(k), 1e-12)
    y_n = (y - y.min()) / max(np.ptp(y), 1e-12)
    p0, p1 = np.array([k_n[0], y_n[0]]), np.array([k_n[-1], y_n[-1]])
    d = p1 - p0
    d = d / np.linalg.norm(d)
    pts = np.column_stack([k_n, y_n]) - p0
    dist = np.abs(pts[:, 0] * d[1] - pts[:, 1] * d[0])
    return int(k[int(dist.argmax())])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    resources.claim()
    matrix = pd.read_parquet(MATRIX)
    spaces = build_spaces(matrix)

    print(f"\nmatrix {matrix.shape[0]} x {matrix.shape[1]}")
    diag = []
    for name, sp in spaces.items():
        h = hopkins(sp["X"])
        s = sparsity_coupling(spaces["raw"]["X"], sp["X"])
        diag.append({"space": name, "n_features": sp["n_features"],
                     "hopkins": h, "sparsity_coupling_spearman": s})
        print(f"  {name:4s}  {sp['n_features']:3d} features  "
              f"hopkins={h:.3f}  distance~shared-zeros rho={s:+.3f}")

    jobs = [(sp, alg, k) for sp in spaces for alg in algorithms(sp)
            for k in K_RANGE]
    jobs += [(sp, "KMeans", k) for sp in spaces for k in TAIL_K]
    gaps = [(sp, k) for sp in spaces for k in list(K_RANGE) + list(TAIL_K)]
    print(f"\n{len(jobs)} criterion fits + {len(gaps)} gap points "
          f"({GAP_B} references each = {len(gaps) * GAP_B:,} reference fits)")
    if args.dry_run:
        print("dry run; nothing fitted")
        return

    workers = args.workers or cfg.N_JOBS
    print(f"{workers} workers\n")

    rows = Parallel(n_jobs=workers, backend="loky", batch_size=1, verbose=1)(
        delayed(score_one)(sp, alg, k, spaces[sp]["X"]) for sp, alg, k in jobs)
    grid = pd.DataFrame([r for r in rows if r])

    gap_rows = Parallel(n_jobs=workers, backend="loky", batch_size=1,
                        verbose=1)(
        delayed(gap_one)(sp, k, spaces[sp]["X"], cfg.RANDOM_STATE)
        for sp, k in gaps)
    gapdf = pd.DataFrame(gap_rows)

    # HDBSCAN sweeps its own parameter and is reported, not ranked.
    hdb = []
    for sp in spaces:
        for size in (15, 25, 40, 60, 100):
            lab = HDBSCAN(min_cluster_size=size).fit_predict(spaces[sp]["X"])
            n = int(len(np.unique(lab[lab >= 0])))
            if n < 2:
                continue
            hdb.append({"space": sp, "min_cluster_size": size, "n_clusters": n,
                        "n_noise": int((lab < 0).sum()),
                        **cl.indices(spaces[sp]["X"], lab)})

    dest = TABLES / "optimal_k.xlsx"
    with pd.ExcelWriter(dest) as xl:
        pd.DataFrame(diag).to_excel(xl, sheet_name="space_diagnostics",
                                    index=False)
        grid.round(5).to_excel(xl, sheet_name="criteria", index=False)
        gapdf.round(5).to_excel(xl, sheet_name="gap_statistic", index=False)
        pd.DataFrame(hdb).round(5).to_excel(xl, sheet_name="hdbscan",
                                            index=False)

    # ── apply the pre-registered rule ────────────────────────────────────
    print("\n" + "=" * 74)
    print("PRE-REGISTERED DECISION RULE")
    print("=" * 74)
    verdict = []
    for sp in spaces:
        g = gapdf[gapdf.space == sp]
        gk, gap_note = gap_choice(g)
        gm = grid[(grid.space == sp) & (grid.algorithm == "GaussianMixture")
                  & grid.bic.notna()]
        bic_k, bic_note = None, "no GMM BIC available"
        if not gm.empty:
            bic_k, bic_note = bic_choice(gm)
        km = grid[(grid.space == sp) & (grid.algorithm == "KMeans")
                  & grid.inertia.notna()].sort_values("k")
        elbow = knee(km.k, km.inertia) if not km.empty else None
        verdict.append({"space": sp, "gap_1se_k": gk,
                        "gap_note": gap_note,
                        "gmm_bic_interior_min_k": bic_k, "bic_note": bic_note,
                        "inertia_elbow_k": elbow})
        print(f"\n[{sp}]")
        print(f"  1. gap statistic, 1-SE rule      -> "
              f"{gk if gk else 'NO optimum'}   [{gap_note}]")
        print(f"  2. GMM BIC interior minimum      -> "
              f"{bic_k if bic_k else 'NO optimum'}   [{bic_note}]")
        print(f"     inertia elbow (max-chord)     -> {elbow}  "
              f"(reported; always returns a value)")
        if gk is None and bic_k is None:
            cand = grid[(grid.space == sp) & (grid.k <= INTERPRETABLE_CEILING)
                        & (grid.stability >= STABILITY_FLOOR)]
            if cand.empty:
                print("  3. FALLBACK: no configuration met the stability floor")
            else:
                best = cand.sort_values(["silhouette", "k"],
                                        ascending=[False, True]).iloc[0]
                print(f"  3. FALLBACK (no natural k): {best.algorithm} "
                      f"k={int(best.k)}  silhouette={best.silhouette:.3f} "
                      f"stability={best.stability:.3f}")
                verdict[-1]["fallback"] = (f"{best.algorithm} k={int(best.k)}")
    pd.DataFrame(verdict).to_excel(TABLES / "optimal_k_verdict.xlsx",
                                   index=False)
    print(f"\n-> {dest}")


if __name__ == "__main__":
    main()
