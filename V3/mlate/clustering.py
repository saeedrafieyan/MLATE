"""
Clustering helpers shared by step 03
====================================

The feature matrix is 153 columns, but 130 of them are biomaterial
concentrations that are zero in roughly 95% of rows, so Euclidean distance
risks being dominated by the sparse block: two formulations judged similar
mostly because they both omit the same 120 materials.

An earlier version of this module applied PCA first and justified it on exactly
that ground. The justification did not survive being tested. Measuring the
Spearman correlation between pairwise distance and the number of jointly-zero
columns gives -0.775 in the raw space and -0.770 after PCA - the projection
does not decouple distance from sparsity, because the leading components are
themselves built out of the sparsity pattern. `reduce()` is therefore offered
but is no longer the default, and the raw space, which is also what the
submitted manuscript used, is primary. 03_clustering/optimal_k.py recomputes
that diagnostic on every run.

Nothing here selects a clustering. It provides the algorithms, the internal
indices and a resampling-based stability score; the choice is made in
03_clustering/run.py from the table these produce.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import (AgglomerativeClustering, BisectingKMeans,
                             KMeans, MiniBatchKMeans)
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score,
                             calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)
from sklearn.mixture import GaussianMixture

from mlate import config as cfg

PCA_VARIANCE = 0.95
K_RANGE = range(2, 31)
N_BOOTSTRAP = 20
BOOTSTRAP_FRACTION = 0.80


def reduce(matrix: np.ndarray, variance: float = PCA_VARIANCE,
           seed: int = cfg.RANDOM_STATE) -> tuple[np.ndarray, PCA]:
    """PCA to a fixed explained-variance target."""
    pca = PCA(n_components=variance, svd_solver="full", random_state=seed)
    return pca.fit_transform(matrix), pca


def algorithms(seed: int = cfg.RANDOM_STATE, space: str = "raw") -> dict:
    """
    k-parameterised algorithms. HDBSCAN chooses its own k and is separate.

    The four of the submitted manuscript - KMeans, MiniBatchKMeans,
    BisectingKMeans and Agglomerative/Ward - plus GaussianMixture. MiniBatch
    appears in the published Table S3, so omitting it would leave a published
    row with no counterpart in the revision.

    GaussianMixture takes full covariance only in a reduced space. In the raw
    153-column matrix, where 130 columns are zero in ~95% of rows, a full
    per-component covariance is rank-deficient: it either fails or is rescued
    by regularisation into a number that means nothing. Diagonal covariance is
    the honest choice there, and the restriction is a property of the space
    rather than a tuning preference.
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


def fit_predict(model, X: np.ndarray) -> np.ndarray:
    """GaussianMixture splits fit and predict; the cluster estimators do not."""
    if hasattr(model, "fit_predict"):
        return np.asarray(model.fit_predict(X))
    return np.asarray(model.fit(X).predict(X))


def indices(X: np.ndarray, labels: np.ndarray) -> dict:
    """Internal validity indices, or NaN where a single cluster makes them undefined."""
    uniq = np.unique(labels[labels >= 0])
    if len(uniq) < 2:
        return {"silhouette": np.nan, "davies_bouldin": np.nan,
                "calinski_harabasz": np.nan}
    keep = labels >= 0            # HDBSCAN marks noise as -1
    return {
        "silhouette": float(silhouette_score(X[keep], labels[keep])),
        "davies_bouldin": float(davies_bouldin_score(X[keep], labels[keep])),
        "calinski_harabasz": float(calinski_harabasz_score(X[keep], labels[keep])),
    }


def _assign_by_centroid(X: np.ndarray, centres: np.ndarray) -> np.ndarray:
    d = ((X[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
    return d.argmin(axis=1)


def stability(X: np.ndarray, make, k: int, n_boot: int = N_BOOTSTRAP,
              fraction: float = BOOTSTRAP_FRACTION,
              seed: int = cfg.RANDOM_STATE) -> float:
    """
    Mean adjusted Rand index between the full-data labelling and labellings
    refitted on resampled subsets.

    A high silhouette on one fit says the partition is compact; it does not say
    the partition would reappear if the corpus had been assembled slightly
    differently. For a literature-mined dataset that second question is the
    important one, so it gets its own number. Subset labels are carried back to
    all rows by nearest centroid, which lets the linkage methods - which have no
    predict() - be scored the same way as the rest.
    """
    rng = np.random.default_rng(seed)
    reference = fit_predict(make(k), X)
    scores = []
    for _ in range(n_boot):
        idx = rng.choice(len(X), size=int(fraction * len(X)), replace=False)
        sub = fit_predict(make(k), X[idx])
        centres = np.vstack([X[idx][sub == c].mean(axis=0)
                             for c in np.unique(sub)])
        scores.append(adjusted_rand_score(reference,
                                          _assign_by_centroid(X, centres)))
    return float(np.mean(scores))



# The plane every clustering figure is drawn in
# --------------------------------------------
REFERENCE_ALGORITHM = "BisectingKMeans"
REFERENCE_K = 4


def reference_plane(X: np.ndarray, seed: int = cfg.RANDOM_STATE):
    """
    One two-dimensional view, shared by Figures 4, 5, S15 and S17.

    Every clustering figure must place a given formulation at the same point,
    or a reader comparing two figures is comparing coordinate systems rather
    than partitions. Fitting each figure's own discriminants - as an earlier
    version did - satisfied nobody: it showed each partition at its best but
    made the figures mutually incomparable.

    The plane chosen is the discriminants of the secondary partition, bisecting
    k-means at k = 4, and the choice is empirical rather than aesthetic. It is
    the only candidate that keeps BOTH reported partitions clearly separated:

                          in the k=3 plane   in the k=4 plane
        k-means k = 3           0.703              0.589
        bisecting k = 4         0.529              0.745

    The k = 3 plane costs the four-cluster partition 0.216 of silhouette; the
    k = 4 plane costs the three-cluster partition 0.114 and still separates it
    plainly. Neither UMAP nor the leading principal components is usable here -
    UMAP scores -0.063 on the reported partition, and two principal components
    of a 153-column matrix collapse the corpus into an overplotted band.

    Because the plane is fitted to labels it flatters the partition it came
    from, so no partition is judged by its appearance in it. Separability is
    always reported from each partition's own discriminants instead.
    """
    from sklearn.cluster import BisectingKMeans
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    labels = BisectingKMeans(n_clusters=REFERENCE_K, n_init=10,
                             random_state=seed).fit_predict(X)
    return (LinearDiscriminantAnalysis(n_components=2)
            .fit(X, labels).transform(X))


def agreement(labels: np.ndarray, other: pd.Series) -> dict:
    """
    How much of a partition is explained by an external label.

    ARI is reported alongside AMI because DOI has 222 categories against a
    handful of clusters, a regime where ARI is pulled toward zero regardless of
    the real association. Purity is the plainest of the three: the share of rows
    sitting in the most common external category of their own cluster.
    """
    other = pd.Series(other).astype(str).to_numpy()
    keep = labels >= 0
    lab, oth = labels[keep], other[keep]
    purity = (pd.DataFrame({"c": lab, "o": oth})
              .groupby("c")["o"].agg(lambda s: s.value_counts().iloc[0]).sum())
    return {
        "ARI": float(adjusted_rand_score(oth, lab)),
        "AMI": float(adjusted_mutual_info_score(oth, lab)),
        "purity": float(purity / len(lab)),
    }
