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
    pca = PCA(n_components=variance, svd_solver="full", random_state=seed)
    return pca.fit_transform(matrix), pca


def algorithms(seed: int = cfg.RANDOM_STATE, space: str = "raw") -> dict:
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
    if hasattr(model, "fit_predict"):
        return np.asarray(model.fit_predict(X))
    return np.asarray(model.fit(X).predict(X))


def indices(X: np.ndarray, labels: np.ndarray) -> dict:
    uniq = np.unique(labels[labels >= 0])
    if len(uniq) < 2:
        return {"silhouette": np.nan, "davies_bouldin": np.nan,
                "calinski_harabasz": np.nan}
    keep = labels >= 0
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


REFERENCE_ALGORITHM = "BisectingKMeans"
REFERENCE_K = 4


def reference_plane(X: np.ndarray, seed: int = cfg.RANDOM_STATE):
    from sklearn.cluster import BisectingKMeans
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    labels = BisectingKMeans(n_clusters=REFERENCE_K, n_init=10,
                             random_state=seed).fit_predict(X)
    return (LinearDiscriminantAnalysis(n_components=2)
            .fit(X, labels).transform(X))


def agreement(labels: np.ndarray, other: pd.Series) -> dict:
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
