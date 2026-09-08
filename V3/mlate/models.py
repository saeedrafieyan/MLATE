from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier,
                              ExtraTreesClassifier, GradientBoostingClassifier,
                              HistGradientBoostingClassifier,
                              RandomForestClassifier, StackingClassifier,
                              VotingClassifier)
from sklearn.linear_model import (LogisticRegression, PassiveAggressiveClassifier,
                                  Perceptron, RidgeClassifier, SGDClassifier)
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

from mlate import config as cfg

SEED = cfg.RANDOM_STATE


@dataclass(frozen=True)
class Spec:
    name: str
    family: str
    build: Callable[..., object]
    threaded: bool = False
    gpu: bool = False
    proba: bool = True
    cost: str = "low"
    notes: str = ""


def _zoo() -> list[Spec]:
    S = Spec
    return [
        S("Dummy (majority)", "baseline",
          lambda **k: DummyClassifier(strategy="most_frequent"),
          notes="predicts the training majority class for every sample"),
        S("Dummy (stratified)", "baseline",
          lambda **k: DummyClassifier(strategy="stratified", random_state=SEED),
          notes="samples from the training class distribution"),

        S("Logistic Regression", "linear",
          lambda n_jobs=1, **k: LogisticRegression(
              max_iter=3000, C=1.0, n_jobs=n_jobs, random_state=SEED),
          threaded=True),
        S("Logistic Regression (balanced)", "linear",
          lambda n_jobs=1, **k: LogisticRegression(
              max_iter=3000, C=1.0, class_weight="balanced", n_jobs=n_jobs,
              random_state=SEED),
          threaded=True,
          notes="class_weight balanced; the minority classes are small"),
        S("Ridge Classifier", "linear",
          lambda **k: RidgeClassifier(alpha=1.0, random_state=SEED),
          proba=False),
        S("SGD (hinge)", "linear",
          lambda **k: SGDClassifier(loss="hinge", max_iter=3000, tol=1e-4,
                                    random_state=SEED),
          proba=False),
        S("Passive Aggressive", "linear",
          lambda **k: PassiveAggressiveClassifier(max_iter=3000, tol=1e-4,
                                                  random_state=SEED),
          proba=False),
        S("Perceptron", "linear",
          lambda **k: Perceptron(max_iter=3000, tol=1e-4, random_state=SEED),
          proba=False),

        S("Linear Discriminant", "discriminant",
          lambda **k: LinearDiscriminantAnalysis(solver="lsqr",
                                                 shrinkage="auto")),
        S("Quadratic Discriminant", "discriminant", _qda,
          notes="PCA to 40 components first; QDA estimates a covariance per "
                "class and the smallest class has fewer samples than the 153 "
                "features, which makes those covariances singular"),
        S("Gaussian Naive Bayes", "naive_bayes", lambda **k: GaussianNB()),
        S("Bernoulli Naive Bayes", "naive_bayes",
          lambda **k: BernoulliNB(),
          notes="binarises at 0.5; suits the sparse biomaterial block"),

        S("k-Nearest Neighbours", "neighbours",
          lambda n_jobs=1, **k: KNeighborsClassifier(n_neighbors=5,
                                                     n_jobs=n_jobs),
          threaded=True),
        S("k-NN (distance weighted)", "neighbours",
          lambda n_jobs=1, **k: KNeighborsClassifier(
              n_neighbors=15, weights="distance", n_jobs=n_jobs),
          threaded=True),
        S("Nearest Centroid", "neighbours",
          lambda **k: NearestCentroid(), proba=False),

        S("SVM (RBF)", "svm",
          lambda **k: SVC(kernel="rbf", C=10.0, gamma="scale",
                          probability=True, random_state=SEED),
          cost="high"),
        S("SVM (polynomial)", "svm",
          lambda **k: SVC(kernel="poly", degree=3, C=10.0, gamma="scale",
                          probability=True, random_state=SEED),
          cost="high"),
        S("Linear SVM", "svm",
          lambda **k: LinearSVC(C=1.0, max_iter=5000, dual="auto",
                                random_state=SEED),
          proba=False),

        S("Decision Tree", "tree",
          lambda **k: DecisionTreeClassifier(max_depth=12, min_samples_leaf=3,
                                             random_state=SEED)),
        S("Extra Tree", "tree",
          lambda **k: ExtraTreeClassifier(max_depth=12, min_samples_leaf=3,
                                          random_state=SEED)),

        S("Random Forest", "bagging",
          lambda n_jobs=1, **k: RandomForestClassifier(
              n_estimators=500, min_samples_leaf=2, n_jobs=n_jobs,
              random_state=SEED),
          threaded=True, cost="medium"),
        S("Random Forest (balanced)", "bagging",
          lambda n_jobs=1, **k: RandomForestClassifier(
              n_estimators=500, min_samples_leaf=2,
              class_weight="balanced_subsample", n_jobs=n_jobs,
              random_state=SEED),
          threaded=True, cost="medium"),
        S("Extra Trees", "bagging",
          lambda n_jobs=1, **k: ExtraTreesClassifier(
              n_estimators=500, min_samples_leaf=2, n_jobs=n_jobs,
              random_state=SEED),
          threaded=True, cost="medium"),
        S("Bagged Trees", "bagging",
          lambda n_jobs=1, **k: BaggingClassifier(
              estimator=DecisionTreeClassifier(max_depth=12,
                                               random_state=SEED),
              n_estimators=200, n_jobs=n_jobs, random_state=SEED),
          threaded=True, cost="medium"),

        S("AdaBoost", "boosting",
          lambda **k: AdaBoostClassifier(n_estimators=300, learning_rate=0.5,
                                         random_state=SEED),
          cost="medium"),
        S("Gradient Boosting", "boosting",
          lambda **k: GradientBoostingClassifier(
              n_estimators=300, learning_rate=0.1, max_depth=3,
              random_state=SEED),
          cost="high"),
        S("Hist Gradient Boosting", "boosting",
          lambda **k: HistGradientBoostingClassifier(
              max_iter=400, learning_rate=0.08, early_stopping=True,
              validation_fraction=0.15, random_state=SEED),
          threaded=True, cost="medium"),
        S("XGBoost", "boosting", _xgboost, threaded=True, gpu=True,
          cost="medium"),
        S("LightGBM", "boosting", _lightgbm, threaded=True, cost="medium"),
        S("CatBoost", "boosting", _catboost, threaded=True, cost="medium",
          notes="ordered boosting; symmetric trees"),

        S("MLP (256-128)", "neural",
          lambda **k: MLPClassifier(
              hidden_layer_sizes=(256, 128), alpha=1e-3, max_iter=600,
              early_stopping=True, n_iter_no_change=25, random_state=SEED),
          threaded=True, cost="medium"),

        S("Soft Voting (RF+XGB+LR)", "meta", _voting, threaded=True,
          cost="high",
          notes="soft vote over a bagging, a boosting and a linear learner"),
        S("Stacking (RF+XGB+LR -> LR)", "meta", _stacking, threaded=True,
          cost="high",
          notes="5-fold internal stacking, logistic meta-learner"),
    ]


def _xgboost(n_jobs: int = 1, device: str | None = None, **kwargs):
    from xgboost import XGBClassifier
    params = dict(n_estimators=600, learning_rate=0.06, max_depth=6,
                  subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
                  tree_method="hist", n_jobs=n_jobs, random_state=SEED,
                  eval_metric="mlogloss")
    if device:
        params["device"] = device
    return XGBClassifier(**params)


def _lightgbm(n_jobs: int = 1, **kwargs):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(n_estimators=600, learning_rate=0.06, max_depth=-1,
                          num_leaves=63, subsample=0.85, subsample_freq=1,
                          colsample_bytree=0.85, n_jobs=n_jobs,
                          random_state=SEED, verbose=-1)


def _qda(**kwargs):
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    return Pipeline([
        ("pca", PCA(n_components=40, random_state=SEED)),
        ("qda", QuadraticDiscriminantAnalysis(reg_param=0.3)),
    ])


def _catboost(n_jobs: int = 1, **kwargs):
    from catboost import CatBoostClassifier
    return CatBoostClassifier(iterations=600, learning_rate=0.06, depth=6,
                              l2_leaf_reg=3.0, thread_count=n_jobs,
                              random_seed=SEED, verbose=0,
                              allow_writing_files=False)


def _bases(n_jobs: int = 1):
    return [
        ("rf", RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                      n_jobs=n_jobs, random_state=SEED)),
        ("xgb", _xgboost(n_jobs=n_jobs)),
        ("lr", LogisticRegression(max_iter=3000, random_state=SEED)),
    ]


def _voting(n_jobs: int = 1, **kwargs):
    return VotingClassifier(estimators=_bases(n_jobs), voting="soft",
                            n_jobs=1)


def _stacking(n_jobs: int = 1, **kwargs):
    return StackingClassifier(
        estimators=_bases(n_jobs),
        final_estimator=LogisticRegression(max_iter=3000, random_state=SEED),
        cv=5, n_jobs=1, passthrough=False)


REGISTRY: dict[str, Spec] = {s.name: s for s in _zoo()}


def names(exclude_baselines: bool = False) -> list[str]:
    return [n for n, s in REGISTRY.items()
            if not (exclude_baselines and s.family == "baseline")]


def build(name: str, n_jobs: int = 1, device: str | None = None):
    spec = REGISTRY[name]
    kwargs = {}
    if spec.threaded:
        kwargs["n_jobs"] = n_jobs
    if spec.gpu and device:
        kwargs["device"] = device
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return spec.build(**kwargs)


def summary():
    import pandas as pd
    return pd.DataFrame([{
        "model": s.name, "family": s.family, "threaded": s.threaded,
        "gpu": s.gpu, "predict_proba": s.proba, "cost": s.cost,
        "notes": s.notes,
    } for s in REGISTRY.values()])
