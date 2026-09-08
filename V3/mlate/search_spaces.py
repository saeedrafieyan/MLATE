from __future__ import annotations

from typing import Callable

from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier,
                              ExtraTreesClassifier, GradientBoostingClassifier,
                              HistGradientBoostingClassifier,
                              RandomForestClassifier, StackingClassifier,
                              VotingClassifier)
from sklearn.linear_model import (LogisticRegression,
                                  PassiveAggressiveClassifier, Perceptron,
                                  RidgeClassifier, SGDClassifier)
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

from mlate import config as cfg

SEED = cfg.RANDOM_STATE

BASELINES = ("Dummy (majority)", "Dummy (stratified)")

META = ("Soft Voting (RF+XGB+LR)", "Stacking (RF+XGB+LR -> LR)")

_MLP_LAYERS = {
    "64": (64,), "128": (128,), "256": (256,),
    "128_64": (128, 64), "256_128": (256, 128), "256_128_64": (256, 128, 64),
}


def _logistic(t) -> dict:
    return {
        "penalty": t.suggest_categorical("penalty", ["l2", None]),
        "tol": t.suggest_float("tol", 1e-5, 1e-2, log=True),
        "C": t.suggest_float("C", 0.01, 100.0, log=True),
        "fit_intercept": t.suggest_categorical("fit_intercept", [True, False]),
        "solver": t.suggest_categorical("solver", ["lbfgs", "newton-cg", "saga"]),
        "max_iter": t.suggest_int("max_iter", 200, 2000),
    }


def _ridge(t) -> dict:
    return {
        "alpha": t.suggest_float("alpha", 0.01, 100.0, log=True),
        "fit_intercept": t.suggest_categorical("fit_intercept", [True, False]),
        "max_iter": t.suggest_int("max_iter", 100, 2000),
        "tol": t.suggest_float("tol", 1e-5, 1e-2, log=True),
        "solver": t.suggest_categorical("solver", ["auto", "sparse_cg", "sag",
                                                   "saga"]),
    }


def _sgd(t) -> dict:
    return {
        "penalty": t.suggest_categorical("penalty", ["l2", "l1", "elasticnet"]),
        "alpha": t.suggest_float("alpha", 1e-6, 1e-1, log=True),
        "l1_ratio": t.suggest_float("l1_ratio", 0.0, 1.0),
        "fit_intercept": t.suggest_categorical("fit_intercept", [True, False]),
        "max_iter": t.suggest_int("max_iter", 200, 3000),
        "tol": t.suggest_float("tol", 1e-5, 1e-2, log=True),
        "learning_rate": t.suggest_categorical(
            "learning_rate", ["constant", "optimal", "invscaling", "adaptive"]),
        "eta0": t.suggest_float("eta0", 1e-5, 1.0, log=True),
        "early_stopping": t.suggest_categorical("early_stopping", [True, False]),
        "validation_fraction": t.suggest_float("validation_fraction", 0.05, 0.3),
        "n_iter_no_change": t.suggest_int("n_iter_no_change", 1, 10),
    }


def _passive_aggressive(t) -> dict:
    return {
        "C": t.suggest_float("C", 1e-4, 1e4, log=True),
        "fit_intercept": t.suggest_categorical("fit_intercept", [True, False]),
        "max_iter": t.suggest_int("max_iter", 100, 3000),
        "tol": t.suggest_float("tol", 1e-5, 1e-1, log=True),
        "early_stopping": t.suggest_categorical("early_stopping", [True, False]),
        "validation_fraction": t.suggest_float("validation_fraction", 0.05, 0.3),
        "n_iter_no_change": t.suggest_int("n_iter_no_change", 1, 10),
        "shuffle": t.suggest_categorical("shuffle", [True, False]),
        "loss": t.suggest_categorical("loss", ["hinge", "squared_hinge"]),
        "class_weight": t.suggest_categorical("class_weight", [None, "balanced"]),
        "average": t.suggest_categorical("average", [False, True]),
    }


def _perceptron(t) -> dict:
    return {
        "penalty": t.suggest_categorical("penalty",
                                         [None, "l2", "l1", "elasticnet"]),
        "alpha": t.suggest_float("alpha", 1e-7, 1e-1, log=True),
        "l1_ratio": t.suggest_float("l1_ratio", 0.0, 1.0),
        "fit_intercept": t.suggest_categorical("fit_intercept", [True, False]),
        "max_iter": t.suggest_int("max_iter", 100, 3000),
        "tol": t.suggest_float("tol", 1e-5, 1e-1, log=True),
        "shuffle": t.suggest_categorical("shuffle", [True, False]),
        "eta0": t.suggest_float("eta0", 1e-3, 10, log=True),
        "early_stopping": t.suggest_categorical("early_stopping", [True, False]),
        "validation_fraction": t.suggest_float("validation_fraction", 0.05, 0.3),
        "n_iter_no_change": t.suggest_int("n_iter_no_change", 1, 10),
        "class_weight": t.suggest_categorical("class_weight", [None, "balanced"]),
    }


def _lda(t) -> dict:
    return {
        "shrinkage": t.suggest_categorical("shrinkage",
                                           [None, "auto", 0.1, 0.5, 0.9]),
        "n_components": t.suggest_int("n_components", 1, 10),
        "tol": t.suggest_float("tol", 1e-5, 1e-2, log=True),
    }


def _qda(t) -> dict:
    return {
        "pca__n_components": t.suggest_int("pca__n_components", 5, 60),
        "reg_param": t.suggest_float("reg_param", 0.0, 1.0),
        "store_covariance": t.suggest_categorical("store_covariance",
                                                  [True, False]),
        "tol": t.suggest_float("tol", 1e-7, 1e-1, log=True),
    }


def _gaussian_nb(t) -> dict:
    return {"var_smoothing": t.suggest_float("var_smoothing", 1e-11, 1e-5,
                                             log=True)}


def _bernoulli_nb(t) -> dict:
    return {
        "alpha": t.suggest_float("alpha", 1e-10, 10.0, log=True),
        "binarize": t.suggest_float("binarize", 0.0, 1.0),
        "fit_prior": t.suggest_categorical("fit_prior", [True, False]),
    }


def _knn(t) -> dict:
    return {
        "n_neighbors": t.suggest_int("n_neighbors", 1, 50),
        "algorithm": t.suggest_categorical("algorithm",
                                           ["auto", "ball_tree", "kd_tree",
                                            "brute"]),
        "leaf_size": t.suggest_int("leaf_size", 10, 100),
        "p": t.suggest_int("p", 1, 5),
        "metric": t.suggest_categorical("metric", ["minkowski", "euclidean",
                                                   "manhattan"]),
    }


def _nearest_centroid(t) -> dict:
    return {
        "metric": t.suggest_categorical("metric", ["euclidean", "manhattan"]),
        "shrink_threshold": t.suggest_float("shrink_threshold", 0.0, 1.0),
    }


def _svc_rbf(t) -> dict:
    return {
        "C": t.suggest_float("C", 0.01, 100.0, log=True),
        "gamma": t.suggest_categorical("gamma", ["scale", "auto"]),
        "tol": t.suggest_float("tol", 1e-4, 1e-2, log=True),
        "shrinking": t.suggest_categorical("shrinking", [True, False]),
        "class_weight": t.suggest_categorical("class_weight", [None, "balanced"]),
        "decision_function_shape": t.suggest_categorical(
            "decision_function_shape", ["ovo", "ovr"]),
    }


def _svc_poly(t) -> dict:
    space = _svc_rbf(t)
    space["degree"] = t.suggest_int("degree", 2, 5)
    space["coef0"] = t.suggest_float("coef0", 0.0, 1.0)
    return space


def _linear_svc(t) -> dict:
    return {
        "loss": t.suggest_categorical("loss", ["hinge", "squared_hinge"]),
        "tol": t.suggest_float("tol", 1e-5, 1e-2, log=True),
        "C": t.suggest_float("C", 0.01, 100.0, log=True),
        "fit_intercept": t.suggest_categorical("fit_intercept", [True, False]),
        "intercept_scaling": t.suggest_float("intercept_scaling", 0.1, 10.0,
                                             log=True),
        "max_iter": t.suggest_int("max_iter", 500, 5000),
        "class_weight": t.suggest_categorical("class_weight", [None, "balanced"]),
    }


def _decision_tree(t) -> dict:
    return {
        "criterion": t.suggest_categorical("criterion",
                                           ["gini", "entropy", "log_loss"]),
        "splitter": t.suggest_categorical("splitter", ["best", "random"]),
        "max_depth": t.suggest_int("max_depth", 1, 50),
        "min_samples_split": t.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 20),
        "max_features": t.suggest_categorical("max_features",
                                              [None, "sqrt", "log2"]),
        "ccp_alpha": t.suggest_float("ccp_alpha", 0.0, 0.05),
    }


def _extra_tree(t) -> dict:
    return {
        "criterion": t.suggest_categorical("criterion",
                                           ["gini", "entropy", "log_loss"]),
        "max_depth": t.suggest_int("max_depth", 1, 50),
        "min_samples_split": t.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 20),
        "max_features": t.suggest_categorical("max_features",
                                              [None, "sqrt", "log2"]),
        "ccp_alpha": t.suggest_float("ccp_alpha", 0.0, 0.05),
    }


def _random_forest(t) -> dict:
    return {
        "n_estimators": t.suggest_int("n_estimators", 100, 500, step=50),
        "criterion": t.suggest_categorical("criterion",
                                           ["gini", "entropy", "log_loss"]),
        "max_depth": t.suggest_int("max_depth", 3, 40),
        "min_samples_split": t.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 10),
        "min_weight_fraction_leaf": t.suggest_float("min_weight_fraction_leaf",
                                                    0.0, 0.2),
        "max_features": t.suggest_categorical("max_features",
                                              ["sqrt", "log2", None]),
        "bootstrap": t.suggest_categorical("bootstrap", [True, False]),
        "ccp_alpha": t.suggest_float("ccp_alpha", 0.0, 0.1),
        "max_samples": t.suggest_categorical("max_samples",
                                             [None, 0.7, 0.8, 0.9]),
    }


def _extra_trees(t) -> dict:
    return {
        "n_estimators": t.suggest_int("n_estimators", 50, 500),
        "criterion": t.suggest_categorical("criterion",
                                           ["gini", "entropy", "log_loss"]),
        "max_depth": t.suggest_int("max_depth", 3, 60),
        "min_samples_split": t.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 20),
        "max_features": t.suggest_categorical("max_features",
                                              ["sqrt", "log2", None]),
        "bootstrap": t.suggest_categorical("bootstrap", [True, False]),
        "ccp_alpha": t.suggest_float("ccp_alpha", 0.0, 0.05),
    }


def _bagging(t) -> dict:
    return {
        "n_estimators": t.suggest_int("n_estimators", 10, 200),
        "max_samples": t.suggest_float("max_samples", 0.5, 1.0),
        "max_features": t.suggest_float("max_features", 0.5, 1.0),
        "bootstrap": t.suggest_categorical("bootstrap", [True, False]),
        "estimator__max_depth": t.suggest_int("estimator__max_depth", 3, 30),
        "estimator__min_samples_leaf": t.suggest_int(
            "estimator__min_samples_leaf", 1, 20),
    }


def _adaboost(t) -> dict:
    return {
        "n_estimators": t.suggest_int("n_estimators", 50, 300),
        "learning_rate": t.suggest_float("learning_rate", 0.01, 2.0, log=True),
    }


def _gradient_boosting(t) -> dict:
    return {
        "learning_rate": t.suggest_float("learning_rate", 0.01, 1.0, log=True),
        "n_estimators": t.suggest_int("n_estimators", 50, 400),
        "subsample": t.suggest_float("subsample", 0.5, 1.0),
        "criterion": t.suggest_categorical("criterion",
                                           ["friedman_mse", "squared_error"]),
        "min_samples_split": t.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 20),
        "max_depth": t.suggest_int("max_depth", 3, 20),
        "max_features": t.suggest_categorical("max_features",
                                              ["sqrt", "log2", None]),
        "ccp_alpha": t.suggest_float("ccp_alpha", 0.0, 0.05),
    }


def _hist_gradient_boosting(t) -> dict:
    return {
        "learning_rate": t.suggest_float("learning_rate", 0.01, 1.0, log=True),
        "max_iter": t.suggest_int("max_iter", 50, 500),
        "max_leaf_nodes": t.suggest_int("max_leaf_nodes", 10, 100),
        "max_depth": t.suggest_int("max_depth", 3, 30),
        "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 50),
        "l2_regularization": t.suggest_float("l2_regularization", 0.0, 1.0),
        "max_bins": t.suggest_int("max_bins", 50, 255),
        "early_stopping": t.suggest_categorical("early_stopping", [True, False]),
        "n_iter_no_change": t.suggest_int("n_iter_no_change", 5, 20),
    }


def _xgboost(t) -> dict:
    return {
        "n_estimators": t.suggest_int("n_estimators", 50, 500),
        "max_depth": t.suggest_int("max_depth", 3, 12),
        "learning_rate": t.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "gamma": t.suggest_float("gamma", 0.0, 5.0),
        "min_child_weight": t.suggest_float("min_child_weight", 1.0, 10.0),
        "subsample": t.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": t.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": t.suggest_float("reg_alpha", 0.0, 10.0),
        "reg_lambda": t.suggest_float("reg_lambda", 0.0, 10.0),
    }


def _lightgbm(t) -> dict:
    return {
        "boosting_type": t.suggest_categorical("boosting_type",
                                               ["gbdt", "dart", "rf"]),
        "num_leaves": t.suggest_int("num_leaves", 31, 256),
        "max_depth": t.suggest_int("max_depth", 3, 50),
        "learning_rate": t.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "n_estimators": t.suggest_int("n_estimators", 50, 500),
        "min_child_weight": t.suggest_float("min_child_weight", 1e-3, 10.0),
        "min_child_samples": t.suggest_int("min_child_samples", 5, 50),
        "subsample": t.suggest_float("subsample", 0.5, 1.0),
        "subsample_freq": t.suggest_int("subsample_freq", 0, 10),
        "colsample_bytree": t.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": t.suggest_float("reg_alpha", 0.0, 10.0),
        "reg_lambda": t.suggest_float("reg_lambda", 0.0, 10.0),
    }


def _catboost(t) -> dict:
    return {
        "iterations": t.suggest_int("iterations", 100, 500),
        "learning_rate": t.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "depth": t.suggest_int("depth", 3, 10),
        "l2_leaf_reg": t.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "bagging_temperature": t.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_strength": t.suggest_float("random_strength", 0.0, 10.0),
        "colsample_bylevel": t.suggest_float("colsample_bylevel", 0.5, 1.0),
        "bootstrap_type": t.suggest_categorical("bootstrap_type",
                                                ["Bayesian", "Bernoulli"]),
    }


def _mlp(t) -> dict:
    return {
        "hidden_layer_sizes": t.suggest_categorical(
            "hidden_layer_sizes", list(_MLP_LAYERS)),
        "activation": t.suggest_categorical("activation",
                                            ["relu", "tanh", "logistic"]),
        "solver": t.suggest_categorical("solver", ["adam", "sgd"]),
        "alpha": t.suggest_float("alpha", 1e-6, 1e-1, log=True),
        "learning_rate": t.suggest_categorical(
            "learning_rate", ["constant", "invscaling", "adaptive"]),
        "learning_rate_init": t.suggest_float("learning_rate_init", 1e-5, 1e-1,
                                              log=True),
        "max_iter": t.suggest_int("max_iter", 200, 1000),
        "tol": t.suggest_float("tol", 1e-6, 1e-3, log=True),
        "early_stopping": t.suggest_categorical("early_stopping", [True, False]),
    }


SPACES: dict[str, Callable] = {
    "Logistic Regression": _logistic,
    "Logistic Regression (balanced)": _logistic,
    "Ridge Classifier": _ridge,
    "SGD (hinge)": _sgd,
    "Passive Aggressive": _passive_aggressive,
    "Perceptron": _perceptron,
    "Linear Discriminant": _lda,
    "Quadratic Discriminant": _qda,
    "Gaussian Naive Bayes": _gaussian_nb,
    "Bernoulli Naive Bayes": _bernoulli_nb,
    "k-Nearest Neighbours": _knn,
    "k-NN (distance weighted)": _knn,
    "Nearest Centroid": _nearest_centroid,
    "SVM (RBF)": _svc_rbf,
    "SVM (polynomial)": _svc_poly,
    "Linear SVM": _linear_svc,
    "Decision Tree": _decision_tree,
    "Extra Tree": _extra_tree,
    "Random Forest": _random_forest,
    "Random Forest (balanced)": _random_forest,
    "Extra Trees": _extra_trees,
    "Bagged Trees": _bagging,
    "AdaBoost": _adaboost,
    "Gradient Boosting": _gradient_boosting,
    "Hist Gradient Boosting": _hist_gradient_boosting,
    "XGBoost": _xgboost,
    "LightGBM": _lightgbm,
    "CatBoost": _catboost,
    "MLP (256-128)": _mlp,
}

PINNED: dict[str, dict] = {
    "Logistic Regression": {"class_weight": None},
    "Logistic Regression (balanced)": {"class_weight": "balanced"},
    "Random Forest": {"class_weight": None},
    "Random Forest (balanced)": {"class_weight": "balanced_subsample"},
    "k-Nearest Neighbours": {"weights": "uniform"},
    "k-NN (distance weighted)": {"weights": "distance"},
}


def suggest(name: str, trial) -> dict:
    if name not in SPACES:
        raise KeyError(f"no search space for {name!r}")
    return SPACES[name](trial)


def build_tuned(name: str, params: dict, n_jobs: int = 1,
                n_classes: int | None = None):
    p = dict(params)
    p.update(PINNED.get(name, {}))

    if name == "Logistic Regression" or name == "Logistic Regression (balanced)":
        return LogisticRegression(random_state=SEED, **p)

    if name == "Ridge Classifier":
        return RidgeClassifier(random_state=SEED, **p)

    if name == "SGD (hinge)":
        if not p.get("early_stopping", False):
            p.pop("validation_fraction", None)
        if p.get("penalty") != "elasticnet":
            p.pop("l1_ratio", None)
        return SGDClassifier(loss="hinge", random_state=SEED, **p)

    if name == "Passive Aggressive":
        if not p.get("early_stopping", False):
            p.pop("validation_fraction", None)
        return PassiveAggressiveClassifier(random_state=SEED, **p)

    if name == "Perceptron":
        if not p.get("early_stopping", False):
            p.pop("validation_fraction", None)
        if p.get("penalty") != "elasticnet":
            p.pop("l1_ratio", None)
        return Perceptron(random_state=SEED, **p)

    if name == "Linear Discriminant":
        if n_classes is not None:
            p["n_components"] = min(p["n_components"], max(1, n_classes - 1))
        solver = "lsqr" if p.get("shrinkage") is not None else "svd"
        return LinearDiscriminantAnalysis(solver=solver, **p)

    if name == "Quadratic Discriminant":
        n_comp = p.pop("pca__n_components")
        return Pipeline([
            ("pca", PCA(n_components=n_comp, random_state=SEED)),
            ("qda", QuadraticDiscriminantAnalysis(**p)),
        ])

    if name == "Gaussian Naive Bayes":
        return GaussianNB(**p)

    if name == "Bernoulli Naive Bayes":
        return BernoulliNB(**p)

    if name in ("k-Nearest Neighbours", "k-NN (distance weighted)"):
        if p.get("metric") != "minkowski":
            p.pop("p", None)
        return KNeighborsClassifier(n_jobs=n_jobs, **p)

    if name == "Nearest Centroid":
        return NearestCentroid(**p)

    if name == "SVM (RBF)":
        return SVC(kernel="rbf", probability=True, max_iter=200_000,
                   cache_size=1000, random_state=SEED, **p)

    if name == "SVM (polynomial)":
        return SVC(kernel="poly", probability=True, max_iter=200_000,
                   cache_size=1000, random_state=SEED, **p)

    if name == "Linear SVM":
        return LinearSVC(dual="auto", random_state=SEED, **p)

    if name == "Decision Tree":
        return DecisionTreeClassifier(random_state=SEED, **p)

    if name == "Extra Tree":
        return ExtraTreeClassifier(random_state=SEED, **p)

    if name in ("Random Forest", "Random Forest (balanced)"):
        if not p.get("bootstrap", True):
            p["max_samples"] = None
        return RandomForestClassifier(n_jobs=n_jobs, random_state=SEED,
                                      oob_score=False, **p)

    if name == "Extra Trees":
        return ExtraTreesClassifier(n_jobs=n_jobs, random_state=SEED, **p)

    if name == "Bagged Trees":
        depth = p.pop("estimator__max_depth")
        leaf = p.pop("estimator__min_samples_leaf")
        return BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=depth,
                                             min_samples_leaf=leaf,
                                             random_state=SEED),
            n_jobs=n_jobs, random_state=SEED, **p)

    if name == "AdaBoost":
        return AdaBoostClassifier(random_state=SEED, **p)

    if name == "Gradient Boosting":
        return GradientBoostingClassifier(random_state=SEED, **p)

    if name == "Hist Gradient Boosting":
        if not p.get("early_stopping", False):
            p.pop("n_iter_no_change", None)
        return HistGradientBoostingClassifier(random_state=SEED, **p)

    if name == "XGBoost":
        from xgboost import XGBClassifier
        return XGBClassifier(tree_method="hist", n_jobs=n_jobs,
                             random_state=SEED, verbosity=0,
                             eval_metric="mlogloss", **p)

    if name == "LightGBM":
        if p.get("boosting_type") == "rf":
            p["subsample_freq"] = max(1, p.get("subsample_freq", 1))
            p["subsample"] = min(p.get("subsample", 0.9), 0.99)
        return __import__("lightgbm").LGBMClassifier(
            n_jobs=n_jobs, random_state=SEED, verbose=-1, **p)

    if name == "CatBoost":
        from catboost import CatBoostClassifier
        if p.get("bootstrap_type") != "Bayesian":
            p.pop("bagging_temperature", None)
        return CatBoostClassifier(boosting_type="Plain", thread_count=n_jobs,
                                  random_seed=SEED, verbose=0,
                                  allow_writing_files=False, **p)

    if name == "MLP (256-128)":
        p["hidden_layer_sizes"] = _MLP_LAYERS[str(p["hidden_layer_sizes"])]
        if not p.get("early_stopping", False):
            p.pop("n_iter_no_change", None)
        return MLPClassifier(random_state=SEED, **p)

    raise KeyError(f"no builder for {name!r}")


def build_meta(name: str, tuned: dict, n_jobs: int = 1):
    from mlate import models as zoo

    def base(model_name: str, key: str):
        params = tuned.get(model_name)
        if params:
            try:
                return (key, build_tuned(model_name, params, n_jobs=n_jobs))
            except Exception:
                pass
        return (key, zoo.build(model_name, n_jobs=n_jobs))

    estimators = [base("Random Forest", "rf"),
                  base("XGBoost", "xgb"),
                  base("Logistic Regression", "lr")]

    if name == "Soft Voting (RF+XGB+LR)":
        return VotingClassifier(estimators=estimators, voting="soft", n_jobs=1)
    if name == "Stacking (RF+XGB+LR -> LR)":
        return StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=3000,
                                               random_state=SEED),
            cv=5, n_jobs=1, passthrough=False)
    raise KeyError(f"not a meta-ensemble: {name!r}")


def coverage() -> dict:
    from mlate import models as zoo
    names = zoo.names()
    return {
        "searched": [n for n in names if n in SPACES],
        "composed": [n for n in names if n in META],
        "untuned": [n for n in names if n in BASELINES],
        "unhandled": [n for n in names
                      if n not in SPACES and n not in META
                      and n not in BASELINES],
    }
