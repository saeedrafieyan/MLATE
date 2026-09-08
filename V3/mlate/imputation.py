from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor

warnings.filterwarnings(
    'ignore', message='.*Early stopping criterion not reached.*',
    category=ConvergenceWarning)

from mlate import config as cfg


class AmbientFill(BaseEstimator, TransformerMixin):

    def __init__(self, value: float = cfg.AMBIENT_TEMPERATURE_C,
                 add_indicator: bool = True):
        self.value = value
        self.add_indicator = add_indicator

    def fit(self, X, y=None):
        self.feature_names_in_ = list(pd.DataFrame(X).columns)
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        out = X.fillna(self.value)
        if self.add_indicator:
            for c in X.columns:
                out[f"{c} [reported]"] = X[c].notna().astype(float)
        return out.to_numpy(dtype=float)

    def get_feature_names_out(self, input_features=None):
        names = list(input_features if input_features is not None
                     else self.feature_names_in_)
        if self.add_indicator:
            names += [f"{c} [reported]" for c in names]
        return np.asarray(names, dtype=object)


class PresentMedianFill(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.feature_names_in_ = list(X.columns)
        self.fill_ = {}
        for c in X.columns:
            present = X[c][X[c].fillna(0) != 0]
            self.fill_[c] = float(present.median()) if len(present) else 0.0
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for c in X.columns:
            X[c] = X[c].fillna(self.fill_.get(c, 0.0))
        return X.to_numpy(dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features if input_features is not None
                          else self.feature_names_in_, dtype=object)


def _make_estimator(name: str, seed: int):
    if name == "extra_trees":
        return ExtraTreesRegressor(n_estimators=cfg.IMPUTER_N_ESTIMATORS,
                                   max_depth=12,
                                   n_jobs=cfg.N_JOBS, random_state=seed)
    if name == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=cfg.IMPUTER_N_ESTIMATORS * 3,
                            max_depth=6, learning_rate=0.1,
                            tree_method="hist", n_jobs=cfg.N_JOBS,
                            verbosity=0, random_state=seed)
    raise ValueError(f"unknown imputer estimator: {name}")


class MedianFill(BaseEstimator, TransformerMixin):

    def __init__(self, add_indicator: bool = True):
        self.add_indicator = add_indicator

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.feature_names_in_ = list(X.columns)
        self.fill_ = {c: (float(X[c].median()) if X[c].notna().any() else 0.0)
                      for c in X.columns}
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        out = X.copy()
        for c in X.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(self.fill_[c])
        if self.add_indicator:
            for c in X.columns:
                out[f"{c} [reported]"] = X[c].notna().astype(float)
        return out.to_numpy(dtype=float)

    def get_feature_names_out(self, input_features=None):
        names = list(input_features if input_features is not None
                     else self.feature_names_in_)
        if self.add_indicator:
            names += [f"{c} [reported]" for c in names]
        return np.asarray(names, dtype=object)


class ModelBasedFill(BaseEstimator, TransformerMixin):

    def __init__(self, add_indicator: bool = True,
                 random_state: int = cfg.RANDOM_STATE,
                 max_iter: int = cfg.IMPUTER_MAX_ITER,
                 estimator: str = cfg.IMPUTER_ESTIMATOR):
        self.add_indicator = add_indicator
        self.random_state = random_state
        self.max_iter = max_iter
        self.estimator = estimator

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.feature_names_in_ = list(X.columns)
        self.imputer_ = IterativeImputer(
            estimator=_make_estimator(self.estimator, self.random_state),
            max_iter=self.max_iter, random_state=self.random_state,
            initial_strategy="median", skip_complete=True,
        )
        self.imputer_.fit(X)
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        filled = pd.DataFrame(self.imputer_.transform(X), columns=X.columns,
                              index=X.index)
        if self.add_indicator:
            for c in X.columns:
                filled[f"{c} [reported]"] = X[c].notna().astype(float).values
        return filled.to_numpy(dtype=float)

    def get_feature_names_out(self, input_features=None):
        names = list(input_features if input_features is not None
                     else self.feature_names_in_)
        if self.add_indicator:
            names += [f"{c} [reported]" for c in names]
        return np.asarray(names, dtype=object)


class StoredFill(BaseEstimator, TransformerMixin):

    def __init__(self, values: dict[str, float], add_indicator: bool = True):
        self.values = values
        self.add_indicator = add_indicator

    def fit(self, X, y=None):
        self.feature_names_in_ = list(pd.DataFrame(X).columns)
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        out = X.copy()
        for c in X.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(
                self.values.get(c, 0.0))
        if self.add_indicator:
            for c in X.columns:
                out[f"{c} [reported]"] = X[c].notna().astype(float)
        return out.to_numpy(dtype=float)

    def get_feature_names_out(self, input_features=None):
        names = list(input_features if input_features is not None
                     else self.feature_names_in_)
        if self.add_indicator:
            names += [f"{c} [reported]" for c in names]
        return np.asarray(names, dtype=object)


class StoredGroupFill(BaseEstimator, TransformerMixin):

    def __init__(self, by_group: dict, fallback: float):
        self.by_group = by_group
        self.fallback = fallback

    def fit(self, X, y=None):
        self.feature_names_in_ = list(pd.DataFrame(X).columns)
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        group, value = X.columns[0], X.columns[1]
        v = pd.to_numeric(X[value], errors="coerce")
        mapped = X[group].map(self.by_group).astype(float)
        return v.fillna(mapped).fillna(self.fallback).to_numpy(
            dtype=float).reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        return np.asarray([cfg.CELL_COLS[1]], dtype=object)


class CellDensityFill(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.feature_names_in_ = list(X.columns)
        line, dens = X.columns[0], X.columns[1]
        d = pd.to_numeric(X[dens], errors="coerce")
        self.by_line_ = d.groupby(X[line]).median().to_dict()
        self.global_ = float(d.median()) if d.notna().any() else 0.0
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        line, dens = X.columns[0], X.columns[1]
        d = pd.to_numeric(X[dens], errors="coerce")
        fill = X[line].map(self.by_line_).astype(float)
        return d.fillna(fill).fillna(self.global_).to_numpy(
            dtype=float).reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        return np.asarray([cfg.CELL_COLS[1]], dtype=object)


def fill_within_study(df: pd.DataFrame, columns, group: str = "DOI"
                      ) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    records = []
    for col in columns.print_params:
        missing = out[col].isna()
        if not missing.any():
            continue
        study_median = out.groupby(group)[col].transform("median")
        filled = missing & study_median.notna()
        if not filled.any():
            continue
        records.append(pd.DataFrame({
            "row": out.index[filled],
            group: out.loc[filled, group].to_numpy(),
            "column": col,
            "filled_value": study_median[filled].to_numpy(),
            "n_reported_in_study": out.loc[filled, group].map(
                out.groupby(group)[col].count()).to_numpy(),
        }))
        out.loc[filled, col] = study_median[filled]
    audit = (pd.concat(records, ignore_index=True) if records
             else pd.DataFrame(columns=["row", group, "column",
                                        "filled_value",
                                        "n_reported_in_study"]))
    return out, audit
