"""
Tiered imputation
=================

Blank does not mean the same thing in every column of this dataset, so a
single imputer would be wrong. Four rules, each matched to what a missing
value actually signifies:

Syringe / Substrate Temperature
    Missing because the study printed at ambient and did not think it worth
    reporting. The missingness is informative, so it is filled with a constant
    (22 C, the modal reported value) plus a '_reported' indicator. A
    model-based imputer would be actively harmful here: it learns from the
    rows that DO report a temperature, which are exactly the deliberately
    heated and cooled ones, and would push ambient samples toward controlled
    values.

Pressure, speeds, crosslinking durations, nozzle diameter
    No physical default exists; blank genuinely means unknown. Model-based
    imputation plus a '_reported' indicator.

Biomaterial concentrations
    Per the dataset construction rule, absent = 0 and unreported = blank. A
    blank therefore means the material IS in the formulation at an unstated
    concentration, so it takes the median concentration of that material
    across samples where it is present. Filling 0 would silently delete an
    ingredient.

Cell density
    Strongly cell-line specific, so the median for the same cell line is used
    where available, falling back to the global median.

Every estimator here learns from training rows only. Under grouped
cross-validation the whole pipeline is refitted inside each fold, so no
information crosses the train/test boundary through an imputed value.
"""

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
    """Constant fill for informatively-missing columns, plus an indicator."""

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
    """
    Biomaterial rule: blank means present-but-unreported.

    Fills with the median over training samples where the material is present
    (non-zero). Materials never present in training fall back to 0.
    """

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
    """Regressor driving IterativeImputer. Benchmarked in
    preprocessing/validate_imputation.py."""
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
    """
    Column median plus a [reported] indicator. The default for printing
    parameters.

    This looks like the naive choice and is in fact the evidence-based one.
    Under study-grouped masking - the honest test, matching how the pipeline
    actually runs - the median beats every multivariate imputer tried
    (validate_imputation.py: nMAE 1.30 against 1.54 for iterative ExtraTrees,
    1.90 for XGBoost, 1.99 for KNN), and every method including this one has a
    negative R^2. Printing parameters simply are not predictable across studies:
    knowing a formulation's composition tells you very little about the pressure
    or nozzle speed some other laboratory chose.

    Sophisticated imputation looked far better under ungrouped masking only
    because it was recovering values from the same paper's other rows. What
    carries real information here is not the filled value but the indicator
    saying the value was never reported.
    """

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
    """
    Iterative (multivariate) imputation plus a reported-indicator.

    Retained for the benchmark and as a config option, but NOT the default:
    see MedianFill for why it loses once study leakage is removed. It is also
    the reason the fitted pipeline was hundreds of megabytes, since
    IterativeImputer keeps every sub-model to replay at transform time.
    """

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
    """
    Fill from pre-computed per-column values, plus the same [reported]
    indicator the training imputers emit.

    This is the deployment counterpart of ModelBasedFill. A fitted
    IterativeImputer retains every sub-model so it can replay the imputation
    sequence, which makes it hundreds of megabytes - fine as a training-time
    object, impossible to ship. At inference the user supplies the formulation,
    so there is normally nothing to impute at all; this transformer exists only
    to keep the feature space identical and to cover a field left blank.

    The stored values are the medians of what the training imputer actually
    produced, so a blank field lands where the heavy imputer would have put it.
    """

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
    """
    Deployment counterpart of CellDensityFill.

    Carries the per-cell-line medians learned during training so a blank
    density at inference lands on the same value the training pipeline would
    have used. Filling with a single global median instead would make the
    shipped preprocessor disagree with the one the models were trained on.
    """

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
    """
    Median density of the same cell line, falling back to the global median.

    Expects two columns, in order: [cell line, cell density].
    """

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
    """
    Tier 1 of the two-tier fill: copy a printing parameter from the same study.

    Printing parameters are chosen once per publication and reused for every
    formulation in it - 87-98% of each column's variance lies between studies,
    and 46-94% of studies report a single constant value. So where a study
    reports a parameter in some of its rows and leaves it blank in others, the
    reported value is direct evidence for the blank ones, not a guess. Filling
    it is more accurate than any model, and it is auditable in a way a model is
    not.

    This covers 585 of the 6,152 missing cells (9.5%). The other 90.5% sit in
    studies that never reported the parameter at all, so no within-study
    evidence exists and they fall through to tier 2 - the global median or the
    22 C ambient constant - in build_preprocessor().

    Used for the published feature matrix and the clustering inputs. It is
    deliberately NOT part of the supervised pipeline, where the imputer is
    refitted inside each fold; see 02_preprocessing/METHODS.md section 4.

    Returns the filled frame and a per-cell audit of what tier 1 changed.
    """
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
