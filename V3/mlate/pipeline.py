from __future__ import annotations

import numpy as np
import pandas as pd
from category_encoders.binary import BinaryEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from mlate import config as cfg
from mlate.dataset import Columns
from mlate.imputation import (
    AmbientFill, CellDensityFill, MedianFill, ModelBasedFill,
    PresentMedianFill,
)


def build_preprocessor(columns: Columns,
                       scale: bool = True,
                       model_based: bool = True) -> ColumnTransformer:
    def numeric(step):
        return Pipeline([("impute", step),
                         ("scale", MinMaxScaler() if scale else "passthrough")])

    unknown = [c for c in cfg.UNKNOWN_IF_MISSING if c in columns.print_params]
    ambient = [c for c in cfg.AMBIENT_COLUMNS if c in columns.print_params]

    if cfg.AMBIENT_STRATEGY == "model_based":
        unknown, ambient = unknown + ambient, []
    elif cfg.AMBIENT_STRATEGY != "constant":
        raise ValueError(f"unknown AMBIENT_STRATEGY: {cfg.AMBIENT_STRATEGY}")

    transformers = [
        ("cell_line", BinaryEncoder(
            cols=[columns.cell_line], return_df=False,
            handle_unknown="value", handle_missing="value"),
         [columns.cell_line]),
        ("cell_density", numeric(CellDensityFill()),
         [columns.cell_line, columns.cell_density]),
        ("biomaterials", numeric(PresentMedianFill()), columns.biomaterials),
        ("printing", numeric(
            ModelBasedFill() if (model_based and
                                 cfg.PRINTING_IMPUTATION == "model_based")
            else MedianFill()), unknown),
    ]
    if ambient:
        transformers.insert(3, ("ambient", numeric(AmbientFill()), ambient))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
        n_jobs=None,
    )


def feature_names(preprocessor: ColumnTransformer) -> list[str]:
    try:
        return [str(n) for n in preprocessor.get_feature_names_out()]
    except Exception:
        n = sum(
            t.transform(np.zeros((1, len(c)))).shape[1] if hasattr(t, "transform")
            else len(c)
            for _, t, c in preprocessor.transformers_ if c
        )
        return [f"f{i}" for i in range(n)]


def fit_transform_fold(df: pd.DataFrame, columns: Columns,
                       train_idx, test_idx, **kw
                       ) -> tuple[np.ndarray, np.ndarray, ColumnTransformer]:
    pre = build_preprocessor(columns, **kw)
    X = df[columns.predictors]
    X_train = pre.fit_transform(X.iloc[train_idx])
    X_test = pre.transform(X.iloc[test_idx])
    return np.asarray(X_train, float), np.asarray(X_test, float), pre
