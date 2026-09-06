"""
Imputation benchmark
====================

    python -m preprocessing.validate_imputation

Design
------
Studies are split with GroupKFold on DOI. For each fold the imputer is fitted
on the training studies and applied to the held-out studies, 10% of whose
observed printing-parameter values have been masked. All seven columns are
masked together, because scoring one at a time would leave the other six
complete and hand the imputer information it never has in production.

The grouping is not a formality. Under ungrouped masking, KNN imputation looked
like the clear winner - but 69% of rows carry a biomaterial-plus-cell-line
signature unique to one study, so a masked row's nearest neighbours are its own
paper's other rows and the imputer is recalling a sibling rather than inferring
anything. Production imputers are fitted inside DOI-grouped folds and never get
that shortcut, so the benchmark must not either.

Strategies
----------
    median                    per-column median. Baseline.
    constant 22 C / 25 C      ambient assumption, temperature columns only.
    KNN (k=5)                 distance-weighted neighbours.
    iterative + BayesianRidge linear multivariate.
    iterative + RandomForest  bagged trees.
    iterative + ExtraTrees    randomised trees.
    iterative + XGBoost       gradient boosting - what the manuscript claims.

Metrics
-------
    MAE, RMSE            in each column's own units
    nMAE                 MAE divided by the column's interquartile range, so
                         columns on wildly different scales are comparable
    R^2                  coefficient of determination on the masked values

Caveat that belongs in the paper
--------------------------------
Only reported values can be masked, and reported values are overwhelmingly the
deliberately heated or cooled runs. The benchmark is therefore biased against a
constant ambient fill, which exists to serve the unreported rows. Treat the
constant's score as a lower bound, not a refutation.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

from mlate import config as cfg
from mlate.dataset import load_dataset

warnings.filterwarnings("ignore", category=ConvergenceWarning)

OUT = cfg.step_dir("02_preprocessing", "tables")
MASK_FRACTION = 0.10
REPEATS = 5
MAX_ITER = cfg.IMPUTER_MAX_ITER   # sklearn stops early once tol is met


def _iterative(estimator):
    return IterativeImputer(estimator=estimator, max_iter=MAX_ITER,
                            random_state=cfg.RANDOM_STATE,
                            initial_strategy="median", skip_complete=True)


def strategies() -> dict:
    from xgboost import XGBRegressor
    return {
        "median": SimpleImputer(strategy="median"),
        "KNN (k=5)": KNNImputer(n_neighbors=5, weights="distance"),
        "iterative + BayesianRidge": _iterative(BayesianRidge()),
        "iterative + RandomForest": _iterative(RandomForestRegressor(
            n_estimators=cfg.IMPUTER_N_ESTIMATORS, max_depth=12, n_jobs=cfg.N_JOBS,
            random_state=cfg.RANDOM_STATE)),
        "iterative + ExtraTrees": _iterative(ExtraTreesRegressor(
            n_estimators=cfg.IMPUTER_N_ESTIMATORS, max_depth=12, n_jobs=cfg.N_JOBS,
            random_state=cfg.RANDOM_STATE)),
        "iterative + XGBoost": _iterative(XGBRegressor(
            n_estimators=cfg.IMPUTER_N_ESTIMATORS * 3, max_depth=6, learning_rate=0.1,
            tree_method="hist", n_jobs=cfg.N_JOBS, verbosity=0,
            random_state=cfg.RANDOM_STATE)),
    }


def _score(truth, pred, iqr) -> dict:
    err = np.asarray(pred) - np.asarray(truth)
    mae = float(np.mean(np.abs(err)))
    return {
        "MAE": mae,
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "nMAE": mae / iqr if iqr > 0 else np.nan,
        "R2": float(r2_score(truth, pred)) if len(set(truth)) > 1 else np.nan,
    }


def run() -> pd.DataFrame:
    """
    Study-grouped masking benchmark.

    The imputer is fitted on the training studies of a DOI-grouped fold and
    applied to the held-out studies, whose observed values are partly masked.
    This matters more than it might appear. Under ungrouped masking a
    neighbour-based imputer scores extremely well, but only because a masked
    row's nearest neighbours are its own paper's other rows - 69% of rows carry
    a biomaterial-plus-cell-line signature unique to one study, so the imputer
    is recalling a sibling rather than inferring a value. Fitting on separate
    studies removes that shortcut and measures what the pipeline actually does,
    since production imputers are fitted inside DOI-grouped folds.
    """
    df, columns = load_dataset()
    targets = [c for c in cfg.PRINT_PARAMS if df[c].isna().any()]
    numeric = columns.biomaterials + [columns.cell_density] + cfg.PRINT_PARAMS
    iqr = {c: float(df[c].quantile(.75) - df[c].quantile(.25)) for c in targets}
    groups = df["DOI"].to_numpy()

    records = []
    gkf = GroupKFold(n_splits=REPEATS)
    for fold, (tr, te) in enumerate(gkf.split(df, groups=groups)):
        rng = np.random.default_rng(cfg.RANDOM_STATE + fold)
        train = df.iloc[tr][numeric].astype(float)
        test = df.iloc[te][numeric].astype(float).copy()
        truth_rows = df.iloc[te]

        masked: dict[str, np.ndarray] = {}
        for col in targets:
            obs = np.flatnonzero(truth_rows[col].notna().to_numpy())
            if len(obs) < 5:
                continue
            take = rng.choice(obs, size=max(5, int(len(obs) * MASK_FRACTION)),
                              replace=False)
            masked[col] = take
            test.iloc[take, test.columns.get_loc(col)] = np.nan

        for name, imputer in strategies().items():
            imputer.fit(train)
            filled = pd.DataFrame(imputer.transform(test), columns=numeric)
            for col, pos in masked.items():
                records.append({
                    "column": col, "strategy": name, "repeat": fold,
                    "n_masked": len(pos),
                    **_score(truth_rows[col].to_numpy(float)[pos],
                             filled[col].to_numpy(float)[pos], iqr[col]),
                    "truth": truth_rows[col].to_numpy(float)[pos],
                    "pred": filled[col].to_numpy(float)[pos],
                })

        # Constants are only meaningful where 'not reported' plausibly means
        # ambient, so they are scored on the temperature columns alone and
        # ranked separately - averaging them against methods scored on all
        # seven columns would not be a like-for-like comparison.
        for col in cfg.AMBIENT_COLUMNS:
            if col not in masked:
                continue
            pos = masked[col]
            truth = truth_rows[col].to_numpy(float)[pos]
            for label, value in [(f"constant {cfg.AMBIENT_TEMPERATURE_C:g} C",
                                  cfg.AMBIENT_TEMPERATURE_C),
                                 ("constant 25 C", 25.0)]:
                records.append({
                    "column": col, "strategy": label, "repeat": fold,
                    "n_masked": len(pos),
                    **_score(truth, np.full(len(pos), value), iqr[col]),
                    "truth": truth,
                    "pred": np.full(len(pos), value),
                })
        print(f"  fold {fold + 1}/{REPEATS} done "
              f"({len(set(groups[tr]))} train studies, "
              f"{len(set(groups[te]))} held out)")

    return pd.DataFrame(records)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    per_repeat = run()

    # Pooled R2 per strategy: all masked values from every column and fold in
    # one regression score. The mean of per-column R2 is not usable as a
    # headline - a single column scoring -356 moves the average arbitrarily -
    # so pooling is what gets reported, with the per-column values kept for
    # detail.
    def _pool(frame) -> pd.DataFrame:
        out = []
        for strat, g in frame.groupby("strategy"):
            t = np.concatenate(g["truth"].tolist())
            p_ = np.concatenate(g["pred"].tolist())
            out.append({"strategy": strat, "n_scored": int(len(t)),
                        "R2_pooled": float(r2_score(t, p_)),
                        "MAE_pooled": float(np.mean(np.abs(p_ - t)))})
        return pd.DataFrame(out).sort_values("MAE_pooled")

    is_const = per_repeat["strategy"].str.startswith("constant")
    pooled = _pool(per_repeat[~is_const])
    pooled_temp = _pool(per_repeat[per_repeat["column"].isin(cfg.AMBIENT_COLUMNS)])

    per_repeat = per_repeat.drop(columns=["truth", "pred"])
    summary = (per_repeat
               .groupby(["column", "strategy"])[["MAE", "RMSE", "nMAE", "R2"]]
               .mean().reset_index())
    # Rank general-purpose imputers over every column; rank the constants
    # separately over the temperature columns only, alongside the general
    # methods restricted to those same two columns.
    constants = summary["strategy"].str.startswith("constant")
    overall = (summary[~constants].groupby("strategy")[["nMAE", "R2"]]
               .mean().sort_values("nMAE").reset_index())
    temps = summary[summary["column"].isin(cfg.AMBIENT_COLUMNS)]
    temp_rank = (temps.groupby("strategy")[["nMAE", "R2"]]
                 .mean().sort_values("nMAE").reset_index())
    best = (summary[~constants].loc[
        summary[~constants].groupby("column")["nMAE"].idxmin()]
            [["column", "strategy", "MAE", "nMAE", "R2"]])

    with pd.ExcelWriter(OUT / "imputation_validation.xlsx") as xl:
        overall.merge(pooled, on="strategy", how="left").to_excel(
            xl, sheet_name="overall_ranking", index=False)
        temp_rank.merge(pooled_temp, on="strategy", how="left").to_excel(
            xl, sheet_name="temperature_only", index=False)
        summary.to_excel(xl, sheet_name="per_column", index=False)
        best.to_excel(xl, sheet_name="best_per_column", index=False)
        per_repeat.to_excel(xl, sheet_name="per_repeat", index=False)

    print("\noverall ranking (mean over columns; nMAE lower is better)")
    print(overall.merge(pooled, on="strategy", how="left")
          .round(3).to_string(index=False))
    print(chr(10) + "temperature columns only")
    print(temp_rank.merge(pooled_temp, on="strategy", how="left")
          .round(3).to_string(index=False))
    print("\nbest strategy per column")
    print(best.round(3).to_string(index=False))
    print(f"\nresults -> {OUT / 'imputation_validation.xlsx'}")
    return summary, overall, per_repeat


if __name__ == "__main__":
    main()
