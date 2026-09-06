"""
Build and freeze the split manifests and preprocessing artefacts.

    python -m preprocessing.run

Writes to results/preprocessing/:
    folds_<task>.csv          every fold's row indices, as a manifest
    fold_summary.xlsx         one row per fold, for the supplement
    leakage_report.xlsx       explicit evidence that grouped folds do not leak
    feature_space.xlsx        what the preprocessor emits, per task

Downstream stages read the manifests instead of re-deriving splits, so
clustering, ML and DL are all scored on exactly the same partitions.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import numpy as np
import pandas as pd

from mlate import config as cfg
from mlate.dataset import (
    TASKS, load_dataset, modeling_tissue, target_frame,
)
from mlate import splits
from mlate.pipeline import build_preprocessor, feature_names

OUT = cfg.step_dir("02_preprocessing", "tables")


def leakage_report(df: pd.DataFrame, folds: list[splits.Fold]) -> pd.DataFrame:
    """
    Quantify, per fold, how much study overlap the protocol allows.

    A random fold should show a high sibling rate; a DOI fold must show zero
    shared studies. Printing both side by side is the evidence for the
    revised validation strategy.
    """
    doi = df["DOI"].to_numpy()
    rows = []
    for f in folds:
        tr, te = set(doi[f.train_idx]), doi[f.test_idx]
        shared = np.isin(te, list(tr))
        rows.append({
            "fold": f.name,
            "protocol": f.protocol,
            "n_test": len(te),
            "test_rows_whose_study_is_in_train": int(shared.sum()),
            "share_%": round(100 * shared.mean(), 1),
            "shared_studies": len(tr & set(te)),
        })
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df, columns = load_dataset()
    print(f"{len(df)} samples | {df['DOI'].nunique()} studies "
          f"| {len(columns.predictors)} predictor columns")

    tissue = modeling_tissue(df)
    print(f"modelling tissues: {tissue.nunique()} "
          f"(min {cfg.MIN_TISSUE_SAMPLES} samples, smaller pooled)")

    summaries, leaks, spaces = [], [], []

    for task in TASKS:
        sub, y = target_frame(df, task)
        folds = splits.make_folds(sub.reset_index(drop=True),
                                  y.reset_index(drop=True))

        s = splits.describe(folds, y.reset_index(drop=True))
        s.insert(0, "task", task)
        summaries.append(s)

        lr = leakage_report(sub.reset_index(drop=True), folds)
        lr.insert(0, "task", task)
        leaks.append(lr)

        # persist row indices so downstream stages cannot drift
        manifest = pd.concat([
            pd.DataFrame({"fold": f.name, "protocol": f.protocol,
                          "row": f.test_idx, "split": "test"})
            for f in folds
        ], ignore_index=True)
        manifest.to_csv(OUT / f"folds_{task}.csv", index=False)

        # what the model will actually see, fitted on one representative fold
        ref = next(f for f in folds if f.protocol == "doi")
        pre = build_preprocessor(columns)
        Xt = pre.fit_transform(sub[columns.predictors].iloc[ref.train_idx])
        names = feature_names(pre)
        spaces.append(pd.DataFrame({
            "task": task, "n_rows": len(sub), "n_classes": y.nunique(),
            "n_input_columns": len(columns.predictors),
            "n_output_features": Xt.shape[1],
            "n_indicator_features": sum("[reported]" in n for n in names),
        }, index=[0]))

        n_unseen = sum(
            splits.unseen_material_mask(
                sub.reset_index(drop=True), f, columns.biomaterials).sum()
            for f in folds if f.protocol == "doi")
        print(f"  {task:<24} rows {len(sub):>5} | classes {y.nunique()} "
              f"| folds {len(folds):>2} | features {Xt.shape[1]:>3} "
              f"| grouped test rows with an unseen material: {n_unseen}")

    with pd.ExcelWriter(OUT / "fold_summary.xlsx") as xl:
        pd.concat(summaries, ignore_index=True).to_excel(
            xl, sheet_name="folds", index=False)
        pd.concat(spaces, ignore_index=True).to_excel(
            xl, sheet_name="feature_space", index=False)

    leak = pd.concat(leaks, ignore_index=True)
    leak.to_excel(OUT / "leakage_report.xlsx", index=False)

    print("\nstudy overlap between train and test, by protocol")
    print(leak.groupby("protocol")[
        ["share_%", "shared_studies"]].mean().round(1).to_string())
    print(f"\nartefacts -> {OUT}")


if __name__ == "__main__":
    main()
