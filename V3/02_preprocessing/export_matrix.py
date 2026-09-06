"""
Export the imputed feature matrix
=================================

    python 02_preprocessing/export_matrix.py

Clustering and the descriptive analyses need a complete numeric matrix, and
re-deriving it in each step would let the steps drift apart. This writes it
once, fitted on the whole dataset.

The published dataset in 01_data/processed/ is not touched. Everything here is
derived and lands in results/02_preprocessing/tables/.

Two-tier fill
-------------
tier 1  the same study reports the parameter in another row, so the value is
        copied from it. Direct evidence, 585 cells.
tier 2  the study never reported the parameter, so no within-study evidence
        exists and the global median - or 22 C for the two temperatures - is
        used. 5,567 cells. Study-grouped benchmarking showed no method beats
        this; see 02_preprocessing/METHODS.md section 5.

Every filled cell is recorded in imputation_tiers.xlsx so the split between
evidence and placeholder is auditable rather than asserted.

Three files:

    feature_matrix.parquet    153 model-ready features - encoded, imputed and
                              MinMax scaled. This is what clustering consumes.
    imputed_raw.parquet       the same rows in their original units, imputed but
                              not encoded or scaled, for tables and figures that
                              need interpretable values.
    imputation_tiers.xlsx     which cells were filled, by which tier.

NOT for supervised evaluation. Both matrices are fitted on every row, so a
supervised model scored against them would be reading imputation and scaling
statistics derived partly from its own test set. Supervised steps must call
mlate.artifacts.fold_preprocessor(), which fits inside a fold.

Only the modelling columns are carried through: 130 biomaterials, cell line,
cell density and 7 printing parameters. Reference, DOI, tissue labels, cancer
flag and the duplicate-audit columns are excluded from every transformation -
they are provenance, not predictors. DOI is read for the tier-1 grouping and
then dropped. The row index is preserved so any of them can be joined back on
afterwards.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from mlate import config as cfg
from mlate.dataset import load_dataset
from mlate.imputation import fill_within_study
from mlate.pipeline import build_preprocessor, feature_names

OUT = cfg.step_dir("02_preprocessing", "tables")


def main() -> None:
    df, columns = load_dataset()

    excluded = [c for c in df.columns if c not in columns.predictors]
    print(f"predictors: {len(columns.predictors)}  "
          f"({len(columns.biomaterials)} biomaterials + cell line + density "
          f"+ {len(columns.print_params)} printing parameters)")
    print(f"excluded  : {excluded}")

    # ── tier 1: copy from the same study ────────────────────────────────────
    before = df[columns.print_params].isna().sum()
    filled, tier1 = fill_within_study(df, columns)
    after = filled[columns.print_params].isna().sum()

    tiers = pd.DataFrame({
        "column": columns.print_params,
        "reported": [int(df[c].notna().sum()) for c in columns.print_params],
        "tier1_within_study": (before - after).reindex(
            columns.print_params).to_numpy(),
        "tier2_global": after.reindex(columns.print_params).to_numpy(),
    })
    tiers["pct_reported"] = 100 * tiers["reported"] / len(df)
    print("\nfill tiers")
    print(tiers.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    print(f"\ntier 1 filled {len(tier1)} cells "
          f"({100 * len(tier1) / int(before.sum()):.1f}% of the "
          f"{int(before.sum())} missing); "
          f"{int(after.sum())} fall through to tier 2")

    # ── tier 2 + encoding + scaling ─────────────────────────────────────────
    pre = build_preprocessor(columns).fit(filled[columns.predictors])
    matrix = pd.DataFrame(pre.transform(filled[columns.predictors]),
                          columns=feature_names(pre), index=filled.index)
    matrix.to_parquet(OUT / "feature_matrix.parquet")

    # Same imputation, original units, no encoding or scaling.
    raw = build_preprocessor(columns, scale=False).fit(filled[columns.predictors])
    interpretable = pd.DataFrame(raw.transform(filled[columns.predictors]),
                                 columns=feature_names(raw), index=filled.index)
    keep = [c for c in interpretable.columns if "[reported]" not in c]
    interpretable[keep].to_parquet(OUT / "imputed_raw.parquet")

    with pd.ExcelWriter(OUT / "imputation_tiers.xlsx") as xl:
        tiers.to_excel(xl, sheet_name="summary", index=False)
        tier1.to_excel(xl, sheet_name="tier1_cells", index=False)

    # The [reported] indicators are built from the post-tier-1 frame, so they
    # mark "known for this study" rather than "printed in this row" - which is
    # the distinction that carries information once tier 1 has run.
    assert matrix.notna().all().all(), "feature matrix still contains NaN"
    assert interpretable[keep].notna().all().all(), "raw matrix contains NaN"
    print(f"\nfeature_matrix.parquet  {matrix.shape[0]} x {matrix.shape[1]}"
          f"   ({sum('[reported]' in c for c in matrix.columns)} indicators)")
    print(f"imputed_raw.parquet     {len(interpretable)} x {len(keep)}")
    print(f"imputation_tiers.xlsx   {len(tier1)} tier-1 cells")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
