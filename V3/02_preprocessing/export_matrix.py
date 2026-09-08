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

    pre = build_preprocessor(columns).fit(filled[columns.predictors])
    matrix = pd.DataFrame(pre.transform(filled[columns.predictors]),
                          columns=feature_names(pre), index=filled.index)
    matrix.to_parquet(OUT / "feature_matrix.parquet")

    raw = build_preprocessor(columns, scale=False).fit(filled[columns.predictors])
    interpretable = pd.DataFrame(raw.transform(filled[columns.predictors]),
                                 columns=feature_names(raw), index=filled.index)
    keep = [c for c in interpretable.columns if "[reported]" not in c]
    interpretable[keep].to_parquet(OUT / "imputed_raw.parquet")

    with pd.ExcelWriter(OUT / "imputation_tiers.xlsx") as xl:
        tiers.to_excel(xl, sheet_name="summary", index=False)
        tier1.to_excel(xl, sheet_name="tier1_cells", index=False)

    assert matrix.notna().all().all(), "feature matrix still contains NaN"
    assert interpretable[keep].notna().all().all(), "raw matrix contains NaN"
    print(f"\nfeature_matrix.parquet  {matrix.shape[0]} x {matrix.shape[1]}"
          f"   ({sum('[reported]' in c for c in matrix.columns)} indicators)")
    print(f"imputed_raw.parquet     {len(interpretable)} x {len(keep)}")
    print(f"imputation_tiers.xlsx   {len(tier1)} tier-1 cells")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
