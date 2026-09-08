from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from mlate import config as cfg
from mlate.dataset import load_dataset

OUT = cfg.step_dir("02_preprocessing", "tables")
GROUP = "DOI"


def study_structure(df: pd.DataFrame, columns, group: str = GROUP) -> pd.DataFrame:
    rows = []
    for col in columns.print_params:
        obs = df[[col, group]].dropna()
        if obs.empty:
            continue
        by_study = obs.groupby(group)[col]
        grand = obs[col].mean()
        n = by_study.size()

        within = float(((obs[col] - by_study.transform("mean")) ** 2).sum())
        between = float((n * (by_study.mean() - grand) ** 2).sum())
        total = within + between

        repeated = n[n > 1].index
        single = by_study.nunique().reindex(repeated).eq(1)

        rows.append({
            "column": col,
            "n_reported": int(len(obs)),
            "n_studies": int(obs[group].nunique()),
            "pct_var_between_studies": 100 * between / total if total else float("nan"),
            "n_studies_repeated": int(len(repeated)),
            "pct_studies_single_value": 100 * float(single.mean()) if len(repeated) else float("nan"),
        })
    return pd.DataFrame(rows)


def main() -> None:
    df, columns = load_dataset()
    table = study_structure(df, columns)
    table.to_excel(OUT / "study_structure.xlsx", index=False)

    pd.set_option("display.width", 200)
    print(table.to_string(index=False, float_format=lambda v: f"{v:8.1f}"))
    print(f"\n-> {OUT / 'study_structure.xlsx'}")


if __name__ == "__main__":
    main()
