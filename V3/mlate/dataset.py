"""
Dataset access and the column contract
======================================

One loader, one definition of which columns are what. Every stage uses these
so a column can never be treated as a predictor in one place and metadata in
another.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from mlate import config as cfg


@dataclass(frozen=True)
class Columns:
    """Which columns play which role, resolved against an actual frame."""
    biomaterials: list[str]
    print_params: list[str]
    cell_line: str
    cell_density: str
    targets: list[str]

    @property
    def numeric_predictors(self) -> list[str]:
        return [*self.biomaterials, self.cell_density, *self.print_params]

    @property
    def predictors(self) -> list[str]:
        return [*self.biomaterials, self.cell_line, self.cell_density,
                *self.print_params]


def column_groups(df: pd.DataFrame) -> Columns:
    """
    Resolve the column contract against a loaded frame.

    Biomaterials are defined positionally - everything between the metadata
    block and 'Cell Line' - because their names change as the dataset grows.
    """
    cols = list(df.columns)
    start = len(cfg.META_FRONT)
    biomaterials = cols[start:cols.index(cfg.CELL_COLS[0])]

    missing = [c for c in cfg.PRINT_PARAMS + cfg.TARGETS + cfg.CELL_COLS
               if c not in cols]
    if missing:
        raise ValueError(f"dataset is missing expected columns: {missing}")
    overlap = set(biomaterials) & set(cfg.PRINT_PARAMS + cfg.TARGETS)
    if overlap:
        raise ValueError(f"biomaterial block overlaps other roles: {overlap}")

    return Columns(
        biomaterials=biomaterials,
        print_params=list(cfg.PRINT_PARAMS),
        cell_line=cfg.CELL_COLS[0],
        cell_density=cfg.CELL_COLS[1],
        targets=list(cfg.TARGETS),
    )


def load_dataset(path=None) -> tuple[pd.DataFrame, Columns]:
    """Load the published dataset and its column contract."""
    df = pd.read_excel(path or cfg.DATASET)
    return df, column_groups(df)


def load_taxonomy() -> pd.DataFrame:
    return pd.read_csv(cfg.TAXONOMY)


def modeling_tissue(df: pd.DataFrame) -> pd.Series:
    """
    Tissue label used for leave-one-tissue-out.

    Collapses the four non-anatomical target_tissue values into
    'not_organ_specific', then pools any tissue with too few samples to give a
    meaningful held-out fold.
    """
    t = df["target_tissue"].where(
        ~df["target_tissue"].isin(cfg.NON_ORGAN_TISSUES), "not_organ_specific")
    small = t.value_counts().loc[lambda s: s < cfg.MIN_TISSUE_SAMPLES].index
    return t.where(~t.isin(small), "other_small_tissue")


def is_cellular(df: pd.DataFrame) -> pd.Series:
    """True where the construct actually contained cells."""
    return df[cfg.CELL_COLS[0]] != cfg.ACELLULAR_TOKEN


def target_frame(df: pd.DataFrame, task: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Rows and labels for one prediction task.

    'printability'          all rows, classes 0-3
    'cell_response'         all rows, classes 1-5 (comparable to the submitted
                            version, where class 1 means 'no cells')
    'cell_response_cellular' cellular rows only, classes 2-5 (reviewer R2-3)
    """
    if task == "printability":
        return df, df["Printability"].astype(int)
    if task == "cell_response":
        return df, df["Cell Response"].astype(int)
    if task == "cell_response_cellular":
        mask = is_cellular(df) & df["Cell Response"].isin(
            cfg.CELL_RESPONSE_CELLULAR)
        sub = df[mask]
        return sub, sub["Cell Response"].astype(int)
    raise ValueError(f"unknown task: {task}")


TASKS = ("printability", "cell_response", "cell_response_cellular")
