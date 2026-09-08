from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupShuffleSplit, StratifiedGroupKFold, StratifiedKFold, train_test_split,
)

from mlate import config as cfg
from mlate.dataset import modeling_tissue


@dataclass(frozen=True)
class Fold:
    name: str
    train_idx: np.ndarray
    test_idx: np.ndarray
    protocol: str
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.test_idx)


def _assert_no_group_leak(train_idx, test_idx, groups: np.ndarray,
                          label: str) -> None:
    shared = set(groups[train_idx]) & set(groups[test_idx])
    if shared:
        raise AssertionError(
            f"{label}: {len(shared)} group(s) appear in both train and test, "
            f"e.g. {sorted(shared)[:3]}")


def random_folds(y: pd.Series, n_splits: int = cfg.N_FOLDS,
                 seed: int = cfg.RANDOM_STATE) -> list[Fold]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    idx = np.arange(len(y))
    return [
        Fold(f"random_fold{i}", idx[tr], idx[te], "random", {"seed": seed})
        for i, (tr, te) in enumerate(skf.split(idx, y))
    ]


def random_holdout(y: pd.Series, test_size: float = cfg.TEST_SIZE,
                   seed: int = cfg.RANDOM_STATE,
                   stratify: pd.Series | None = None) -> Fold:
    idx = np.arange(len(y))
    key = y if stratify is None else stratify
    tr, te = train_test_split(idx, test_size=test_size, stratify=key,
                              random_state=seed, shuffle=True)
    return Fold("random_holdout", tr, te, "random",
                {"seed": seed, "test_size": test_size,
                 "stratified_on": getattr(key, "name", "target"),
                 "shared_across_targets": stratify is not None})


def doi_holdout(y: pd.Series, groups: pd.Series,
                test_size: float = cfg.TEST_SIZE,
                seed: int = cfg.RANDOM_STATE) -> Fold:
    g = groups.to_numpy()
    idx = np.arange(len(y))
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(gss.split(idx, y, groups=g))
    _assert_no_group_leak(tr, te, g, "doi_holdout")
    return Fold(
        "doi_holdout", idx[tr], idx[te], "doi",
        {"seed": seed, "requested_test_size": test_size,
         "realised_test_size": round(len(te) / len(idx), 4),
         "n_train_studies": len(set(g[tr])),
         "n_test_studies": len(set(g[te]))},
    )


def make_holdout(df: pd.DataFrame, y: pd.Series, protocol: str) -> list[Fold]:
    if protocol == "random":
        shared = df[cfg.TARGETS[0]] if cfg.TARGETS[0] in df.columns else None
        return [random_holdout(y, stratify=shared)]
    if protocol == "doi":
        return [doi_holdout(y, df["DOI"])]
    raise ValueError(f"no hold-out defined for protocol {protocol!r}")


def doi_folds(y: pd.Series, groups: pd.Series, n_splits: int = cfg.N_FOLDS,
              seed: int = cfg.RANDOM_STATE) -> list[Fold]:
    g = groups.to_numpy()
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                random_state=seed)
    idx = np.arange(len(y))
    folds = []
    for i, (tr, te) in enumerate(sgkf.split(idx, y, groups=g)):
        _assert_no_group_leak(tr, te, g, f"doi_fold{i}")
        folds.append(Fold(
            f"doi_fold{i}", idx[tr], idx[te], "doi",
            {"seed": seed,
             "n_train_studies": len(set(g[tr])),
             "n_test_studies": len(set(g[te]))},
        ))
    return folds


def tissue_folds(y: pd.Series, tissues: pd.Series, doi: pd.Series,
                 min_test: int = 30) -> list[Fold]:
    t = tissues.to_numpy()
    g = doi.to_numpy()
    idx = np.arange(len(y))
    folds = []
    for tissue in pd.unique(t):
        te = idx[t == tissue]
        contaminated = set(g[te])
        tr = idx[(t != tissue) & ~np.isin(g, list(contaminated))]
        naive = int(((t != tissue).sum()) - len(tr))
        if len(te) < min_test or y.iloc[tr].nunique() < y.nunique():
            continue
        _assert_no_group_leak(tr, te, t, f"tissue_{tissue}")
        _assert_no_group_leak(tr, te, g, f"tissue_{tissue} (study)")
        folds.append(Fold(
            f"tissue_{tissue}", tr, te, "tissue",
            {"held_out_tissue": tissue, "n_test": len(te),
             "train_rows_dropped_for_study_overlap": naive,
             "n_train_studies": len(set(g[tr]))},
        ))
    return folds


def make_folds(df: pd.DataFrame, y: pd.Series,
               protocols=("random", "doi", "tissue")) -> list[Fold]:
    folds: list[Fold] = []
    if "random" in protocols:
        folds += random_folds(y)
    if "doi" in protocols:
        folds += doi_folds(y, df["DOI"])
    if "tissue" in protocols:
        folds += tissue_folds(y, modeling_tissue(df), df["DOI"])
    return folds


def unseen_material_mask(df: pd.DataFrame, fold: Fold,
                         biomaterials: list[str]) -> np.ndarray:
    present = (df[biomaterials].fillna(0) != 0).to_numpy()
    seen = present[fold.train_idx].any(axis=0)
    return present[fold.test_idx][:, ~seen].any(axis=1)


def describe(folds: list[Fold], y: pd.Series) -> pd.DataFrame:
    rows = []
    for f in folds:
        rows.append({
            "fold": f.name,
            "protocol": f.protocol,
            "n_train": len(f.train_idx),
            "n_test": len(f.test_idx),
            "test_classes": y.iloc[f.test_idx].nunique(),
            "train_classes": y.iloc[f.train_idx].nunique(),
            **f.meta,
        })
    return pd.DataFrame(rows)
