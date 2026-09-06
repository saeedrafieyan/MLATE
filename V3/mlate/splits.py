"""
Validation splits
=================

Three protocols, deliberately kept side by side because they answer different
questions and the paper reports all three.

random    stratified hold-out and K-fold. Estimates INTERPOLATION: predicting a
          formulation from a study you have already partly seen. Comparable to
          the originally submitted evaluation.

doi       StratifiedGroupKFold grouped on DOI, so no study appears in both
          train and test. Estimates EXTRAPOLATION to a new study, which is the
          condition the web application actually operates in. Primary result.

tissue    leave-one-tissue-out. Supports the cross-tissue claim.

Why grouping matters here: 89% of rows in a random 20% test set have a
same-study sibling in training, ~45% of label variance sits between studies,
and 68% of rows carry a material+cell-line signature unique to one study. A
model can therefore recognise the study and inherit its rating offset instead
of learning material-outcome relationships.
"""

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
    """One train/test partition, carrying enough context to audit it."""
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
    """Stratified K-fold ignoring study membership - the interpolation view."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    idx = np.arange(len(y))
    return [
        Fold(f"random_fold{i}", idx[tr], idx[te], "random", {"seed": seed})
        for i, (tr, te) in enumerate(skf.split(idx, y))
    ]


def random_holdout(y: pd.Series, test_size: float = cfg.TEST_SIZE,
                   seed: int = cfg.RANDOM_STATE,
                   stratify: pd.Series | None = None) -> Fold:
    """
    Single stratified hold-out, matching the originally submitted design.

    `stratify` defaults to `y`, but callers evaluating more than one target
    should pass ONE shared column so every target is scored on the same
    scaffold records. The submitted manuscript states this explicitly in 2.7:
    "A single shared split was used for both target variables to ensure that
    printability and cell-response models were evaluated on the same scaffold
    records."

    It is not only a matter of matching the submitted design. WSSQ combines a
    printability prediction and a cell-response prediction for the SAME
    scaffold; if the two targets are split independently their test sets only
    partially overlap, and WSSQ becomes computable on the intersection alone.
    """
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
    """
    Single study-grouped hold-out: the grouped analogue of `random_holdout`.

    GroupShuffleSplit partitions whole DOIs, so the requested fraction is met
    in studies rather than in rows and the realised test share will not land
    exactly on `test_size`. That is inherent to grouping - a study is
    indivisible - and the realised counts are recorded on the fold so the
    Methods can state what was actually held out rather than what was asked
    for.
    """
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
    """
    The single train/test partition for the hold-out evaluation design.

    Returned as a one-element list so callers can treat the hold-out and
    cross-validated designs through the same interface. Under this design the
    10-fold cross-validation happens INSIDE the training partition, for
    hyper-parameter selection only, and the returned test partition is scored
    exactly once.
    """
    # Stratified on Printability for every target, so the two targets share
    # one partition. The grouped hold-out is shared automatically, because
    # GroupShuffleSplit partitions on studies and never looks at the target.
    if protocol == "random":
        shared = df[cfg.TARGETS[0]] if cfg.TARGETS[0] in df.columns else None
        return [random_holdout(y, stratify=shared)]
    if protocol == "doi":
        return [doi_holdout(y, df["DOI"])]
    raise ValueError(f"no hold-out defined for protocol {protocol!r}")


def doi_folds(y: pd.Series, groups: pd.Series, n_splits: int = cfg.N_FOLDS,
              seed: int = cfg.RANDOM_STATE) -> list[Fold]:
    """
    Study-grouped stratified K-fold. The primary protocol.

    StratifiedGroupKFold balances the class distribution as well as it can
    while keeping every DOI wholly inside one fold; perfect stratification is
    not achievable under a grouping constraint, which is expected.
    """
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
    """
    Leave-one-tissue-out, with the study boundary enforced as well.

    A study can contribute rows to more than one tissue. Holding out a tissue
    without also removing those studies from training would let the model see
    the same lab, bioink system and rating style it is about to be tested on,
    which is the very effect the grouped protocol exists to remove. Any study
    represented in the held-out tissue is therefore dropped from training, and
    the count of rows this costs is recorded on the fold.
    """
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
    """Build every requested protocol for one task."""
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
    """
    Which test rows contain a material absent from this fold's training data.

    63 of 130 materials appear in exactly one study, so a grouped fold can
    hold out the only evidence for a material. Predictions for those rows are
    uninformed rather than merely unbiased, and the paper reports grouped
    performance split on this mask so the two effects are not conflated.
    """
    present = (df[biomaterials].fillna(0) != 0).to_numpy()
    seen = present[fold.train_idx].any(axis=0)
    return present[fold.test_idx][:, ~seen].any(axis=1)


def describe(folds: list[Fold], y: pd.Series) -> pd.DataFrame:
    """Summary of a fold set, for the supplement and for sanity checks."""
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
