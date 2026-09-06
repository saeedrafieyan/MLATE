"""
Fitted-artefact persistence
===========================

Fitting a preprocessor is expensive, so it is done once and reused. Two kinds
of artefact, with different purposes:

fold preprocessors
    Fitted on one fold's TRAINING rows only, cached under results/. Every model
    scored on that fold reuses the same object, so the imputer is fitted once
    per fold rather than once per model per hyper-parameter trial. Cached by a
    hash of the training row set, so tasks that share a partition - printability
    and cell_response run on the same 2,646 rows - share one artefact.
    These are working files and are NOT committed.

deployment preprocessor
    Fitted on the whole dataset and shipped with the release. This is what the
    web application loads to transform a user's formulation, and what a reader
    needs to reproduce inference. Committed, alongside the feature-name list and
    a manifest recording library versions and file hashes.

A note on size: IterativeImputer keeps every fitted sub-model so it can replay
the imputation sequence at transform time, so an artefact grows linearly with
max_iter (roughly 8 MB per iteration here). Keep cfg.IMPUTER_MAX_ITER at a value
justified by preprocessing/sweep_iterations.py, or the release becomes
undistributable.
"""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from mlate import config as cfg
from mlate.dataset import Columns
from mlate.pipeline import build_preprocessor, feature_names

CACHE_DIR = cfg.step_dir("02_preprocessing", "models") / "fold_cache"
RELEASE_DIR = cfg.PREPROCESSOR_DIR
COMPRESS = 3          # joblib zlib level; trees compress well


def _config_fingerprint() -> str:
    """Anything that changes the fitted object must change the cache key."""
    payload = {
        "estimator": cfg.IMPUTER_ESTIMATOR,
        "max_iter": cfg.IMPUTER_MAX_ITER,
        "ambient_strategy": cfg.AMBIENT_STRATEGY,
        "ambient_value": cfg.AMBIENT_TEMPERATURE_C,
        "seed": cfg.RANDOM_STATE,
        "sklearn": sklearn.__version__,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def cache_key(train_idx: np.ndarray, columns: Columns | None = None) -> str:
    """
    Cache identity of a fitted fold preprocessor.

    The predictor list is part of the key, not just the rows and the imputation
    settings. A preprocessor is fitted against a specific column contract, so
    two runs that disagree about which columns are predictors must not share an
    artefact - otherwise changing the input set silently reloads a transformer
    fitted on the old one, and the mismatch surfaces much later as a shape
    error, or does not surface at all.
    """
    rows = hashlib.sha256(
        np.asarray(sorted(train_idx), dtype=np.int64).tobytes()).hexdigest()[:16]
    if columns is None:
        return f"{rows}_{_config_fingerprint()}"
    cols = hashlib.sha256(
        json.dumps(list(columns.predictors)).encode()).hexdigest()[:8]
    return f"{rows}_{_config_fingerprint()}_{cols}"


def fold_preprocessor(df: pd.DataFrame, columns: Columns,
                      train_idx: np.ndarray, use_cache: bool = True):
    """
    Fitted preprocessor for one fold's training rows.

    Returns (preprocessor, path, was_cached). Fitting uses training rows only,
    so the cached object can be reused for every model scored on that fold
    without leaking anything.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"pre_{cache_key(train_idx, columns)}.pkl"
    if use_cache and path.exists():
        return joblib.load(path), path, True

    pre = build_preprocessor(columns)
    pre.fit(df[columns.predictors].iloc[train_idx])
    joblib.dump(pre, path, compress=COMPRESS)
    return pre, path, False


def build_release(df: pd.DataFrame, columns: Columns) -> dict:
    """
    Fit and save the artefacts that ship with the paper.

    The deployment preprocessor is fitted on every row on purpose: at inference
    time there is no held-out set to protect, and a user's formulation should be
    transformed using everything the dataset knows. It is never used to score a
    model - that is what the fold preprocessors are for.
    """
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)

    # Exactly the same object the models were trained behind, fitted on every
    # row. There was once a separate lightweight deployment variant, needed
    # only because an IterativeImputer-based pipeline weighed hundreds of
    # megabytes; with median imputation the real pipeline is ~10 kB, so keeping
    # a second implementation in step would only invite the two to drift apart.
    pre = build_preprocessor(columns).fit(df[columns.predictors])
    names = feature_names(pre)

    joblib.dump(pre, RELEASE_DIR / "preprocessor.pkl", compress=COMPRESS)
    joblib.dump(list(columns.predictors), RELEASE_DIR / "input_columns.pkl",
                compress=COMPRESS)
    joblib.dump(names, RELEASE_DIR / "feature_names.pkl", compress=COMPRESS)

    # Targets are already integer-coded; ship the human-readable meanings so the
    # application never has to hard-code them.
    schema = {
        "Printability": {
            "0": "the ink was not extruded",
            "1": "the ink behaved like a liquid, formed beads",
            "2": "extrudable but not optimised",
            "3": "extrudable and optimised",
        },
        "Cell Response": {
            "1": "no cells included - not applicable",
            "2": "inadequate short-term response",
            "3": "good short-term response",
            "4": "good short-term, poor long-term",
            "5": "good short- and long-term response",
        },
        "acellular_token": cfg.ACELLULAR_TOKEN,
    }
    (RELEASE_DIR / "target_schema.json").write_text(
        json.dumps(schema, indent=2), encoding="utf-8")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "config": {
            "imputer_estimator": cfg.IMPUTER_ESTIMATOR,
            "imputer_max_iter": cfg.IMPUTER_MAX_ITER,
            "ambient_strategy": cfg.AMBIENT_STRATEGY,
            "ambient_temperature_c": cfg.AMBIENT_TEMPERATURE_C,
            "random_state": cfg.RANDOM_STATE,
        },
        "dataset": {
            "rows": int(len(df)),
            "input_columns": len(columns.predictors),
            "output_features": len(names),
            "indicator_features": sum("[reported]" in n for n in names),
        },
        "files": {},
    }
    for p in sorted(RELEASE_DIR.iterdir()):
        if p.name == "manifest.json":
            continue
        manifest["files"][p.name] = {
            "bytes": p.stat().st_size,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        }
    (RELEASE_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_release():
    """Load the shipped preprocessor. Used by the web application."""
    pre = joblib.load(RELEASE_DIR / "preprocessor.pkl")
    cols = joblib.load(RELEASE_DIR / "input_columns.pkl")
    names = joblib.load(RELEASE_DIR / "feature_names.pkl")
    schema = json.loads((RELEASE_DIR / "target_schema.json").read_text("utf-8"))
    return pre, cols, names, schema


if __name__ == "__main__":
    from mlate.dataset import load_dataset

    frame, cols = load_dataset()
    info = build_release(frame, cols)
    total = sum(f["bytes"] for f in info["files"].values())
    print(f"release artefacts -> {RELEASE_DIR}")
    for name, meta in info["files"].items():
        print(f"  {name:<28} {meta['bytes'] / 2**20:8.1f} MB")
    print(f"  {'TOTAL':<28} {total / 2**20:8.1f} MB"
          + ("   << exceeds GitHub's 100 MB file limit"
             if any(f["bytes"] > 100 * 2**20 for f in info["files"].values())
             else ""))
