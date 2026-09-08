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
COMPRESS = 3


def _config_fingerprint() -> str:
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
    rows = hashlib.sha256(
        np.asarray(sorted(train_idx), dtype=np.int64).tobytes()).hexdigest()[:16]
    if columns is None:
        return f"{rows}_{_config_fingerprint()}"
    cols = hashlib.sha256(
        json.dumps(list(columns.predictors)).encode()).hexdigest()[:8]
    return f"{rows}_{_config_fingerprint()}_{cols}"


def fold_preprocessor(df: pd.DataFrame, columns: Columns,
                      train_idx: np.ndarray, use_cache: bool = True):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"pre_{cache_key(train_idx, columns)}.pkl"
    if use_cache and path.exists():
        return joblib.load(path), path, True

    pre = build_preprocessor(columns)
    pre.fit(df[columns.predictors].iloc[train_idx])
    joblib.dump(pre, path, compress=COMPRESS)
    return pre, path, False


def build_release(df: pd.DataFrame, columns: Columns) -> dict:
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)

    pre = build_preprocessor(columns).fit(df[columns.predictors])
    names = feature_names(pre)

    joblib.dump(pre, RELEASE_DIR / "preprocessor.pkl", compress=COMPRESS)
    joblib.dump(list(columns.predictors), RELEASE_DIR / "input_columns.pkl",
                compress=COMPRESS)
    joblib.dump(names, RELEASE_DIR / "feature_names.pkl", compress=COMPRESS)

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
