"""
Supervised benchmark
====================

    python 04_machine_learning/run.py                  # everything
    python 04_machine_learning/run.py --tasks printability --protocols doi
    python 04_machine_learning/run.py --force          # ignore checkpoints

Fits every model in the registry against every task under every validation
protocol and writes out-of-fold predictions, one row per test sample per model.
Metrics are computed downstream in report.py from those predictions, so the
scoring can be changed or extended without refitting anything.

Protocols
---------
random   StratifiedKFold over samples. Answers "can the model interpolate
         within this corpus?". 97% of test rows share a publication with the
         training set under this scheme, so it is reported as an upper bound
         and never as a generalisation estimate.
doi      StratifiedGroupKFold on DOI. Answers "will this transfer to a study
         the model has not seen?", which is the question the tool actually
         faces and the one reviewers R1-2 and R2-5 raise.
tissue   Leave-one-tissue-out, with every study represented in the held-out
         tissue also removed from training. The hardest protocol: unseen
         tissue and unseen laboratory at once.

Scheduling
----------
The preprocessor is fitted once per fold and reused by all 30 models; refitting
it per model would repeat the most expensive shared step 30 times. Models are
then run in two passes. Models that parallelise internally are run one at a
time with the full CPU budget, because a 500-tree forest saturates the machine
on its own. Models that do not are run concurrently, since each would otherwise
occupy one core and leave the rest idle. Mixing the two in a single pool
oversubscribes the CPU and runs slower than either arrangement alone.

Checkpointing
-------------
Predictions are written per task and protocol as soon as that combination
finishes, and an existing file is skipped unless --force is passed. A run that
dies at hour three resumes rather than restarting.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from mlate import config as cfg
from mlate import models as zoo
from mlate import resources, splits
from mlate.artifacts import fold_preprocessor
from mlate.dataset import TASKS, load_dataset, target_frame

warnings.filterwarnings("ignore")

TABLES = cfg.step_dir("04_machine_learning", "tables")
PREDS = TABLES / "predictions"
PROTOCOLS = ("random", "doi", "tissue")


def _fit_one(name: str, Xtr, ytr, Xte, n_jobs: int, device: str | None,
             classes: np.ndarray):
    """
    Fit and predict a single model. Never raises: a failure is a result.

    Labels are encoded to 0..n-1 before fitting and decoded afterwards. XGBoost
    rejects any other labelling outright, and Cell Response runs 1-5, so without
    this the strongest boosting model silently disappears from two of the three
    tasks. Encoding centrally rather than per-model keeps every classifier on
    identical inputs. `classes` is the task-level label set, not the fold's, so
    the encoding is stable across folds even where a fold lacks a class.
    """
    t0 = time.perf_counter()
    try:
        code = {c: i for i, c in enumerate(classes)}
        ytr_enc = np.asarray([code[v] for v in ytr])
        model = zoo.build(name, n_jobs=n_jobs, device=device)
        model.fit(Xtr, ytr_enc)
        pred = classes[np.asarray(model.predict(Xte)).ravel().astype(int)]
        # Full probability vector, aligned to model.classes_, so ROC-AUC, log
        # loss and the Brier score can be computed downstream. Storing only the
        # maximum would make all three impossible.
        # NB: not `classes`, which is the parameter holding the task label set.
        proba, seen = None, None
        if hasattr(model, "predict_proba"):
            try:
                proba = np.asarray(model.predict_proba(Xte), dtype=float)
                seen = classes[np.asarray(model.classes_).astype(int)]
            except Exception:
                proba, seen = None, None
        return {"model": name, "pred": pred, "proba": proba,
                "classes": seen, "seconds": time.perf_counter() - t0,
                "error": ""}
    except Exception as exc:
        return {"model": name, "pred": None, "proba": None, "classes": None,
                "seconds": time.perf_counter() - t0,
                "error": f"{type(exc).__name__}: {exc}"[:300]}


def run_fold(df, columns, fold, y, model_names, budget, devices) -> list[dict]:
    # Fitted on this fold's training rows only, then applied to both sides.
    pre, _, _ = fold_preprocessor(df, columns, fold.train_idx)
    features = df[columns.predictors]
    Xtr = np.asarray(pre.transform(features.iloc[fold.train_idx]), dtype=float)
    Xte = np.asarray(pre.transform(features.iloc[fold.test_idx]), dtype=float)
    ytr = y.to_numpy()[fold.train_idx]
    yte = y.to_numpy()[fold.test_idx]

    all_labels = np.asarray(sorted(pd.unique(y)))
    threaded = [n for n in model_names if zoo.REGISTRY[n].threaded]
    light = [n for n in model_names if not zoo.REGISTRY[n].threaded]

    results = []
    # Serial pass: each model gets the whole CPU, and a GPU if it wants one.
    for i, name in enumerate(threaded):
        device = (devices[i % len(devices)]
                  if (zoo.REGISTRY[name].gpu and devices) else None)
        results.append(_fit_one(name, Xtr, ytr, Xte, budget.n_jobs, device,
                                all_labels))
    # Concurrent pass: many single-threaded models at once.
    if light:
        results += Parallel(n_jobs=min(len(light), budget.n_jobs),
                            backend="loky", verbose=0)(
            delayed(_fit_one)(n, Xtr, ytr, Xte, 1, None, all_labels)
            for n in light)

    unseen = splits.unseen_material_mask(df, fold, columns.biomaterials)
    rows = []
    for r in results:
        if r["pred"] is None:
            rows.append({"model": r["model"], "fold": fold.name,
                         "protocol": fold.protocol, "row": -1,
                         "y_true": -1, "y_pred": -1, "confidence": np.nan,
                         "unseen_material": False,
                         "seconds": r["seconds"], "error": r["error"]})
            continue
        block = pd.DataFrame({
            "model": r["model"], "fold": fold.name, "protocol": fold.protocol,
            "row": df.index.to_numpy()[fold.test_idx],
            "y_true": yte, "y_pred": r["pred"],
            "unseen_material": unseen,
            "seconds": r["seconds"], "error": "",
        })
        # One column per class label, NaN for models without predict_proba.
        # Columns are keyed by the label itself, not the model's column order,
        # so a fold missing a class cannot silently shift the alignment.
        for c in all_labels:
            block[f"p_{c}"] = np.nan
        if r["proba"] is not None and r["classes"] is not None:
            for j, c in enumerate(r["classes"]):
                block[f"p_{c}"] = r["proba"][:, j]
            block["confidence"] = r["proba"].max(axis=1)
        else:
            block["confidence"] = np.nan
        rows.append(block)
    frames = [r for r in rows if isinstance(r, pd.DataFrame)]
    failures = [r for r in rows if isinstance(r, dict)]
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return out, pd.DataFrame(failures)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", nargs="*", default=list(TASKS))
    ap.add_argument("--protocols", nargs="*", default=list(PROTOCOLS))
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    budget = resources.claim()
    devices = resources.devices()
    PREDS.mkdir(parents=True, exist_ok=True)

    df, columns = load_dataset()
    model_names = args.models or zoo.names()
    print(f"\n{len(model_names)} models | {len(args.tasks)} tasks | "
          f"{len(args.protocols)} protocols")
    zoo.summary().to_excel(TABLES / "model_registry.xlsx", index=False)

    timings = []
    for task in args.tasks:
        sub, y = target_frame(df, task)
        for protocol in args.protocols:
            out_path = PREDS / f"{task}__{protocol}.parquet"
            if out_path.exists() and not args.force:
                print(f"  skip {task} / {protocol} (checkpoint present)")
                continue

            folds = splits.make_folds(sub, y, protocols=(protocol,))
            print(f"\n{task} / {protocol}: {len(folds)} folds, "
                  f"{len(sub)} rows, classes {sorted(y.unique())}")

            parts, fails = [], []
            for fold in folds:
                t0 = time.perf_counter()
                got, failed = run_fold(sub, columns, fold, y, model_names,
                                       budget, devices)
                parts.append(got)
                if len(failed):
                    fails.append(failed)
                print(f"  {fold.name:28s} train {len(fold.train_idx):5d} "
                      f"test {len(fold.test_idx):5d}  "
                      f"{time.perf_counter() - t0:6.1f}s"
                      + (f"  [{len(failed)} failed]" if len(failed) else ""))
                timings.append({"task": task, "protocol": protocol,
                                "fold": fold.name,
                                "seconds": time.perf_counter() - t0})

            preds = pd.concat(parts, ignore_index=True)
            preds.to_parquet(out_path, index=False)
            if fails:
                pd.concat(fails, ignore_index=True).to_excel(
                    TABLES / f"failures__{task}__{protocol}.xlsx", index=False)
            print(f"  -> {out_path.name}  ({len(preds):,} predictions)")

    if timings:
        pd.DataFrame(timings).to_excel(TABLES / "fold_timings.xlsx",
                                       index=False)
    print(f"\ntables -> {TABLES}")


if __name__ == "__main__":
    main()
