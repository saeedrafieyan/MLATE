from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from mlate import config as cfg
from mlate import evaluation as ev
from mlate import resources, splits
from mlate.artifacts import fold_preprocessor
from mlate.dataset import load_dataset, target_frame

warnings.filterwarnings("ignore")

TABLES = cfg.step_dir("05_deep_learning", "tables")
UNITS = TABLES / "foundation" / "units"
PREDS = TABLES / "predictions_foundation"

PROTOCOLS = ("random", "doi")
N_ESTIMATORS = 8
MODEL_VERSION = "tabpfn-v2.6-classifier-v2.6_default.ckpt"
DEFAULT_MODELS = ("TabPFN", "TabPFN (thinking)", "TabICL")


ENV_FILES = (Path(__file__).resolve().parent / ".env", cfg.ROOT / ".env")
TOKEN_KEYS = ("TABPFN_TOKEN", "TABPFN_ACCESS_TOKEN", "PRIORLABS_TOKEN",
              "PRIOR_LABS_TOKEN")


def token() -> str | None:
    for key in TOKEN_KEYS:
        value = os.environ.get(key)
        if value:
            return value.strip()

    wanted = {k.upper() for k in TOKEN_KEYS}
    for path in ENV_FILES:
        if not path.exists():
            continue
        for line in path.read_text("utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            if key.strip().upper() in wanted:
                return value.strip().strip('"').strip("'")
    return None


def token_status() -> str:
    tok = token()
    if not tok:
        return "not set - using locally downloaded weights"
    return f"present ({len(tok)} chars, ...{tok[-4:]})"


def apply_token_env() -> bool:
    tok = token()
    os.environ.setdefault("TABPFN_NO_BROWSER", "1")
    if tok:
        os.environ["TABPFN_TOKEN"] = tok
        return True
    return False


def build(name: str, device: str):
    if name.startswith("TabPFN"):
        from tabpfn import TabPFNClassifier
        from tabpfn.inference_tuning import ClassifierTuningConfig
        kwargs = dict(device=device, n_estimators=N_ESTIMATORS,
                      ignore_pretraining_limits=True,
                      model_path=MODEL_VERSION,
                      random_state=cfg.RANDOM_STATE)
        if name.endswith("(thinking)"):
            kwargs["eval_metric"] = "f1"
            kwargs["tuning_config"] = ClassifierTuningConfig(
                calibrate_temperature=True, tune_decision_thresholds=True)
        return TabPFNClassifier(**kwargs)
    if name == "TabICL":
        from tabicl import TabICLClassifier
        return TabICLClassifier(device=device, n_estimators=N_ESTIMATORS,
                                allow_auto_download=True,
                                random_state=cfg.RANDOM_STATE)
    raise KeyError(name)


def get_folds(sub, y, protocol: str, design: str):
    if design == "holdout":
        return splits.make_holdout(sub, y, protocol)
    return splits.make_folds(sub, y, protocols=(protocol,))


def safe_key(task: str, protocol: str, model: str, fold_index: int) -> str:
    name = f"{task}__{protocol}__{model}__fold{fold_index}"
    for ch in " ()+->/\\":
        name = name.replace(ch, "_")
    return "_".join(part for part in name.split("_") if part)


def run_unit(task: str, protocol: str, model: str, fold_index: int,
             device: str, units_dir: str, design: str) -> dict:
    units = Path(units_dir)
    apply_token_env()
    key = safe_key(task, protocol, model, fold_index)
    t0 = time.perf_counter()
    try:
        df, columns = load_dataset()
        sub, y = target_frame(df, task)
        fold = get_folds(sub, y, protocol, design)[fold_index]
        pre, _, _ = fold_preprocessor(sub, columns, fold.train_idx)
        feats = sub[columns.predictors]
        Xtr = np.asarray(pre.transform(feats.iloc[fold.train_idx]),
                         dtype=np.float32)
        Xte = np.asarray(pre.transform(feats.iloc[fold.test_idx]),
                         dtype=np.float32)
        labels = np.asarray(sorted(pd.unique(y)))
        code = {c: i for i, c in enumerate(labels)}
        ytr = np.asarray([code[v] for v in y.to_numpy()[fold.train_idx]])
        yte = y.to_numpy()[fold.test_idx]

        clf = build(model, device)
        clf.fit(Xtr, ytr)

        unseen = splits.unseen_material_mask(sub, fold, columns.biomaterials)
        partitions = {
            "test": (Xte, yte, sub.index.to_numpy()[fold.test_idx], unseen),
            "train": (Xtr, y.to_numpy()[fold.train_idx],
                      sub.index.to_numpy()[fold.train_idx],
                      np.zeros(len(fold.train_idx), dtype=bool)),
        }
        blocks = []
        for split_name, (Xs, y_ref, row_ids, unseen_flag) in partitions.items():
            proba = np.asarray(clf.predict_proba(Xs), dtype=float)
            pred = labels[proba.argmax(axis=1)]
            block = pd.DataFrame({
                "model": model, "task": task, "protocol": protocol,
                "selection": "zero_shot", "split": split_name,
                "fold": fold.name, "row": row_ids,
                "y_true": y_ref, "y_pred": pred,
                "unseen_material": unseen_flag,
                "confidence": proba.max(axis=1),
                "seconds": time.perf_counter() - t0, "error": "",
            })
            for i, c in enumerate(labels):
                block[f"p_{c}"] = proba[:, i]
            blocks.append(block)

        units.mkdir(parents=True, exist_ok=True)
        pd.concat(blocks, ignore_index=True).to_parquet(
            units / f"{key}.parquet", index=False)
        return {"key": key, "model": model, "task": task,
                "protocol": protocol, "fold": fold.name, "device": device,
                "seconds": time.perf_counter() - t0, "error": ""}
    except Exception as exc:
        return {"key": key, "model": model, "task": task,
                "protocol": protocol, "fold": f"fold{fold_index}",
                "device": device, "seconds": time.perf_counter() - t0,
                "error": f"{type(exc).__name__}: {exc}"[:300]}


def aggregate(df) -> None:
    PREDS.mkdir(parents=True, exist_ok=True)
    files = sorted(UNITS.glob("*.parquet"))
    if not files:
        print("no completed units to aggregate")
        return
    allp = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    for (task, protocol), g in allp.groupby(["task", "protocol"]):
        out = PREDS / f"{task}__{protocol}.parquet"
        g.drop(columns=["task"]).to_parquet(out, index=False)
        print(f"  {out.name:40s} {len(g):>8,} predictions  "
              f"{g['model'].nunique()} models  {g['fold'].nunique():2d} folds")

    pooled = []
    for (model, task, protocol, split), g in allp.groupby(
            ["model", "task", "protocol", "split"]):
        labels = sorted(pd.unique(target_frame(df, task)[1]))
        pcols = [f"p_{c}" for c in labels]
        proba = g[pcols].to_numpy(float) if all(c in g for c in pcols) else None
        pooled.append({"model": model, "task": task, "protocol": protocol,
                       "selection": "zero_shot", "split": split,
                       "n_scored": len(g), "n_folds": g["fold"].nunique(),
                       **ev.scores(g["y_true"], g["y_pred"], proba, labels)})
    pooled = pd.DataFrame(pooled)
    with pd.ExcelWriter(TABLES / "foundation_models.xlsx") as xl:
        pooled.round(4).to_excel(xl, sheet_name="pooled", index=False)

    from mlate import style as ms

    test = pooled[pooled["split"] == "test"] if "split" in pooled.columns \
        else pooled
    pd.set_option("display.width", 220)
    cols = [c for c in ms.metric_columns() if c in test.columns]
    view = test[["model", "task", "protocol"] + cols].copy()
    view.columns = ["model", "task", "protocol"] + [
        ms.METRIC_LABELS[c] for c in cols]
    print("\npooled results (TEST partition)")
    print(view.round(3).to_string(index=False))

    for metric in ("weighted_f1", "macro_f1"):
        gap = test.pivot_table(index=["model", "task"], columns="protocol",
                               values=metric)
        if {"random", "doi"} <= set(gap.columns):
            gap["random_minus_doi"] = gap["random"] - gap["doi"]
            print(f"\nprotocol gap, {ms.METRIC_LABELS.get(metric, metric)} "
                  f"(test)")
            print(gap.round(3).to_string())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="*", default=list(DEFAULT_MODELS))
    ap.add_argument("--tasks", nargs="*",
                    default=["printability", "cell_response"])
    ap.add_argument("--protocols", nargs="*", default=list(PROTOCOLS))
    ap.add_argument("--workers", type=int, default=None,
                    help="concurrent inferences; default 2 per visible GPU")
    ap.add_argument("--design", default="holdout",
                    choices=["holdout", "nested"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()

    UNITS.mkdir(parents=True, exist_ok=True)
    df, columns = load_dataset()
    if args.aggregate_only:
        aggregate(df)
        return

    resources.claim()
    devices = resources.devices()
    workers = args.workers or max(1, 2 * len(devices))
    have = apply_token_env()
    print(f"TabPFN token: {token_status()}")
    if not have:
        print("  ! TabPFN 2.6 cannot download weights without a token and "
              "will fail; set TABPFN_TOKEN or put it in 05_deep_learning/.env")

    counts = {}
    for task in args.tasks:
        sub, y = target_frame(df, task)
        for protocol in args.protocols:
            folds = get_folds(sub, y, protocol, args.design)
            counts[(task, protocol)] = len(folds)
            for fold in folds:
                fold_preprocessor(sub, columns, fold.train_idx)

    units = [(t, p, m, k)
             for t in args.tasks for p in args.protocols
             for m in args.models for k in range(counts[(t, p)])]
    todo = units if args.force else [
        u for u in units if not (UNITS / f"{safe_key(*u)}.parquet").exists()]

    print(f"{len(units)} units ({len(todo)} to run) | {workers} concurrent "
          f"across {devices}")
    if args.dry_run:
        print("(dry run, nothing run)")
        return
    if not todo:
        print("nothing to do")
        aggregate(df)
        return

    t0 = time.perf_counter()
    out = Parallel(n_jobs=workers, backend="loky", verbose=0,
                   batch_size=1)(
        delayed(run_unit)(t, p, m, k, devices[i % len(devices)],
                          str(UNITS), args.design)
        for i, (t, p, m, k) in enumerate(todo))

    log = pd.DataFrame(out)
    log.to_excel(TABLES / "foundation_run_log.xlsx", index=False)
    failed = log[log["error"] != ""]
    print(f"\n{len(log) - len(failed)}/{len(log)} units succeeded in "
          f"{(time.perf_counter() - t0) / 60:.1f} min")
    for r in failed.head(10).itertuples():
        print(f"    {r.key}: {r.error}")

    print("\naggregating")
    aggregate(df)
    print(f"\ntables -> {TABLES}")


if __name__ == "__main__":
    main()
