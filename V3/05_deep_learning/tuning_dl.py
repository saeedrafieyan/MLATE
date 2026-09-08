from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.model_selection import (GroupShuffleSplit,
                                     StratifiedShuffleSplit)

from mlate import config as cfg
from mlate import deep
from mlate import resources, splits
from mlate.artifacts import fold_preprocessor
from mlate.dataset import load_dataset, target_frame

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

TABLES = cfg.step_dir("05_deep_learning", "tables")
TUNING = TABLES / "tuning"
PREDS = TABLES / "predictions_dl"
MODELS = cfg.step_dir("05_deep_learning", "models")

N_TRIALS = 30
VAL_FRACTION = 0.15
PATIENCE = 25

SELECTION_METRICS = ("macro", "weighted", "quadratic")

UNITS = TUNING / "units"
PARAMS = TUNING / "params"


def configure(args) -> str:
    global UNITS, PARAMS
    payload = {"trials": args.trials, "val_fraction": args.val_fraction,
               "patience": args.patience, "objective": args.objective,
               "design": args.design,
               "outer_folds": cfg.N_FOLDS, "seed": cfg.RANDOM_STATE}
    fp = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()).hexdigest()[:10]
    UNITS = TUNING / fp / "units"
    PARAMS = TUNING / fp / "params"
    UNITS.mkdir(parents=True, exist_ok=True)
    PARAMS.mkdir(parents=True, exist_ok=True)
    (TUNING / fp / "run_config.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    return fp


def get_folds(sub, y, protocol: str, design: str):
    if design == "holdout":
        return splits.make_holdout(sub, y, protocol)
    return splits.make_folds(sub, y, protocols=(protocol,))


def _score(y_true, y_pred, metric: str) -> float:
    if metric == "quadratic":
        try:
            return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
        except Exception:
            return 0.0
    return float(f1_score(y_true, y_pred, average=metric, zero_division=0))


def inner_holdout(protocol: str, y: np.ndarray, groups: np.ndarray,
                  val_fraction: float):
    idx = np.arange(len(y))
    if protocol in ("doi", "tissue"):
        gss = GroupShuffleSplit(n_splits=1, test_size=val_fraction,
                                random_state=cfg.RANDOM_STATE)
        tr, va = next(gss.split(idx, y, groups=groups))
    else:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction,
                                     random_state=cfg.RANDOM_STATE)
        tr, va = next(sss.split(idx, y))
    if len(np.unique(y[va])) < len(np.unique(y)):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction,
                                     random_state=cfg.RANDOM_STATE)
        tr, va = next(sss.split(idx, y))
    return tr, va


def tune(arch: str, Xtr, ytr, groups, protocol: str, n_trials: int,
         val_fraction: float, patience: int, objective: str, n_classes: int,
         device: str):
    tr, va = inner_holdout(protocol, ytr, groups, val_fraction)
    Xa, ya, Xb, yb = Xtr[tr], ytr[tr], Xtr[va], ytr[va]
    others = [m for m in SELECTION_METRICS if m != objective]
    epochs_run = 0

    def objective_fn(trial):
        nonlocal epochs_run
        params = deep.suggest(trial)

        def report(epoch, score):
            trial.report(score, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        model, info = deep.train(arch, params, Xa, ya, Xb, yb, n_classes,
                                 device, patience=patience, report=report)
        epochs_run += info.get("epochs_run", 0)
        if info.get("diverged"):
            return 0.0
        proba = deep.predict_proba(model, Xb, device)
        pred = proba.argmax(axis=1)
        for m in others:
            trial.set_user_attr(f"sel_{m}", _score(yb, pred, m))
        trial.set_user_attr("best_epoch", int(info.get("best_epoch", 0)))
        return _score(yb, pred, objective)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=cfg.RANDOM_STATE,
                                           multivariate=True, group=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5,
                                           n_warmup_steps=15))
    study.optimize(objective_fn, n_trials=n_trials, n_jobs=1,
                   show_progress_bar=False)

    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError(f"no trial completed for {arch}")

    winners = {objective: (study.best_params, float(study.best_value))}
    for m in others:
        scored = [(t.user_attrs.get(f"sel_{m}"), t) for t in completed]
        scored = [(v, t) for v, t in scored if v is not None]
        if scored:
            best_v, best_t = max(scored, key=lambda vt: vt[0])
            winners[m] = (best_t.params, float(best_v))

    pruned = [t for t in study.trials
              if t.state == optuna.trial.TrialState.PRUNED]
    stats = {"n_trials_complete": len(completed),
             "n_trials_pruned": len(pruned),
             "epochs_run": epochs_run,
             "n_val": int(len(va)), "n_inner_train": int(len(tr))}
    return winners, stats


def run_unit(task: str, protocol: str, arch: str, fold_index: int,
             n_trials: int, val_fraction: float, patience: int,
             objective: str, device: str, units_dir: str,
             params_dir: str, design: str) -> dict:
    units = Path(units_dir)
    params_out = Path(params_dir)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    optuna.logging.disable_default_handler()

    key = f"{task}__{protocol}__{arch}__fold{fold_index}"
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
        doi = sub["DOI"].astype(str).to_numpy()[fold.train_idx]

        winners, stats = tune(arch, Xtr, ytr, doi, protocol, n_trials,
                              val_fraction, patience, objective, len(labels),
                              device)

        tr, va = inner_holdout(protocol, ytr, doi, val_fraction)
        unseen = splits.unseen_material_mask(sub, fold, columns.biomaterials)
        blocks, recorded, histories = [], {}, []
        for selection, (params, inner_score) in winners.items():
            model, info = deep.train(arch, params, Xtr[tr], ytr[tr],
                                     Xtr[va], ytr[va], len(labels), device,
                                     patience=patience)
            partitions = {
                "test": (Xte, yte, sub.index.to_numpy()[fold.test_idx],
                         unseen),
                "train": (Xtr, y.to_numpy()[fold.train_idx],
                          sub.index.to_numpy()[fold.train_idx],
                          np.zeros(len(fold.train_idx), dtype=bool)),
            }
            for split_name, (Xs, y_ref, row_ids, unseen_flag) in                     partitions.items():
                proba = deep.predict_proba(model, Xs, device)
                pred = labels[proba.argmax(axis=1)]
                block = pd.DataFrame({
                    "model": arch, "task": task, "protocol": protocol,
                    "selection": selection, "split": split_name,
                    "fold": fold.name, "row": row_ids,
                    "y_true": y_ref, "y_pred": pred,
                    "unseen_material": unseen_flag,
                    "confidence": proba.max(axis=1),
                    "seconds": time.perf_counter() - t0, "error": "",
                })
                for i, c in enumerate(labels):
                    block[f"p_{c}"] = proba[:, i]
                blocks.append(block)

            hist = pd.DataFrame(info.get("history", []))
            if len(hist):
                hist.insert(0, "selection", selection)
                hist.insert(0, "fold", fold.name)
                hist.insert(0, "protocol", protocol)
                hist.insert(0, "task", task)
                hist.insert(0, "model", arch)
                histories.append(hist)

            recorded[selection] = {
                "params": params, "inner_score": inner_score,
                "best_epoch": info.get("best_epoch"),
                "epochs_run": info.get("epochs_run"),
                "best_val_macro_f1": info.get("best_val_macro_f1"),
            }

        units.mkdir(parents=True, exist_ok=True)
        params_out.mkdir(parents=True, exist_ok=True)
        pd.concat(blocks, ignore_index=True).to_parquet(
            units / f"{key}.parquet", index=False)
        if histories:
            hist_dir = units.parent / "history"
            hist_dir.mkdir(parents=True, exist_ok=True)
            pd.concat(histories, ignore_index=True).to_parquet(
                hist_dir / f"{key}.parquet", index=False)
        (params_out / f"{key}.json").write_text(json.dumps({
            "task": task, "protocol": protocol, "model": arch,
            "fold": fold.name, "fold_index": fold_index, "device": device,
            "objective": objective, "n_trials": n_trials,
            "seconds": time.perf_counter() - t0,
            "stats": stats, "selections": recorded,
        }, indent=2, default=str), encoding="utf-8")

        return {"key": key, "model": arch, "task": task, "protocol": protocol,
                "fold": fold.name, "device": device,
                "seconds": time.perf_counter() - t0, "error": "", **stats}

    except Exception as exc:
        return {"key": key, "model": arch, "task": task, "protocol": protocol,
                "fold": f"fold{fold_index}", "device": device,
                "seconds": time.perf_counter() - t0,
                "error": f"{type(exc).__name__}: {exc}"[:300]}


def done(key: str) -> bool:
    return (UNITS / f"{key}.parquet").exists()


TASK_LABELS = {"printability": [0, 1, 2, 3],
               "cell_response": [1, 2, 3, 4, 5],
               "cell_response_cellular": [2, 3, 4, 5]}


def score(allp: pd.DataFrame) -> pd.DataFrame:
    from mlate import evaluation as ev

    rows = []
    keys = ["model", "task", "protocol", "selection", "split"]
    for (model, task, protocol, selection, split), g in allp.groupby(
            keys, sort=False):
        labels = TASK_LABELS[task]
        pcols = [f"p_{c}" for c in labels]
        proba = None
        if all(c in g.columns for c in pcols):
            arr = g[pcols].to_numpy(dtype=float)
            proba = None if np.isnan(arr).any() else arr
        by_fold = pd.DataFrame([
            ev.scores(h["y_true"], h["y_pred"], None, labels)
            for _, h in g.groupby("fold")])
        rows.append({
            "model": model, "task": task, "protocol": protocol,
            "selection": selection, "split": split, "n_scored": int(len(g)),
            "n_folds": int(g["fold"].nunique()),
            **ev.scores(g["y_true"], g["y_pred"], proba, labels),
            "macro_f1_sd": float(by_fold["macro_f1"].std(ddof=1)),
            "weighted_f1_sd": float(by_fold["weighted_f1"].std(ddof=1)),
        })
    return pd.DataFrame(rows)


def aggregate() -> None:
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
              f"{g['model'].nunique()} architectures  "
              f"{g['fold'].nunique():2d} folds")

    board = score(allp)
    board.round(4).to_excel(TABLES / "dl_benchmark.xlsx", index=False)
    print(f"  {'dl_benchmark.xlsx':40s} {len(board):>8,} rows")
    pd.set_option("display.width", 220)
    from mlate import style as ms

    lead = "weighted" if "weighted" in set(board["selection"]) else \
        board["selection"].iloc[0]
    show = board[(board["split"] == "test") & (board["selection"] == lead)]
    cols = [c for c in ms.metric_columns() if c in show.columns]
    view = show[["model", "task", "protocol"] + cols].copy()
    view.columns = ["model", "task", "protocol"] + [
        ms.METRIC_LABELS[c] for c in cols]
    print(f"\ntest partition, '{lead}' selection")
    print(view.round(3).to_string(index=False))

    rows = []
    for path in sorted(PARAMS.glob("*.json")):
        try:
            rec = json.loads(path.read_text("utf-8"))
        except Exception:
            continue
        base = {k: rec.get(k) for k in ("task", "protocol", "model", "fold",
                                        "objective", "device", "seconds")}
        for selection, payload in (rec.get("selections") or {}).items():
            rows.append(base | {
                "selection": selection,
                "inner_score": payload.get("inner_score"),
                "best_epoch": payload.get("best_epoch"),
                "params": json.dumps(payload.get("params", {}), default=str)})
    if rows:
        pd.DataFrame(rows).to_excel(TABLES / "dl_best_params.xlsx",
                                    index=False)
        print(f"  {'dl_best_params.xlsx':40s} {len(rows):>8,} rows")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="*", default=deep.names())
    ap.add_argument("--tasks", nargs="*",
                    default=["printability", "cell_response"])
    ap.add_argument("--protocols", nargs="*", default=["random", "doi"])
    ap.add_argument("--trials", type=int, default=N_TRIALS)
    ap.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    ap.add_argument("--patience", type=int, default=PATIENCE)
    ap.add_argument("--objective", default="weighted",
                    choices=list(SELECTION_METRICS))
    ap.add_argument("--design", default="holdout",
                    choices=["holdout", "nested"],
                    help="must match step 04, or the two are scored on "
                         "different test rows")
    ap.add_argument("--workers", type=int, default=None,
                    help="concurrent trainings; default 4 per visible GPU")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()

    PREDS.mkdir(parents=True, exist_ok=True)
    fingerprint = configure(args)
    if args.aggregate_only:
        aggregate()
        return

    resources.claim()
    devices = resources.devices()
    workers = args.workers or max(1, 4 * len(devices))

    df, columns = load_dataset()
    counts, todo = {}, []
    for task in args.tasks:
        sub, y = target_frame(df, task)
        for protocol in args.protocols:
            folds = get_folds(sub, y, protocol, args.design)
            counts[(task, protocol)] = len(folds)
            for fold in folds:
                _, _, _ = fold_preprocessor(sub, columns, fold.train_idx)

    units = [(t, p, m, k)
             for t in args.tasks for p in args.protocols
             for m in args.models
             for k in range(counts[(t, p)])]
    todo = units if args.force else [
        u for u in units
        if not done(f"{u[0]}__{u[1]}__{u[2]}__fold{u[3]}")]

    print(f"\nrun configuration {fingerprint}: {args.trials} trials | "
          f"objective {args.objective} | {cfg.N_FOLDS} outer folds")
    print(f"checkpoints -> {UNITS.parent}")
    print(f"{len(units)} units ({len(todo)} to run) | {workers} concurrent "
          f"trainings across {devices}")

    if args.dry_run:
        print("\n(dry run, nothing trained)")
        return
    if not todo:
        print("nothing to do")
        aggregate()
        return

    t0 = time.perf_counter()
    out = Parallel(n_jobs=workers, backend="loky", verbose=0,
                   batch_size=1)(
        delayed(run_unit)(t, p, m, k, args.trials, args.val_fraction,
                          args.patience, args.objective,
                          devices[i % len(devices)], str(UNITS), str(PARAMS),
                          args.design)
        for i, (t, p, m, k) in enumerate(todo))

    log = pd.DataFrame(out)
    log.to_excel(TABLES / "dl_run_log.xlsx", index=False)
    failed = log[log["error"] != ""]
    print(f"\n{len(log) - len(failed)}/{len(log)} units succeeded in "
          f"{(time.perf_counter() - t0) / 3600:.2f} h")
    for r in failed.head(10).itertuples():
        print(f"    {r.key}: {r.error}")

    print("\naggregating")
    aggregate()
    print(f"\ntables -> {TABLES}")


if __name__ == "__main__":
    main()
