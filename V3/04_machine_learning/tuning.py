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
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from mlate import config as cfg
from mlate import models as zoo
from mlate import resources, scheduling, splits
from mlate import search_spaces as ss
from mlate.artifacts import fold_preprocessor
from mlate.dataset import load_dataset, target_frame

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

TABLES = cfg.step_dir("04_machine_learning", "tables")
TUNING = TABLES / "tuning"
PREDS = TABLES / "predictions_tuned"

UNITS = TUNING / "units"
PARAMS = TUNING / "params"


def configure(args) -> str:
    global UNITS, PARAMS
    payload = {
        "trials": args.trials, "inner_folds": args.inner_folds,
        "objective": args.objective, "pruner": args.pruner,
        "design": args.design,
        "outer_folds": cfg.N_FOLDS, "seed": cfg.RANDOM_STATE,
        "sampler": "tpe_multivariate",
    }
    fp = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()).hexdigest()[:10]
    UNITS = TUNING / fp / "units"
    PARAMS = TUNING / fp / "params"
    UNITS.mkdir(parents=True, exist_ok=True)
    PARAMS.mkdir(parents=True, exist_ok=True)
    (TUNING / fp / "run_config.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    return fp

N_TRIALS = 50
INNER_FOLDS = 10

WORKER_FRACTION = 0.85


def worker_count() -> int:
    return max(1, int(WORKER_FRACTION * cfg.CPU_TOTAL))


def make_sampler(seed: int):
    return optuna.samplers.TPESampler(seed=seed, multivariate=True, group=True)


def make_pruner(kind: str):
    warmup = dict(n_startup_trials=5, n_warmup_steps=3)
    if kind == "median":
        return optuna.pruners.MedianPruner(**warmup)
    if kind == "percentile75":
        return optuna.pruners.PercentilePruner(75.0, **warmup)
    if kind == "successive":
        return optuna.pruners.SuccessiveHalvingPruner()
    if kind == "hyperband":
        return optuna.pruners.HyperbandPruner(min_resource=2)
    if kind == "wilcoxon":
        return optuna.pruners.WilcoxonPruner(p_threshold=0.1,
                                             n_startup_steps=2)
    raise ValueError(f"unknown pruner: {kind}")


PRUNERS = ("median", "percentile75", "successive", "hyperband", "wilcoxon")


def get_folds(sub, y, protocol: str, design: str):
    if design == "holdout":
        return splits.make_holdout(sub, y, protocol)
    return splits.make_folds(sub, y, protocols=(protocol,))


def inner_splits(protocol: str, ytr: np.ndarray, groups_tr: np.ndarray,
                 inner_folds: int):
    if protocol in ("doi", "tissue"):
        n = min(inner_folds, len(np.unique(groups_tr)))
        return list(StratifiedGroupKFold(n_splits=max(2, n)).split(
            np.zeros(len(ytr)), ytr, groups=groups_tr))
    return list(StratifiedKFold(n_splits=inner_folds, shuffle=True,
                                random_state=cfg.RANDOM_STATE)
                .split(np.zeros(len(ytr)), ytr))


SELECTION_METRICS = ("macro", "weighted", "quadratic")


def _score(y_true, y_pred, metric: str) -> float:
    if metric == "quadratic":
        try:
            return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
        except Exception:
            return 0.0
    return float(f1_score(y_true, y_pred, average=metric, zero_division=0))


def tune(model: str, Xtr, ytr, groups_tr, protocol: str, n_trials: int,
         inner_folds: int, objective: str, n_classes: int, pruner: str):
    folds = inner_splits(protocol, ytr, groups_tr, inner_folds)
    performed = 0
    others = [m for m in SELECTION_METRICS if m != objective]

    report_running_mean = pruner != "wilcoxon"

    def objective_fn(trial):
        nonlocal performed
        params = ss.suggest(model, trial)
        primary = []
        secondary = {m: [] for m in others}
        for a, b in folds:
            try:
                est = ss.build_tuned(model, params, n_jobs=1,
                                     n_classes=n_classes)
                est.fit(Xtr[a], ytr[a])
                pred = est.predict(Xtr[b])
                primary.append(_score(ytr[b], pred, objective))
                for m in others:
                    secondary[m].append(_score(ytr[b], pred, m))
            except Exception:
                primary.append(0.0)
                for m in others:
                    secondary[m].append(0.0)
            performed += 1
            trial.report(
                float(np.mean(primary)) if report_running_mean else primary[-1],
                len(primary))
            if trial.should_prune():
                raise optuna.TrialPruned()
        for m in others:
            trial.set_user_attr(f"sel_{m}", float(np.mean(secondary[m])))
        return float(np.mean(primary))

    study = optuna.create_study(
        direction="maximize",
        sampler=make_sampler(cfg.RANDOM_STATE),
        pruner=make_pruner(pruner))
    study.optimize(objective_fn, n_trials=n_trials, n_jobs=1,
                   show_progress_bar=False)

    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError(f"no trial completed for {model}")

    winners = {objective: (study.best_params, float(study.best_value))}
    for m in others:
        scored = [(t.user_attrs.get(f"sel_{m}"), t) for t in completed]
        scored = [(v, t) for v, t in scored if v is not None]
        if scored:
            best_v, best_t = max(scored, key=lambda vt: vt[0])
            winners[m] = (best_t.params, float(best_v))

    pruned = [t for t in study.trials
              if t.state == optuna.trial.TrialState.PRUNED]
    unpruned = n_trials * len(folds)
    stats = {
        "n_trials_complete": len(completed),
        "n_trials_pruned": len(pruned),
        "inner_fits": performed,
        "inner_fits_unpruned": unpruned,
        "pruning_saving": (1.0 - performed / unpruned) if unpruned else 0.0,
        "n_distinct_winners": len({json.dumps(w[0], sort_keys=True,
                                               default=str)
                                   for w in winners.values()}),
    }
    return winners, stats


def run_unit(task: str, protocol: str, model: str, fold_index: int,
             n_trials: int, inner_folds: int, objective: str, pruner: str,
             wave: int, units_dir: str, params_dir: str,
             design: str) -> dict:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    optuna.logging.disable_default_handler()

    _threads = resources.single_thread()

    units = Path(units_dir)
    params_out = Path(params_dir)

    unit = scheduling.Unit(task=task, protocol=protocol, model=model,
                           fold_index=fold_index)
    t0 = time.perf_counter()
    try:
        df, columns = load_dataset()
        sub, y = target_frame(df, task)
        fold = get_folds(sub, y, protocol, design)[fold_index]

        pre, _, _ = fold_preprocessor(sub, columns, fold.train_idx)
        feats = sub[columns.predictors]
        Xtr = np.asarray(pre.transform(feats.iloc[fold.train_idx]), float)
        Xte = np.asarray(pre.transform(feats.iloc[fold.test_idx]), float)

        labels = np.asarray(sorted(pd.unique(y)))
        code = {c: i for i, c in enumerate(labels)}
        ytr = np.asarray([code[v] for v in y.to_numpy()[fold.train_idx]])
        yte = y.to_numpy()[fold.test_idx]
        doi = sub["DOI"].astype(str).to_numpy()[fold.train_idx]

        stats = {}
        if model in ss.BASELINES:
            winners = {"untuned": ({}, float("nan"))}
        elif model in ss.META:
            tuned = _load_base_params(task, protocol, fold_index, params_out)
            winners = {"composed": ({"composed_from": sorted(tuned)},
                                    float("nan"))}
        else:
            winners, stats = tune(model, Xtr, ytr, doi, protocol, n_trials,
                                  inner_folds, objective, len(labels), pruner)

        unseen = splits.unseen_material_mask(sub, fold, columns.biomaterials)
        blocks, recorded = [], {}
        for selection, (params, inner_score) in winners.items():
            if model in ss.BASELINES:
                est = zoo.build(model, n_jobs=1)
            elif model in ss.META:
                est = ss.build_meta(
                    model, _load_base_params(task, protocol, fold_index,
                                             params_out), n_jobs=1)
            else:
                est = ss.build_tuned(model, params, n_jobs=1,
                                     n_classes=len(labels))
            est.fit(Xtr, ytr)

            partitions = {
                "test": (fold.test_idx, yte, unseen),
                "train": (fold.train_idx, y.to_numpy()[fold.train_idx],
                          np.zeros(len(fold.train_idx), dtype=bool)),
            }
            for split_name, (row_idx, y_ref, unseen_flag) in partitions.items():
                Xs = Xte if split_name == "test" else Xtr
                pred = labels[np.asarray(est.predict(Xs)).ravel().astype(int)]

                proba, seen = None, None
                if hasattr(est, "predict_proba"):
                    try:
                        proba = np.asarray(est.predict_proba(Xs), dtype=float)
                        seen = labels[np.asarray(est.classes_).astype(int)]
                    except Exception:
                        proba, seen = None, None

                block = pd.DataFrame({
                    "model": model, "task": task, "protocol": protocol,
                    "selection": selection, "split": split_name,
                    "fold": fold.name,
                    "row": sub.index.to_numpy()[row_idx],
                    "y_true": y_ref, "y_pred": pred,
                    "unseen_material": unseen_flag,
                    "seconds": time.perf_counter() - t0, "error": "",
                })
                for c in labels:
                    block[f"p_{c}"] = np.nan
                if proba is not None and seen is not None:
                    for j, c in enumerate(seen):
                        block[f"p_{c}"] = proba[:, j]
                    block["confidence"] = proba.max(axis=1)
                else:
                    block["confidence"] = np.nan
                blocks.append(block)
            recorded[selection] = {
                "params": params,
                "inner_score": None if np.isnan(inner_score) else inner_score,
            }

        units.mkdir(parents=True, exist_ok=True)
        params_out.mkdir(parents=True, exist_ok=True)
        pd.concat(blocks, ignore_index=True).to_parquet(
            units / f"{unit.safe_key}.parquet", index=False)
        (params_out / f"{unit.safe_key}.json").write_text(json.dumps({
            "task": task, "protocol": protocol, "model": model,
            "fold": fold.name, "fold_index": fold_index, "wave": wave,
            "objective": objective, "pruner": pruner, "n_trials": n_trials,
            "inner_folds": inner_folds,
            "seconds": time.perf_counter() - t0,
            "stats": stats,
            "selections": recorded,
        }, indent=2, default=str), encoding="utf-8")

        return {"key": unit.key, "model": model, "task": task,
                "protocol": protocol, "fold": fold.name,
                "seconds": time.perf_counter() - t0,
                "selections": "|".join(recorded),
                "inner_score": next(
                    (v["inner_score"] for v in recorded.values()
                     if v["inner_score"] is not None), float("nan")),
                "error": "", **stats}

    except Exception as exc:
        return {"key": unit.key, "model": model, "task": task,
                "protocol": protocol, "fold": f"fold{fold_index}",
                "seconds": time.perf_counter() - t0,
                "inner_score": float("nan"),
                "error": f"{type(exc).__name__}: {exc}"[:300]}


def _load_base_params(task: str, protocol: str, fold_index: int,
                      params_dir) -> dict:
    out = {}
    for base in ("Random Forest", "XGBoost", "Logistic Regression"):
        unit = scheduling.Unit(task=task, protocol=protocol, model=base,
                               fold_index=fold_index)
        path = Path(params_dir) / f"{unit.safe_key}.json"
        if path.exists():
            try:
                rec = json.loads(path.read_text("utf-8"))
                sel = rec.get("selections") or {}
                pick = sel.get("macro") or next(iter(sel.values()), {})
                out[base] = pick.get("params", {})
            except Exception:
                pass
    return out


def prewarm(tasks, protocols, design: str) -> dict:
    df, columns = load_dataset()
    counts = {}
    for task in tasks:
        sub, y = target_frame(df, task)
        for protocol in protocols:
            folds = get_folds(sub, y, protocol, design)
            counts[(task, protocol)] = len(folds)
            hits = 0
            for fold in folds:
                _, _, cached = fold_preprocessor(sub, columns, fold.train_idx)
                hits += bool(cached)
            print(f"  {task:16s} {protocol:7s} {len(folds):2d} folds  "
                  f"({hits} preprocessors cached, {len(folds) - hits} fitted)")
    return counts


def done(unit: scheduling.Unit) -> bool:
    return (UNITS / f"{unit.safe_key}.parquet").exists()


def _run_configs() -> dict:
    out = {}
    for cfg_path in sorted(TUNING.glob("*/run_config.json")):
        try:
            out[cfg_path.parent.name] = json.loads(
                cfg_path.read_text("utf-8"))
        except Exception:
            out[cfg_path.parent.name] = {}
    return out


def aggregate() -> None:
    PREDS.mkdir(parents=True, exist_ok=True)
    configs = _run_configs()
    if not configs:
        configs = {"": {}}

    def budget(fp: str) -> int:
        return int(configs.get(fp, {}).get("trials", 0) or 0)

    frames, param_files = [], []
    for fp in sorted(configs, key=budget):
        udir, pdir = TUNING / fp / "units", TUNING / fp / "params"
        for f in sorted(udir.glob("*.parquet")):
            d = pd.read_parquet(f)
            d["run_config"] = fp
            frames.append(d)
        param_files += [(fp, p) for p in sorted(pdir.glob("*.json"))]

    if not frames:
        print("no completed units to aggregate")
        return
    allp = pd.concat(frames, ignore_index=True)

    key = [c for c in ("task", "protocol", "model", "fold", "selection",
                       "split", "row") if c in allp.columns]
    before = len(allp)
    allp = allp.drop_duplicates(subset=key, keep="last")
    if len(allp) != before:
        print(f"  ! {before - len(allp):,} predictions superseded by a "
              f"larger-budget configuration")

    for (task, protocol), g in allp.groupby(["task", "protocol"]):
        out = PREDS / f"{task}__{protocol}.parquet"
        g.drop(columns=["task"]).to_parquet(out, index=False)
        print(f"  {out.name:44s} {len(g):>8,} predictions  "
              f"{g['model'].nunique():2d} models  "
              f"{g['fold'].nunique():2d} folds")

    rows = []
    for fp, path in param_files:
        try:
            rec = json.loads(path.read_text("utf-8"))
        except Exception:
            continue
        base = {k: rec.get(k) for k in
                ("task", "protocol", "model", "fold", "objective", "pruner",
                 "seconds")}
        stats = rec.get("stats") or {}
        for selection, payload in (rec.get("selections") or {}).items():
            rows.append(base | {
                "selection": selection,
                "inner_score": payload.get("inner_score"),
                "pruning_saving": stats.get("pruning_saving"),
                "n_trials_pruned": stats.get("n_trials_pruned"),
                "params": json.dumps(payload.get("params", {}), default=str),
                "run_config": fp,
                "n_trials_budget": budget(fp),
            })
    if rows:
        d = pd.DataFrame(rows).sort_values("n_trials_budget")
        d = d.drop_duplicates(subset=["task", "protocol", "model", "fold",
                                      "selection"], keep="last")
        d.to_excel(TABLES / "tuned_best_params.xlsx", index=False)
        counts = d.groupby("protocol").size().to_dict()
        print(f"  {'tuned_best_params.xlsx':44s} {len(d):>8,} rows  "
              f"{counts}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", nargs="*",
                    default=["printability", "cell_response"])
    ap.add_argument("--protocols", nargs="*", default=["random", "doi"])
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--trials", type=int, default=N_TRIALS)
    ap.add_argument("--inner-folds", type=int, default=INNER_FOLDS)
    ap.add_argument("--objective", default="macro",
                    choices=list(SELECTION_METRICS),
                    help="metric the search maximises; the other two are "
                         "still selected on from the same trials")
    ap.add_argument("--design", default="holdout",
                    choices=["holdout", "nested"],
                    help="holdout: one 80:20 split, 10-fold CV inside the "
                         "training partition (the submitted design). "
                         "nested: 10 outer folds each with its own search.")
    ap.add_argument("--pruner", default="wilcoxon", choices=PRUNERS,
                    help="pruning strategy; the main lever on total work")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--force", action="store_true",
                    help="re-run units that already have a checkpoint")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the schedule and cost estimate, fit nothing")
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()

    PREDS.mkdir(parents=True, exist_ok=True)
    fingerprint = configure(args)

    if args.aggregate_only:
        aggregate()
        return

    budget = resources.claim()
    workers = args.workers or worker_count()
    models = args.models or zoo.names()
    print(f"\nrun configuration {fingerprint}: {args.design} design | "
          f"{args.trials} trials x {args.inner_folds} inner folds | "
          f"objective {args.objective} | {args.pruner} pruner")
    print(f"checkpoints -> {UNITS.parent}")

    print(f"\npre-warming fold preprocessors")
    counts = prewarm(args.tasks, args.protocols, args.design)

    costs = scheduling.load_fit_costs()
    if not costs:
        print("\n  ! no fit_costs.xlsx - scheduling on the registry's coarse "
              "cost tags.\n    Run 04_machine_learning/calibrate.py first for "
              "a materially better schedule.")

    wave1 = [m for m in models if m not in ss.META]
    wave2 = [m for m in models if m in ss.META]

    units1 = scheduling.build_units(args.tasks, args.protocols, wave1, counts,
                                    args.trials, args.inner_folds, costs)
    units2 = scheduling.build_units(args.tasks, args.protocols, wave2, counts,
                                    args.trials, args.inner_folds, costs)

    plan = scheduling.makespan(units1 + units2, workers)
    print(f"\n{plan['n_units']} units | {workers} workers x 1 thread "
          f"= {workers}/{cfg.CPU_TOTAL} cores "
          f"({workers / cfg.CPU_TOTAL:.0%})")
    print(f"estimated {plan['core_hours']:.0f} core-hours -> "
          f"{plan['makespan_hours']:.1f} h wall-clock "
          f"at {plan['utilisation']:.0%} utilisation")

    table = scheduling.plan_table(units1 + units2)
    pd.set_option("display.width", 200)
    print("\ncost by model (estimated)")
    print(table.head(12).round(3).to_string(index=False))

    if args.dry_run:
        table.to_excel(TABLES / "tuning_plan.xlsx", index=False)
        print(f"\n-> {TABLES / 'tuning_plan.xlsx'}  (dry run, nothing fitted)")
        return

    results = []
    for wave, units in ((1, units1), (2, units2)):
        todo = units if args.force else [u for u in units if not done(u)]
        if not todo:
            print(f"\nwave {wave}: nothing to do "
                  f"({len(units)} units already complete)")
            continue
        skipped = len(units) - len(todo)
        print(f"\nwave {wave}: {len(todo)} units"
              + (f" ({skipped} already complete, skipped)" if skipped else ""))
        t0 = time.perf_counter()
        out = Parallel(n_jobs=workers, backend="loky", verbose=0,
                       batch_size=1)(
            delayed(run_unit)(u.task, u.protocol, u.model, u.fold_index,
                              args.trials, args.inner_folds, args.objective,
                              args.pruner, wave, str(UNITS), str(PARAMS),
                              args.design)
            for u in todo)
        results += out
        elapsed = time.perf_counter() - t0
        failed = [r for r in out if r["error"]]
        print(f"  wave {wave} done in {elapsed / 3600:.2f} h"
              + (f"  [{len(failed)} failed]" if failed else ""))
        for r in failed[:10]:
            print(f"    {r['key']}: {r['error']}")

    if results:
        log = pd.DataFrame(results)
        log.to_excel(TABLES / "tuning_run_log.xlsx", index=False)
        ok = log[log["error"] == ""]
        print(f"\n{len(ok)}/{len(log)} units succeeded, "
              f"{ok['seconds'].sum() / 3600:.1f} core-hours actual "
              f"(estimated {plan['core_hours']:.0f})")
        if "pruning_saving" in ok and ok["pruning_saving"].notna().any():
            saved = ok["pruning_saving"].dropna()
            fits = ok["inner_fits"].dropna().sum()
            full = ok["inner_fits_unpruned"].dropna().sum()
            print(f"pruning ({args.pruner}): {fits:,.0f} of {full:,.0f} inner "
                  f"fits performed, {1 - fits / full:.0%} avoided "
                  f"(per-unit mean {saved.mean():.0%})")

    print("\naggregating")
    aggregate()
    print(f"\ntables -> {TABLES}")


if __name__ == "__main__":
    main()
