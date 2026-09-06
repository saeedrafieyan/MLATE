"""
Refit and ship every model, under both splitting protocols
==========================================================

    python 06_webapp/export_deployment.py --dry-run     # matrix only, no fitting
    python 06_webapp/export_deployment.py
    python 06_webapp/export_deployment.py --families ml --protocols doi

Takes the tuned hyper-parameters from steps 04 and 05, refits each model on the
whole dataset behind the release preprocessor, and writes a deployment bundle
under deploy/models/ for upload to Hugging Face.

What is shipped, and why all of it
----------------------------------
Every model, not a top-3 shortlist. The application lets a user choose the
predictor per target, so a shortlist would decide for them; and the argument of
the paper is that the right model depends on which regime the user is in.
Shipping the full set makes that choice real rather than rhetorical.

Both protocols, for the same reason
-----------------------------------
A model appears twice per target: once carrying the hyper-parameters chosen
under the random split, once carrying those chosen under study-grouped
splitting. They are genuinely different models, because the two protocols
select different configurations, and they answer different questions.

  random   interpolation inside the design space the dataset covers - adjusting
           a concentration, swapping a cell line, moving a pressure within
           observed ranges. This is what a user of the tool is usually doing.
  doi      extrapolation to a study the model has never seen, which is the
           honest bound for a genuinely new laboratory.

Shipping only the random-split models would leave the conservative numbers in
the manuscript with no artefact behind them, and shipping only the grouped ones
would hand users a model tuned for a harder task than the one they face. The
application selects on protocol, so the choice stays with the person who knows
which regime they are in.

Which configuration
-------------------
Under the 80:20 hold-out design there is one tuned configuration per
(task, protocol, model): Optuna searched inside the training partition, scored
by 10-fold cross-validation on that partition, and `inner_score` is that
cross-validated mean. No test row informed any hyper-parameter. Where several
rows exist for a group the highest inner score wins, which is a no-op for the
current single-hold-out tables but keeps the rule correct if outer folds ever
come back.

Fitted on every row
-------------------
Each shipped model is refitted on the complete dataset behind the release
preprocessor, which is also fitted on every row. At inference time there is no
held-out set to protect, and a formulation submitted by a user deserves
everything the dataset knows. The numbers in the paper come from the hold-out
partitions and are reproducible from the stored per-row predictions under
results/04_machine_learning/tables/predictions_tuned and its step-05
counterparts; these artefacts are for serving, not for re-deriving the tables.

Foundation models
-----------------
TabPFN and TabICL have no trained weights of ours to save - they are in-context
learners, and the fitted object IS the training set. What ships is the
transformed context matrix, its labels and the checkpoint identifier, so the
application can instantiate the vendor classifier and fit it in one call. Being
hyper-parameter-free, they carry no protocol distinction.

Every artefact records its hyper-parameters, the protocol they were chosen
under, that configuration's inner score and the model's hold-out metrics, so a
number in the paper can always be traced to the file that produces it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd

from mlate import artifacts
from mlate import config as cfg
from mlate import models as zoo
from mlate import search_spaces as ss
from mlate.dataset import load_dataset, target_frame

ML_TABLES = cfg.step_dir("04_machine_learning", "tables")
DL_TABLES = cfg.step_dir("05_deep_learning", "tables")
OUT = cfg.MODEL_DIR
COMPRESS = 3

TASKS = ("printability", "cell_response")
PROTOCOLS = ("random", "doi")


def slugify(name: str) -> str:
    s = name.lower()
    for ch in " ()+->/\\,":
        s = s.replace(ch, "_")
    return "_".join(p for p in s.split("_") if p)


def best_params(table: Path, protocol: str, selection: str) -> pd.DataFrame:
    """
    One row per (task, model) for this protocol: the highest inner score.

    The meta-ensembles come along under selection "composed". They have no
    search space of their own - their configuration is the set of base
    estimators - so they carry no inner score and cannot be filtered on one.
    Excluding them would drop Stacking, which is the leading Cell Response
    model, from the bundle. The dummy baselines are left out: they exist to
    calibrate the tables, and nobody deploys a majority-class predictor.
    """
    if not table.exists():
        return pd.DataFrame()
    df = pd.read_excel(table)
    df = df[(df["protocol"] == protocol) & df["task"].isin(TASKS)]
    tuned = df[(df["selection"] == selection) & df["inner_score"].notna()]
    composed = df[df["selection"] == "composed"]
    df = pd.concat([tuned, composed], ignore_index=True)
    if df.empty:
        return df
    df["_rank"] = df["inner_score"].fillna(-np.inf)
    idx = df.groupby(["task", "model"])["_rank"].idxmax()
    return df.loc[idx].drop(columns="_rank").reset_index(drop=True)


def holdout_metrics(path: Path, protocol: str) -> dict:
    """
    Test-partition metrics keyed by (task, model, selection), for the manifest.

    Keyed on the selection as well as the model because one model appears under
    up to three of them and the scores differ. An earlier version filtered to a
    single selection before building the map, which left the two meta-ensembles
    - exported under "composed" while everything else exports under "weighted"
    - carrying an empty metrics block. The application ranks its menu by these
    metrics, so Stacking, the leading Cell Response model, sorted to the bottom
    of it.
    """
    if not path.exists():
        return {}
    df = pd.read_excel(path)
    for col, val in (("protocol", protocol), ("split", "test")):
        if col in df.columns:
            df = df[df[col] == val]
    keep = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1",
            "quadratic_kappa", "kappa", "mcc", "roc_auc_weighted_ovr"]
    out = {}
    for r in df.itertuples():
        out[(r.task, r.model, getattr(r, "selection", None))] = {
            k: round(float(getattr(r, k)), 4) for k in keep
            if hasattr(r, k) and pd.notna(getattr(r, k))}
    return out


def fingerprint(path: Path) -> tuple[int, str]:
    """
    Size and digest from ONE read of the bytes on disk.

    Not path.stat().st_size: the deployment directory lives on a Google Drive
    mount whose metadata lags the write, so a stat() taken immediately after
    joblib.dump or torch.save can report a size of zero for a file that is
    perfectly intact. Reading the file back forces the data through and makes
    the two fields consistent with each other by construction - a digest taken
    over a partially flushed file would be worse than no digest at all.
    """
    data = path.read_bytes()
    return len(data), hashlib.sha256(data).hexdigest()


def _design(task: str, pre, columns, df):
    sub, y = target_frame(df, task)
    X = np.asarray(pre.transform(sub[columns.predictors]), dtype=float)
    labels = sorted(pd.unique(y))
    code = {c: i for i, c in enumerate(labels)}
    return X, np.asarray([code[v] for v in y.to_numpy()]), labels


def export_ml(protocol: str, selection: str, dry_run: bool,
              skip_existing: bool = False) -> list[dict]:
    rows = best_params(ML_TABLES / "tuned_best_params.xlsx", protocol,
                       selection)
    if rows.empty:
        print(f"  [{protocol}] no tuned ML parameters - run 04/tuning.py first")
        return []

    df, columns = load_dataset()
    pre, _, _, _ = artifacts.load_release()
    metrics = holdout_metrics(ML_TABLES / "tuned_benchmark.xlsx", protocol)
    dest = OUT / "classifiers" / protocol
    dest.mkdir(parents=True, exist_ok=True)
    cache: dict = {}

    exported = []
    for r in rows.itertuples():
        task, name = r.task, r.model
        params = json.loads(r.params) if isinstance(r.params, str) else {}
        path = dest / f"{task}__{slugify(name)}.pkl"
        row_selection = getattr(r, "selection", selection)
        record = {"family": "ml", "task": task, "model": name,
                  "protocol": protocol, "selection": row_selection,
                  "file": f"classifiers/{protocol}/{path.name}",
                  "source_fold": r.fold,
                  "inner_score": (None if pd.isna(r.inner_score)
                                  else round(float(r.inner_score), 4)),
                  "params": params,
                  "metrics": metrics.get((task, name, row_selection), {})}
        if dry_run:
            exported.append(record | {"bytes": None})
            continue

        if skip_existing and path.exists() and path.stat().st_size:
            record["bytes"], record["sha256"] = fingerprint(path)
            record["reused"] = True
            exported.append(record)
            continue

        if task not in cache:
            cache[task] = _design(task, pre, columns, df)
        X, yv, labels = cache[task]
        record["classes"] = [int(c) for c in labels]

        try:
            if name in ss.BASELINES:
                est = zoo.build(name, n_jobs=1)
            elif name in ss.META:
                est = zoo.build(name, n_jobs=cfg.N_JOBS)
            else:
                est = ss.build_tuned(name, params, n_jobs=cfg.N_JOBS,
                                     n_classes=len(labels))
            est.fit(X, yv)
            joblib.dump({"estimator": est, "classes": labels, "task": task,
                         "model": name, "protocol": protocol},
                        path, compress=COMPRESS)
        except Exception as exc:
            print(f"    ! {task}/{name}: {type(exc).__name__}: {exc}"[:140])
            continue

        record["bytes"], record["sha256"] = fingerprint(path)
        exported.append(record)
        print(f"    {protocol:6s} {task:14s} {name:32s} "
              f"{record['bytes'] / 2**20:7.2f} MB", flush=True)
    return exported


def export_dl(protocol: str, selection: str, dry_run: bool,
              skip_existing: bool = False) -> list[dict]:
    rows = best_params(DL_TABLES / "dl_best_params.xlsx", protocol, selection)
    if rows.empty:
        print(f"  [{protocol}] no tuned DL parameters - run 05/tuning_dl.py")
        return []

    import torch
    from sklearn.model_selection import StratifiedShuffleSplit

    from mlate import deep, resources

    df, columns = load_dataset()
    pre, _, _, _ = artifacts.load_release()
    metrics = holdout_metrics(DL_TABLES / "dl_benchmark.xlsx", protocol)
    device = resources.devices()[0]
    dest = OUT / "deep" / protocol
    dest.mkdir(parents=True, exist_ok=True)
    cache: dict = {}

    exported = []
    for r in rows.itertuples():
        task, name = r.task, r.model
        params = json.loads(r.params) if isinstance(r.params, str) else {}
        path = dest / f"{task}__{slugify(name)}.pt"
        row_selection = getattr(r, "selection", selection)
        record = {"family": "dl", "task": task, "model": name,
                  "protocol": protocol, "selection": row_selection,
                  "file": f"deep/{protocol}/{path.name}",
                  "source_fold": r.fold,
                  "inner_score": (None if pd.isna(r.inner_score)
                                  else round(float(r.inner_score), 4)),
                  "params": params,
                  "metrics": metrics.get((task, name, row_selection), {})}
        if dry_run:
            exported.append(record | {"bytes": None})
            continue

        if skip_existing and path.exists() and path.stat().st_size:
            record["bytes"], record["sha256"] = fingerprint(path)
            record["reused"] = True
            exported.append(record)
            continue

        if task not in cache:
            X, yv, labels = _design(task, pre, columns, df)
            cache[task] = (X.astype(np.float32), yv, labels)
        X, yv, labels = cache[task]
        record["classes"] = [int(c) for c in labels]

        try:
            # Early stopping still needs a held-out slice; it comes out of the
            # training data, exactly as it did during tuning.
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15,
                                         random_state=cfg.RANDOM_STATE)
            tr, va = next(sss.split(X, yv))
            model, info = deep.train(name, params, X[tr], yv[tr], X[va],
                                     yv[va], len(labels), device, patience=25)
            torch.save({"state_dict": {k: v.cpu()
                                       for k, v in model.state_dict().items()},
                        "architecture": name, "params": params,
                        "input_dim": int(X.shape[1]),
                        "classes": [int(c) for c in labels],
                        "task": task, "protocol": protocol,
                        "best_epoch": info.get("best_epoch")}, path)
        except Exception as exc:
            print(f"    ! {task}/{name}: {type(exc).__name__}: {exc}"[:140])
            continue

        record["bytes"], record["sha256"] = fingerprint(path)
        exported.append(record)
        print(f"    {protocol:6s} {task:14s} {name:32s} "
              f"{record['bytes'] / 2**20:7.2f} MB", flush=True)
    return exported


def export_foundation(dry_run: bool,
                      skip_existing: bool = False) -> list[dict]:
    """
    Context sets for the in-context learners.

    There are no weights of ours to ship. What the application needs is the
    matrix these models condition on, plus the checkpoint identifier, so it can
    build the vendor classifier and call fit() once at start-up. Protocol does
    not enter: neither model has a hyper-parameter for a protocol to select.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                           / "05_deep_learning"))
    try:
        import foundation_models as fm
    except Exception as exc:
        print(f"  foundation: cannot import runner ({exc})"[:120])
        return []

    df, columns = load_dataset()
    pre, _, _, _ = artifacts.load_release()

    metrics = {}
    fnd = DL_TABLES / "foundation_models.xlsx"
    if fnd.exists():
        d = pd.read_excel(fnd, sheet_name="pooled")
        if "split" in d.columns:
            d = d[d["split"] == "test"]
        for r in d.itertuples():
            metrics[(r.task, r.model)] = {
                k: round(float(getattr(r, k)), 4)
                for k in ("accuracy", "macro_f1", "weighted_f1", "mcc")
                if hasattr(r, k) and pd.notna(getattr(r, k))}

    dest = OUT / "foundation"
    dest.mkdir(parents=True, exist_ok=True)
    exported = []
    for task in TASKS:
        X, yv, labels = _design(task, pre, columns, df)
        for name in getattr(fm, "DEFAULT_MODELS", ("TabPFN", "TabICL")):
            path = dest / f"{task}__{slugify(name)}.pkl"
            checkpoint = (getattr(fm, "MODEL_VERSION", None)
                          if name.startswith("TabPFN") else None)
            record = {"family": "foundation", "task": task, "model": name,
                      "protocol": "n/a", "selection": "zero_shot",
                      "file": f"foundation/{path.name}",
                      "checkpoint": checkpoint,
                      "n_estimators": getattr(fm, "N_ESTIMATORS", None),
                      "context_rows": int(X.shape[0]),
                      "classes": [int(c) for c in labels],
                      "metrics": metrics.get((task, name), {})}
            if dry_run:
                exported.append(record | {"bytes": None})
                continue
            if skip_existing and path.exists() and path.stat().st_size:
                record["bytes"], record["sha256"] = fingerprint(path)
                record["reused"] = True
                exported.append(record)
                continue
            joblib.dump({"context_X": X.astype(np.float32), "context_y": yv,
                         "classes": labels, "task": task, "model": name,
                         "checkpoint": checkpoint,
                         "n_estimators": record["n_estimators"]},
                        path, compress=COMPRESS)
            record["bytes"], record["sha256"] = fingerprint(path)
            exported.append(record)
            print(f"    context {task:14s} {name:24s} "
                  f"{record['bytes'] / 2**20:7.2f} MB", flush=True)
    return exported


def merge_manifest(fresh: list[dict]) -> list[dict]:
    """Fresh records win; previously recorded artefacts still on disk survive."""
    path = OUT / "deployment_manifest.json"
    if not path.exists():
        return fresh
    try:
        prior = json.loads(path.read_text(encoding="utf-8")).get("models", [])
    except Exception:
        return fresh

    def key(r):
        return (r.get("family"), r.get("task"), r.get("model"),
                r.get("protocol"))

    seen = {key(r) for r in fresh}
    kept = [r for r in prior
            if key(r) not in seen and (OUT / r.get("file", "")).exists()]
    return fresh + kept


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--families", nargs="*",
                    default=["ml", "dl", "foundation"],
                    choices=["ml", "dl", "foundation"])
    ap.add_argument("--protocols", nargs="*", default=list(PROTOCOLS),
                    choices=list(PROTOCOLS))
    ap.add_argument("--selection", default="weighted",
                    help="which tuned winner to ship")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-existing", action="store_true",
                    help="reuse artefacts already on disk and only rebuild "
                         "the manifest; refits nothing")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    exported = []
    for protocol in args.protocols:
        if "ml" in args.families:
            print(f"\nconventional models - {protocol}")
            exported += export_ml(protocol, args.selection, args.dry_run,
                                  args.skip_existing)
        if "dl" in args.families:
            print(f"\ndeep architectures - {protocol}")
            exported += export_dl(protocol, args.selection, args.dry_run,
                                  args.skip_existing)
    if "foundation" in args.families:
        print("\nfoundation model contexts")
        exported += export_foundation(args.dry_run, args.skip_existing)

    if not exported:
        raise SystemExit("nothing exported")

    # A partial run must not erase the rest of the bundle from the manifest.
    # Running --families foundation, or one protocol, describes only the slice
    # it rebuilt; entries for artefacts still sitting on disk are carried over
    # and only the ones this run actually touched are replaced.
    if not args.dry_run:
        exported = merge_manifest(exported)

    total = sum(r.get("bytes") or 0 for r in exported)
    manifest = {
        "created": date.today().isoformat(),
        "selection": args.selection,
        "protocols": list(args.protocols),
        "fitted_on": "the complete dataset, behind the release preprocessor",
        "selected_on": "the tuned configuration for each (task, protocol, "
                       "model), chosen inside the training partition by "
                       "10-fold cross-validation",
        "n_artefacts": len(exported),
        "total_bytes": total,
        "models": exported,
    }
    if not args.dry_run:
        (OUT / "deployment_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8")

    grid = pd.DataFrame(exported).groupby(
        ["family", "protocol", "task"]).size().rename("n")
    print("\n" + grid.to_string())
    print(f"\n{len(exported)} artefacts, {total / 2**20:,.1f} MB total")
    big = sorted((r for r in exported if r.get("bytes")),
                 key=lambda r: -r["bytes"])[:8]
    if big:
        print("largest:")
        for r in big:
            print(f"  {r['bytes'] / 2**20:8.2f} MB  {r['protocol']:6s} "
                  f"{r['task']}/{r['model']}")
    if total > 5 * 2**30:
        print("\n  ! over 5 GB - Hugging Face free repositories will struggle; "
              "consider dropping the heaviest ensembles")
    if not args.dry_run:
        print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
