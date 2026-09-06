"""
Tabular foundation models: TabPFN and TabICL
============================================

    python 05_deep_learning/foundation_models.py --dry-run
    python 05_deep_learning/foundation_models.py
    python 05_deep_learning/foundation_models.py --models TabPFN --protocols doi

Both models are pre-trained transformers that classify by in-context learning:
the training rows are supplied as context at inference time and the weights are
never updated on our data. There is therefore nothing to tune - which is why
this step has no search - but everything else matches step 04 and the deep
models: the same folds, the same fold-fitted preprocessor, out-of-fold
predictions in the same schema, per-unit checkpoints and resume.

Study grouping still applies, and that is worth stating plainly because the
opposite is easy to assume. The weights never saw this corpus, but the
*prediction* still conditions on whatever training rows are placed in the
context window. If a test row's own publication sits in that context, the model
can match against it exactly as a fitted model would - the mechanism is
attention over provided examples rather than memorised parameters, but the
information available is identical. Running both protocols turns that into a
measurement rather than an assumption: if in-context learning were immune to
study structure, the random and grouped scores would coincide.

Access tokens
-------------
TabICL downloads its weights unauthenticated. **TabPFN 2.6 does not** - it
requires a Prior Labs account token before it will fetch a checkpoint, and
without one it opens a browser, starts a localhost callback server and blocks
on stdin. In a script that surfaces as

    OSError: [WinError 10038] An operation was attempted on something that
             is not a socket

which reads like a multiprocessing fault and is nothing of the kind. Put the
token in 05_deep_learning/.env (any of the key spellings in TOKEN_KEYS, case
insensitive) or in the environment; apply_token_env() exports it as
TABPFN_TOKEN and sets TABPFN_NO_BROWSER so a missing token fails loudly instead
of hanging.

A token must never be written into a source file. The submitted repository's
transformer_benchmark.py carries a live JWT on line 24; that one has since
expired, and any token committed to a public repository should be treated as
compromised and rotated regardless.

Reproducibility note for the Methods: this step therefore needs an account,
unlike every other step in the repository. Worth one sentence, since Referee 1
comment 4 is about a reader being able to reproduce the work.
"""

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
# The exact TabPFN checkpoint, so the manuscript can name the model that
# produced the numbers. This must be a filename the package offers, not a
# version string: `model_path="v2.6"` raises
#   ValueError: Model v2.6 not found in available models:
#       ['tabpfn-v2.6-classifier-v2.6_default.ckpt']
# Passing None would take the default, but then a future default silently
# changes the results.
MODEL_VERSION = "tabpfn-v2.6-classifier-v2.6_default.ckpt"
DEFAULT_MODELS = ("TabPFN", "TabPFN (thinking)", "TabICL")


# Where a token may live, and what it may be called. Both lists are searched
# because a token is easy to put in a reasonable place that the loader does not
# happen to check, and a silently-missing token degrades to a different code
# path rather than to an error - which is the worst way for this to fail.
ENV_FILES = (Path(__file__).resolve().parent / ".env", cfg.ROOT / ".env")
TOKEN_KEYS = ("TABPFN_TOKEN", "TABPFN_ACCESS_TOKEN", "PRIORLABS_TOKEN",
              "PRIOR_LABS_TOKEN")


def token() -> str | None:
    """
    Hosted-API token, from the environment or an uncommitted .env.

    Never from a source file: the submitted repository's
    transformer_benchmark.py carries a live JWT on line 24, which is exactly
    the failure this function exists to avoid. Key matching is
    case-insensitive because .env files are written by people.
    """
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
    """Describe the token without ever revealing it."""
    tok = token()
    if not tok:
        return "not set - using locally downloaded weights"
    return f"present ({len(tok)} chars, ...{tok[-4:]})"


def apply_token_env() -> bool:
    """
    Put the token where TabPFN looks for it, and stop it opening a browser.

    TabPFN 2.6 will not download its weights unauthenticated. Without
    TABPFN_TOKEN it starts an interactive login - it opens a browser, spins up
    a callback server on localhost, and blocks on stdin waiting for an API key
    pasted by hand. In any non-interactive process that collapses into

        OSError: [WinError 10038] An operation was attempted on something
                 that is not a socket

    which looks like a multiprocessing fault and is not one. Reading the token
    is not enough; it has to be exported, and TABPFN_NO_BROWSER has to be set
    so a failure surfaces as an error rather than as a process hanging on a
    login prompt no one can see.

    Called in the parent and again inside each worker, because a worker that
    somehow starts without it would hang rather than fail.
    """
    tok = token()
    os.environ.setdefault("TABPFN_NO_BROWSER", "1")
    if tok:
        os.environ["TABPFN_TOKEN"] = tok
        return True
    return False


def build(name: str, device: str):
    """
    TabPFN is evaluated in two inference modes.

    "non-thinking" is the default forward pass: one in-context prediction per
    ensemble member, argmax of the averaged probabilities.

    "thinking" spends additional compute at inference time. TabPFN-3 exposes
    this through its tuning configuration: it holds out part of the supplied
    context, calibrates a softmax temperature on it, and searches per-class
    decision thresholds against a chosen metric. No weights change and no extra
    data is used - the model reconsiders how to turn its own probabilities into
    decisions. This is the mode that should matter here, because the plain
    model ranks well while deciding badly, which is the signature of a
    threshold problem rather than a signal problem.

    Thresholds are tuned against F1 so the tuning objective matches the metric
    the paper reports, and the holdout comes from the training context only.
    """
    if name.startswith("TabPFN"):
        from tabpfn import TabPFNClassifier
        from tabpfn.inference_tuning import ClassifierTuningConfig
        kwargs = dict(device=device, n_estimators=N_ESTIMATORS,
                      # 153 features exceeds the pre-training shape; permit it.
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
    """Identical resolution to steps 04 and 05, so all three share test rows."""
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
    """One (model, task, protocol, fold) job. Writes its own checkpoint."""
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
        # In-context "training" performance means predicting the very rows
        # supplied as context, so it is an upper bound of a different kind
        # from a fitted model's training score. It is recorded for consistency
        # with steps 04 and 05, and is a diagnostic only.
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

    # Pooled scoring, matching step 04: concatenate out-of-fold predictions and
    # score once, rather than averaging per-fold scores.
    pooled = []
    for (model, task, protocol, split), g in allp.groupby(
            ["model", "task", "protocol", "split"]):
        labels = sorted(pd.unique(target_frame(df, task)[1]))
        pcols = [f"p_{c}" for c in labels]
        proba = g[pcols].to_numpy(float) if all(c in g for c in pcols) else None
        # `selection` is carried even though nothing is selected here: steps 04
        # and 05 both emit it, and a downstream table that filters on it would
        # silently drop every foundation-model row if this column were absent.
        pooled.append({"model": model, "task": task, "protocol": protocol,
                       "selection": "zero_shot", "split": split,
                       "n_scored": len(g), "n_folds": g["fold"].nunique(),
                       **ev.scores(g["y_true"], g["y_pred"], proba, labels)})
    pooled = pd.DataFrame(pooled)
    with pd.ExcelWriter(TABLES / "foundation_models.xlsx") as xl:
        pooled.round(4).to_excel(xl, sheet_name="pooled", index=False)

    # Everything below is TEST ONLY. Filtering here is not cosmetic: the table
    # carries a train row per model, and a pivot_table over an unfiltered frame
    # silently AVERAGES train and test into one number that is neither. That
    # produced a protocol-gap table where TabICL/cell_response read 0.592,
    # exactly the mean of its 0.392 test and 0.792 train macro F1.
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
    # Foundation models hold a large transformer plus the whole training set as
    # context on the device, so far fewer fit per card than the small networks
    # in tuning_dl.py.
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
    # batch_size=1 is essential, not a tuning knob. joblib defaults to
    # batch_size="auto", which groups short tasks together to amortise
    # dispatch overhead. Most units here are cheap, so joblib grows the
    # batch - and then a single worker can receive one pathological
    # unit with sixty cheap ones queued BEHIND it, while every other
    # worker drains its batch and exits. Observed exactly that: one
    # worker at 16.9 CPU-hours on a single SVM fit, 53 workers gone,
    # 67 cheap units never dispatched, 63 cores idle overnight.
    # Batching also silently defeats the longest-first ordering that
    # mlate/scheduling.py exists to compute.
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
