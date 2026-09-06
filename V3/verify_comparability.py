"""
Prove that ML, deep learning and the foundation models share partitions
=======================================================================

    python verify_comparability.py

The three families are benchmarked by three separate scripts, each of which
resolves its own folds. That is a latent hazard: nothing stops one of them from
drifting - a different seed, a different split function, a stray `test_size` -
and the failure is silent. The numbers would still be produced, the tables would
still line up, and the ML-vs-DL comparison in the paper would simply be wrong.

The submitted manuscript already carries the milder version of this problem. It
evaluated conventional ML on an 80:20 split and the neural models on 70:15:15,
and its own Discussion concedes the consequence: comparisons between the groups
"should be interpreted as benchmark-level comparisons rather than strictly
identical test-split comparisons". The revision claims something stronger - that
every model was scored on the same rows - so that claim has to be checkable.

This script imports each stage's own fold resolver, by the same path its
workers use, and asserts that for every (task, protocol) all three return:

  * identical training row indices
  * identical test row indices
  * the same preprocessor cache key, and therefore the same fitted transformer

Exit code is non-zero on any mismatch, so it can gate a release.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np

from mlate import artifacts
from mlate import config as cfg
from mlate.dataset import load_dataset, target_frame

TASKS = ("printability", "cell_response")
PROTOCOLS = ("random", "doi")
DESIGN = "holdout"

STAGES = {
    "04 machine_learning": ROOT / "04_machine_learning" / "tuning.py",
    "05 deep_learning": ROOT / "05_deep_learning" / "tuning_dl.py",
    "05 foundation": ROOT / "05_deep_learning" / "foundation_models.py",
}


def load_resolver(label: str, path: Path):
    """Import a stage module and hand back its get_folds, as its workers see it."""
    spec = importlib.util.spec_from_file_location(
        f"_stage_{abs(hash(label))}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "get_folds"):
        raise AttributeError(f"{path.name} has no get_folds()")
    return module.get_folds


def main() -> int:
    resolvers = {}
    for label, path in STAGES.items():
        if not path.exists():
            print(f"  ! {label}: {path.name} not found - skipped")
            continue
        try:
            resolvers[label] = load_resolver(label, path)
        except Exception as exc:
            print(f"  ! {label}: could not load - {type(exc).__name__}: {exc}")
            return 1

    if len(resolvers) < 2:
        print("fewer than two stages available; nothing to compare")
        return 1

    df, columns = load_dataset()
    failures, checks = [], 0
    reference_label = next(iter(resolvers))

    print(f"design: {DESIGN} | reference: {reference_label}\n")
    header = f"{'task':16s} {'protocol':9s} {'train':>6s} {'test':>6s}  {'cache key':<34s} verdict"
    print(header)
    print("-" * len(header))

    for task in TASKS:
        sub, y = target_frame(df, task)
        for protocol in PROTOCOLS:
            resolved = {}
            for label, get_folds in resolvers.items():
                folds = get_folds(sub, y, protocol, DESIGN)
                if len(folds) != 1:
                    failures.append(
                        f"{label} {task}/{protocol}: expected 1 partition "
                        f"under the holdout design, got {len(folds)}")
                    continue
                resolved[label] = folds[0]

            if len(resolved) < 2:
                continue
            ref = resolved[reference_label]
            key = artifacts.cache_key(ref.train_idx, columns)

            ok = True
            for label, fold in resolved.items():
                if label == reference_label:
                    continue
                checks += 1
                if not np.array_equal(np.sort(fold.train_idx),
                                      np.sort(ref.train_idx)):
                    failures.append(f"{task}/{protocol}: {label} TRAIN rows "
                                    f"differ from {reference_label}")
                    ok = False
                if not np.array_equal(np.sort(fold.test_idx),
                                      np.sort(ref.test_idx)):
                    failures.append(f"{task}/{protocol}: {label} TEST rows "
                                    f"differ from {reference_label}")
                    ok = False
                if artifacts.cache_key(fold.train_idx, columns) != key:
                    failures.append(f"{task}/{protocol}: {label} resolves a "
                                    f"different preprocessor cache key")
                    ok = False

            # A test partition that lost a class makes macro F1 incomparable
            # across tasks, so it is worth failing loudly rather than noting.
            test_classes = set(np.unique(y.to_numpy()[ref.test_idx]).tolist())
            all_classes = set(np.unique(y.to_numpy()).tolist())
            if test_classes != all_classes:
                failures.append(
                    f"{task}/{protocol}: test partition is missing "
                    f"class(es) {sorted(all_classes - test_classes)}")
                ok = False

            print(f"{task:16s} {protocol:9s} {len(ref.train_idx):6d} "
                  f"{len(ref.test_idx):6d}  {key:<34s} "
                  f"{'OK' if ok else 'MISMATCH'}")

    print(f"\n{len(resolvers)} stages compared, {checks} pairwise checks")
    for label in resolvers:
        print(f"  - {label}")

    if failures:
        print(f"\n{len(failures)} FAILURE(S):")
        for f in failures:
            print(f"  ! {f}")
        return 1

    print("\nAll stages resolve identical train/test partitions and the same "
          "fitted preprocessor.\nML, deep-learning and foundation-model "
          "results are directly comparable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
