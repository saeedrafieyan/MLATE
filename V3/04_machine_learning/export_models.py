from __future__ import annotations

import hashlib
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
import sklearn

from mlate import config as cfg
from mlate import models as zoo
from mlate import resources
from mlate.dataset import TASKS, load_dataset, target_frame
from mlate.pipeline import build_preprocessor, feature_names

TABLES = cfg.step_dir("04_machine_learning", "tables")
MODELS = cfg.step_dir("04_machine_learning", "models")
DEPLOY = cfg.MODEL_DIR / "classifiers"
TOP_N = 3
SELECT_ON = ("doi", "macro_f1")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    budget = resources.claim()
    devices = resources.devices()
    DEPLOY.mkdir(parents=True, exist_ok=True)

    board = pd.read_excel(TABLES / "model_benchmark.xlsx",
                          sheet_name="leaderboard")
    protocol, metric = SELECT_ON
    df, columns = load_dataset()

    manifest = {
        "created": date.today().isoformat(),
        "selected_on": {"protocol": protocol, "metric": metric},
        "dataset": {"rows": int(len(df)), "studies": int(df["DOI"].nunique()),
                    "biomaterials": len(columns.biomaterials)},
        "versions": {"scikit-learn": sklearn.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
        "tasks": {},
    }

    for task in TASKS:
        cell = board[(board["task"] == task) & (board["protocol"] == protocol)]
        if cell.empty:
            print(f"  {task}: no {protocol} results, skipping")
            continue
        ranked = cell[cell["family"] != "baseline"].nlargest(TOP_N, metric)
        baseline = cell[cell["model"] == "Dummy (majority)"]

        sub, y = target_frame(df, task)
        pre = build_preprocessor(columns).fit(sub[columns.predictors])
        X = np.asarray(pre.transform(sub[columns.predictors]), dtype=float)

        entries = []
        for rank, row in enumerate(ranked.itertuples(), start=1):
            model = zoo.build(row.model, n_jobs=budget.n_jobs,
                              device=devices[0] if devices else None)
            model.fit(X, y.to_numpy())

            slug = row.model.lower()
            for ch in " ()+->/":
                slug = slug.replace(ch, "_")
            slug = "_".join(filter(None, slug.split("_")))
            path = DEPLOY / f"{task}__{rank}__{slug}.pkl"
            joblib.dump({"model": model, "preprocessor": pre,
                         "input_columns": list(columns.predictors),
                         "feature_names": feature_names(pre),
                         "classes": [int(c) for c in np.unique(y)],
                         "task": task, "name": row.model},
                        path, compress=3)

            entries.append({
                "rank": rank, "name": row.model, "family": row.family,
                "file": path.name,
                "sha256": _sha256(path),
                "size_kb": round(path.stat().st_size / 1024, 1),
                f"{protocol}_accuracy": round(float(row.accuracy), 4),
                f"{protocol}_balanced_accuracy": round(
                    float(row.balanced_accuracy), 4),
                f"{protocol}_macro_f1": round(float(row.macro_f1), 4),
                f"{protocol}_quadratic_kappa": round(
                    float(row.quadratic_kappa), 4),
                f"{protocol}_mcc": round(float(row.mcc), 4),
            })
            print(f"  {task:24s} #{rank} {row.model:28s} "
                  f"{metric} {getattr(row, metric):.3f}  "
                  f"{entries[-1]['size_kb']:7.1f} KB")

        manifest["tasks"][task] = {
            "classes": [int(c) for c in np.unique(y)],
            "n_samples": int(len(sub)),
            "class_distribution": {str(k): int(v)
                                   for k, v in y.value_counts().sort_index().items()},
            "baseline_majority": {
                "accuracy": round(float(baseline["accuracy"].iloc[0]), 4),
                "macro_f1": round(float(baseline["macro_f1"].iloc[0]), 4),
            } if len(baseline) else None,
            "models": entries,
        }

    with open(DEPLOY / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    total = sum(p.stat().st_size for p in DEPLOY.glob("*.pkl")) / 1024
    print(f"\nbundle: {len(list(DEPLOY.glob('*.pkl')))} models, "
          f"{total:.1f} KB total")
    print(f"-> {DEPLOY}  (git-ignored; Hugging Face only)")


if __name__ == "__main__":
    main()
