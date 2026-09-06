"""
Does domain feature engineering close the protocol gap?
=======================================================

    python 04_machine_learning/feature_engineering.py

The benchmark's own result motivates this test. Bernoulli naive Bayes - which
treats every feature as an independent binary indicator - leads the grouped
leaderboard on all three tasks, which says the transferable signal is close to
"which components are present" rather than their exact amounts. If that is
right, then features encoding *what kind* of component is present, rather than
*which specific* one, should transfer better: a laboratory using a different
gelatin-like polymer becomes recognisable instead of looking unrelated.

Feature blocks
--------------
class_agg  total concentration within each of the 13 biomaterial classes of the
           curated taxonomy. Gelatin and collagen are different columns but the
           same class, so a study substituting one for the other is invisible to
           the raw matrix and obvious here.
totals     total polymer, total crosslinker, crosslinker-to-polymer ratio and
           number of distinct components. Formulation-level descriptors that do
           not depend on which specific materials were chosen.
physics    a wall-shear-stress proxy and a speed-to-diameter ratio. For flow
           through a cylindrical nozzle the wall shear stress is
           tau = dP * D / (4L); with nozzle length unknown but roughly constant
           across the corpus, the product of extrusion pressure and nozzle
           diameter is proportional to it. The model currently sees pressure and
           diameter as unrelated numbers, but the physics combines them, and
           shear stress is the mechanism by which extrusion damages cells.
all        every block at once.

Leakage
-------
Every engineered feature is a row-wise function of that row's own values, so
computing them on the full frame cannot leak. The fold-dependent steps -
imputing a missing engineered value and scaling - are fitted on the training
partition only, exactly as in the main pipeline.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from mlate import config as cfg
from mlate import evaluation as ev
from mlate import models as zoo
from mlate import resources, splits
from mlate.artifacts import fold_preprocessor
from mlate.dataset import load_dataset, load_taxonomy, target_frame

warnings.filterwarnings("ignore")

TABLES = cfg.step_dir("04_machine_learning", "tables")

MODELS = ["Bernoulli Naive Bayes", "Random Forest (balanced)", "XGBoost",
          "LightGBM", "Logistic Regression (balanced)",
          "Hist Gradient Boosting"]
PROTOCOLS = ("random", "doi")
BLOCKS = ("base", "class_agg", "totals", "physics", "all")

POLYMER_CLASSES = {"natural_polymer", "modified_natural_polymer",
                   "synthetic_polymer", "ECM_derived"}
CROSSLINK_CLASSES = {"crosslinker", "initiator"}


def engineer(df: pd.DataFrame, columns) -> dict[str, pd.DataFrame]:
    """Build each feature block. Row-wise only."""
    tax = load_taxonomy().set_index("column")["material_class"]
    bio = df[columns.biomaterials].fillna(0.0)
    present = bio > 0

    # ── class aggregates ────────────────────────────────────────────────────
    class_agg = pd.DataFrame(index=df.index)
    for cls in sorted(set(tax.values)):
        cols = [c for c in columns.biomaterials if tax.get(c) == cls]
        class_agg[f"class_sum[{cls}]"] = bio[cols].sum(axis=1)
        class_agg[f"class_n[{cls}]"] = present[cols].sum(axis=1)

    # ── formulation totals ──────────────────────────────────────────────────
    poly_cols = [c for c in columns.biomaterials
                 if tax.get(c) in POLYMER_CLASSES]
    xl_cols = [c for c in columns.biomaterials
               if tax.get(c) in CROSSLINK_CLASSES]
    total_polymer = bio[poly_cols].sum(axis=1)
    total_xl = bio[xl_cols].sum(axis=1)
    totals = pd.DataFrame({
        "total_polymer": total_polymer,
        "total_crosslinker": total_xl,
        # Guarded: a formulation with no polymer would divide by zero.
        "crosslinker_per_polymer": total_xl / total_polymer.replace(0, np.nan),
        "n_components": present.sum(axis=1),
        "n_material_classes": class_agg[[c for c in class_agg
                                         if c.startswith("class_n")]].gt(0).sum(axis=1),
    }, index=df.index)

    # ── printing physics ────────────────────────────────────────────────────
    P = pd.to_numeric(df["Extrusion Pressure (kPa)"], errors="coerce")
    D = pd.to_numeric(df["Nozzle Diameter (µm)"], errors="coerce")
    V = pd.to_numeric(df["Nozzle Movement Speed (mm/s)"], errors="coerce")
    physics = pd.DataFrame({
        # tau_wall = dP*D/(4L); L unknown but near-constant, so tau ~ P*D.
        "shear_stress_proxy": P * D,
        "pressure_per_diameter": P / D.replace(0, np.nan),
        "speed_per_diameter": V / D.replace(0, np.nan),
    }, index=df.index)

    return {"class_agg": class_agg, "totals": totals, "physics": physics}


def block_for(block: str, parts: dict) -> pd.DataFrame | None:
    if block == "base":
        return None
    if block == "all":
        return pd.concat(parts.values(), axis=1)
    return parts[block]


def run(df, columns, task, protocol, block, parts, budget) -> list[dict]:
    sub, y = target_frame(df, task)
    labels = sorted(pd.unique(y))
    extra_all = block_for(block, parts)
    folds = splits.make_folds(sub, y, protocols=(protocol,))
    rows = []

    for fold in folds:
        pre, _, _ = fold_preprocessor(sub, columns, fold.train_idx)
        feats = sub[columns.predictors]
        Xtr = np.asarray(pre.transform(feats.iloc[fold.train_idx]), float)
        Xte = np.asarray(pre.transform(feats.iloc[fold.test_idx]), float)

        if extra_all is not None:
            e = extra_all.loc[sub.index]
            etr = e.iloc[fold.train_idx].to_numpy(dtype=float)
            ete = e.iloc[fold.test_idx].to_numpy(dtype=float)
            # Impute and scale on the training partition only.
            med = np.nanmedian(etr, axis=0)
            med = np.where(np.isnan(med), 0.0, med)
            etr = np.where(np.isnan(etr), med, etr)
            ete = np.where(np.isnan(ete), med, ete)
            lo, hi = etr.min(axis=0), etr.max(axis=0)
            span = np.where(hi - lo == 0, 1.0, hi - lo)
            Xtr = np.column_stack([Xtr, (etr - lo) / span])
            Xte = np.column_stack([Xte, np.clip((ete - lo) / span, -1, 2)])

        ytr = y.to_numpy()[fold.train_idx]
        yte = y.to_numpy()[fold.test_idx]
        code = {c: i for i, c in enumerate(labels)}
        decode = np.asarray(labels)
        ytr_enc = np.asarray([code[v] for v in ytr])

        for name in MODELS:
            try:
                model = zoo.build(name, n_jobs=budget.n_jobs)
                model.fit(Xtr, ytr_enc)
                pred = decode[np.asarray(model.predict(Xte)).astype(int)]
                proba = (model.predict_proba(Xte)
                         if hasattr(model, "predict_proba") else None)
                rows.append({"task": task, "protocol": protocol,
                             "block": block, "model": name, "fold": fold.name,
                             **ev.scores(yte, pred, proba, labels)})
            except Exception as exc:
                print(f"    ! {name} / {block}: "
                      f"{type(exc).__name__}: {str(exc)[:80]}")
    return rows


def main() -> None:
    budget = resources.claim()
    df, columns = load_dataset()
    parts = engineer(df, columns)
    print(f"\nengineered features: "
          + ", ".join(f"{k} {v.shape[1]}" for k, v in parts.items())
          + f"  (total {sum(v.shape[1] for v in parts.values())})")

    records = []
    for task in ("printability", "cell_response_cellular"):
        for protocol in PROTOCOLS:
            for block in BLOCKS:
                t0 = time.perf_counter()
                records += run(df, columns, task, protocol, block, parts,
                               budget)
                print(f"  {task:24s} {protocol:7s} {block:10s} "
                      f"{time.perf_counter() - t0:6.1f}s")

    raw = pd.DataFrame(records)
    summary = (raw.groupby(["task", "protocol", "block", "model"])
               [["accuracy", "balanced_accuracy", "macro_f1",
                 "quadratic_kappa", "mcc"]].mean().reset_index())
    base = (summary[summary["block"] == "base"]
            .set_index(["task", "protocol", "model"])["macro_f1"])
    summary["macro_f1_vs_base"] = summary.apply(
        lambda r: r["macro_f1"] - base.get((r["task"], r["protocol"],
                                            r["model"]), np.nan), axis=1)

    with pd.ExcelWriter(TABLES / "feature_engineering.xlsx") as xl:
        summary.round(4).to_excel(xl, sheet_name="by_model", index=False)
        (summary.groupby(["task", "protocol", "block"])
         [["macro_f1", "balanced_accuracy", "quadratic_kappa",
           "macro_f1_vs_base"]].mean().round(4).reset_index()
         .to_excel(xl, sheet_name="by_block", index=False))
        raw.round(4).to_excel(xl, sheet_name="per_fold", index=False)

    pd.set_option("display.width", 200)
    order = list(BLOCKS)
    print("\nchange in macro F1 against base, mean over six models")
    print(summary.pivot_table(index=["task", "protocol"], columns="block",
                              values="macro_f1_vs_base")[order].round(3)
          .to_string())
    print("\nabsolute macro F1")
    print(summary.pivot_table(index=["task", "protocol"], columns="block",
                              values="macro_f1")[order].round(3).to_string())
    print("\nbest block per model, DOI-grouped only")
    doi = summary[summary["protocol"] == "doi"]
    print(doi.pivot_table(index="model", columns=["task", "block"],
                          values="macro_f1_vs_base").round(3).to_string())
    print(f"\n-> {TABLES / 'feature_engineering.xlsx'}")


if __name__ == "__main__":
    main()
