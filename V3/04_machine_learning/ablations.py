from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import StratifiedKFold

from mlate import config as cfg
from mlate import evaluation as ev
from mlate import models as zoo
from mlate import resources, splits
from mlate.artifacts import fold_preprocessor
from mlate.dataset import load_dataset, target_frame

warnings.filterwarnings("ignore")

TABLES = cfg.step_dir("04_machine_learning", "tables")

MODELS = ["Bernoulli Naive Bayes", "Random Forest (balanced)", "XGBoost",
          "LightGBM", "Logistic Regression (balanced)",
          "Hist Gradient Boosting"]
PROTOCOLS = ("random", "doi")


class Ordinal(BaseEstimator, ClassifierMixin):

    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.estimators_ = []
        for k in self.classes_[:-1]:
            est = clone(self.estimator)
            est.fit(X, (y > k).astype(int))
            self.estimators_.append(est)
        return self

    def predict_proba(self, X):
        cum = np.column_stack([e.predict_proba(X)[:, 1]
                               for e in self.estimators_])
        cum = np.minimum.accumulate(cum, axis=1)
        n, K = len(cum), len(self.classes_)
        out = np.zeros((n, K))
        out[:, 0] = 1.0 - cum[:, 0]
        for i in range(1, K - 1):
            out[:, i] = cum[:, i - 1] - cum[:, i]
        out[:, -1] = cum[:, -1]
        out = np.clip(out, 1e-9, None)
        return out / out.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def _resample(kind, X, y, seed=cfg.RANDOM_STATE):
    counts = pd.Series(y).value_counts()
    if kind == "smote":
        from imblearn.over_sampling import SMOTE
        k = int(min(5, counts.min() - 1))
        if k < 1:
            return X, y
        return SMOTE(random_state=seed, k_neighbors=k).fit_resample(X, y)
    if kind == "ros":
        from imblearn.over_sampling import RandomOverSampler
        return RandomOverSampler(random_state=seed).fit_resample(X, y)
    return X, y


def _printability_feature(df, columns, fold, kind, budget):
    truth = df["Printability"].to_numpy(int)
    if kind == "true":
        return truth.astype(float)

    pre, _, _ = fold_preprocessor(df, columns, fold.train_idx)
    feats = df[columns.predictors]
    Xtr = np.asarray(pre.transform(feats.iloc[fold.train_idx]), float)
    Xte = np.asarray(pre.transform(feats.iloc[fold.test_idx]), float)
    ytr = truth[fold.train_idx]

    out = np.zeros(len(df), dtype=float)
    inner = StratifiedKFold(n_splits=5, shuffle=True,
                            random_state=cfg.RANDOM_STATE)
    for a, b in inner.split(Xtr, ytr):
        m = zoo.build("Random Forest (balanced)", n_jobs=budget.n_jobs)
        m.fit(Xtr[a], ytr[a])
        out[fold.train_idx[b]] = m.predict(Xtr[b])
    m = zoo.build("Random Forest (balanced)", n_jobs=budget.n_jobs)
    m.fit(Xtr, ytr)
    out[fold.test_idx] = m.predict(Xte)
    return out


def run_condition(df, columns, task, protocol, condition, budget) -> list[dict]:
    sub, y = target_frame(df, task)
    labels = sorted(pd.unique(y))
    folds = splits.make_folds(sub, y, protocols=(protocol,))
    rows = []

    for fold in folds:
        pre, _, _ = fold_preprocessor(sub, columns, fold.train_idx)
        feats = sub[columns.predictors]
        Xtr = np.asarray(pre.transform(feats.iloc[fold.train_idx]), float)
        Xte = np.asarray(pre.transform(feats.iloc[fold.test_idx]), float)

        if condition in ("+printability", "+pred_print"):
            kind = "true" if condition == "+printability" else "pred"
            extra = _printability_feature(sub, columns, fold, kind, budget)
            Xtr = np.column_stack([Xtr, extra[fold.train_idx] / 3.0])
            Xte = np.column_stack([Xte, extra[fold.test_idx] / 3.0])

        ytr = y.to_numpy()[fold.train_idx]
        yte = y.to_numpy()[fold.test_idx]
        if condition in ("smote", "ros"):
            Xtr, ytr = _resample(condition, Xtr, ytr)

        code = {c: i for i, c in enumerate(labels)}
        decode = np.asarray(labels)
        ytr_enc = np.asarray([code[v] for v in ytr])

        for name in MODELS:
            try:
                base = zoo.build(name, n_jobs=budget.n_jobs)
                model = Ordinal(base) if condition == "ordinal" else base
                model.fit(Xtr, ytr_enc)
                pred = decode[np.asarray(model.predict(Xte)).astype(int)]
                proba = (model.predict_proba(Xte)
                         if hasattr(model, "predict_proba") else None)
                rows.append({"task": task, "protocol": protocol,
                             "condition": condition, "model": name,
                             "fold": fold.name, "y_true": yte, "y_pred": pred,
                             "proba": proba})
            except Exception as exc:
                print(f"    ! {name} / {condition}: "
                      f"{type(exc).__name__}: {str(exc)[:90]}")
    return rows


def main() -> None:
    budget = resources.claim()
    df, columns = load_dataset()

    plan = {
        "printability": ["base", "smote", "ros", "ordinal"],
        "cell_response_cellular": ["base", "smote", "ros", "ordinal",
                                   "+printability", "+pred_print"],
    }

    records = []
    for task, conditions in plan.items():
        for protocol in PROTOCOLS:
            for condition in conditions:
                t0 = time.perf_counter()
                rows = run_condition(df, columns, task, protocol, condition,
                                     budget)
                labels = sorted(pd.unique(target_frame(df, task)[1]))
                for r in rows:
                    records.append({
                        "task": r["task"], "protocol": r["protocol"],
                        "condition": r["condition"], "model": r["model"],
                        "fold": r["fold"],
                        **ev.scores(r["y_true"], r["y_pred"], r["proba"],
                                    labels),
                    })
                print(f"  {task:24s} {protocol:7s} {condition:15s} "
                      f"{time.perf_counter() - t0:6.1f}s")

    raw = pd.DataFrame(records)
    summary = (raw.groupby(["task", "protocol", "condition", "model"])
               [["accuracy", "balanced_accuracy", "macro_f1",
                 "quadratic_kappa", "mcc"]].mean().reset_index())

    base = (summary[summary["condition"] == "base"]
            .set_index(["task", "protocol", "model"])["macro_f1"])
    summary["macro_f1_vs_base"] = summary.apply(
        lambda r: r["macro_f1"] - base.get((r["task"], r["protocol"],
                                            r["model"]), np.nan), axis=1)

    with pd.ExcelWriter(TABLES / "ablations.xlsx") as xl:
        summary.round(4).to_excel(xl, sheet_name="by_model", index=False)
        (summary.groupby(["task", "protocol", "condition"])
         [["macro_f1", "balanced_accuracy", "quadratic_kappa",
           "macro_f1_vs_base"]].mean().round(4).reset_index()
         .to_excel(xl, sheet_name="by_condition", index=False))
        raw.round(4).to_excel(xl, sheet_name="per_fold", index=False)

    pd.set_option("display.width", 200)
    print("\nmean over the six models, change in macro F1 against base")
    piv = (summary.pivot_table(index=["task", "protocol"], columns="condition",
                               values="macro_f1_vs_base")
           .round(3))
    print(piv.to_string())
    print("\nabsolute macro F1")
    print((summary.pivot_table(index=["task", "protocol"], columns="condition",
                               values="macro_f1").round(3)).to_string())
    print(f"\n-> {TABLES / 'ablations.xlsx'}")


if __name__ == "__main__":
    main()
