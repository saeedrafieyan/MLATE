from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from mlate import config as cfg
from mlate import wssq as wssq_mod

try:  # pragma: no cover - import guard only
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except Exception:  # pragma: no cover
    optuna = None


@dataclass
class Variable:
    name: str
    low: float
    high: float
    step: float | None = None

    def suggest(self, trial):
        if self.low >= self.high:
            return float(self.low)
        if self.step:
            return float(trial.suggest_float(self.name, self.low, self.high,
                                             step=self.step))
        return float(trial.suggest_float(self.name, self.low, self.high))


@dataclass
class SearchSpace:
    cell_line: str
    biomaterials: list[Variable] = field(default_factory=list)
    printing: list[Variable] = field(default_factory=list)
    cell_density: Variable | None = None

    @property
    def is_acellular(self) -> bool:
        return self.cell_line == cfg.ACELLULAR_TOKEN


def class_distribution(estimator, X: np.ndarray, classes: np.ndarray
                       ) -> tuple[np.ndarray, list[dict[float, float]]]:
    labels = np.asarray(classes, dtype=float)
    if not hasattr(estimator, "predict_proba"):
        idx = np.asarray(estimator.predict(X)).astype(int)
        return labels[idx], [{} for _ in idx]
    proba = np.asarray(estimator.predict_proba(X), dtype=float)
    return proba @ labels, [{float(l): float(pr) for l, pr in zip(labels, row)}
                            for row in proba]


def expected_class_value(estimator, X: np.ndarray,
                         classes: np.ndarray) -> np.ndarray:
    labels = np.asarray(classes, dtype=float)
    if hasattr(estimator, "predict_proba"):
        return np.asarray(estimator.predict_proba(X), dtype=float) @ labels
    idx = np.asarray(estimator.predict(X)).astype(int)
    return labels[idx]


DEFAULT_BATCH_SIZE = 32


def _as_scorer(model):
    if isinstance(model, dict):
        return {"estimator": model["estimator"],
                "classes": np.asarray(model["classes"], dtype=float)}
    return {"estimator": model, "classes": np.asarray(model.classes,
                                                      dtype=float)}


class Objective:

    def __init__(self, space: SearchSpace, preprocessor, feature_columns,
                 print_model, cell_model,
                 print_weight: float = wssq_mod.DEFAULT_PRINT_WEIGHT,
                 cell_weight: float = wssq_mod.DEFAULT_CELL_WEIGHT):
        self.space = space
        self.pre = preprocessor
        self.feature_columns = list(feature_columns)
        self.print_model = _as_scorer(print_model)
        self.cell_model = _as_scorer(cell_model)
        self.print_weight = print_weight
        self.cell_weight = cell_weight
        self.history: list[dict] = []

    def candidate_frame(self, values: dict | list[dict]) -> pd.DataFrame:
        batch = [values] if isinstance(values, dict) else list(values)
        rows = []
        for v in batch:
            row = {c: 0.0 for c in self.feature_columns}
            row.update({k: x for k, x in v.items() if k in row})
            row[cfg.CELL_COLS[0]] = self.space.cell_line
            rows.append(row)
        return pd.DataFrame(rows)[self.feature_columns]

    def evaluate_batch(self, batch: list[dict]) -> list[dict]:
        X = np.asarray(self.pre.transform(self.candidate_frame(batch)),
                       dtype=float)

        exp_p, proba_p = class_distribution(
            self.print_model["estimator"], X, self.print_model["classes"])

        if self.space.is_acellular:
            exp_c = np.ones(len(batch))
            proba_c = [{}] * len(batch)
        else:
            exp_c, proba_c = class_distribution(
                self.cell_model["estimator"], X, self.cell_model["classes"])

        out = []
        for i in range(len(batch)):
            out.append({
                "wssq": float(wssq_mod.compute_wssq(
                    float(exp_p[i]), float(exp_c[i]),
                    self.print_weight, self.cell_weight)),
                "expected_printability": float(exp_p[i]),
                "expected_cell_response": float(exp_c[i]),
                "printability_proba": proba_p[i],
                "cell_response_proba": proba_c[i]})
        return out

    def evaluate(self, values: dict) -> dict:
        return self.evaluate_batch([values])[0]

    def suggest(self, trial) -> dict:
        values = {v.name: v.suggest(trial)
                  for v in self.space.biomaterials + self.space.printing}
        if self.space.cell_density is not None and not self.space.is_acellular:
            values[cfg.CELL_COLS[1]] = self.space.cell_density.suggest(trial)
        else:
            values[cfg.CELL_COLS[1]] = 0.0
        return values

    def record(self, values: dict, out: dict) -> None:
        self.history.append({**values, **{k: v for k, v in out.items()
                                          if not k.endswith("_proba")}})

    def __call__(self, trial) -> float:
        values = self.suggest(trial)
        out = self.evaluate(values)
        self.record(values, out)
        return out["wssq"]


def optimise(objective: Objective, n_trials: int = 100,
             seed: int = cfg.RANDOM_STATE, progress=None,
             batch_size: int = DEFAULT_BATCH_SIZE):
    if optuna is None:  # pragma: no cover
        raise RuntimeError("optuna is required for scaffold optimisation")

    sampler = optuna.samplers.TPESampler(
        seed=seed, n_startup_trials=min(30, max(5, n_trials // 4)),
        multivariate=True, group=True, consider_prior=True)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    done = 0
    while done < n_trials:
        size = min(max(1, batch_size), n_trials - done)
        trials = [study.ask() for _ in range(size)]
        batch = [objective.suggest(t) for t in trials]
        results = objective.evaluate_batch(batch)
        for trial, values, out in zip(trials, batch, results):
            objective.record(values, out)
            study.tell(trial, out["wssq"])
        done += size
        if progress is not None:
            progress(done, n_trials, study.best_value)

    return study.best_trial.params, float(study.best_value), study


def distance_report(values: dict, corpus: pd.DataFrame) -> dict:
    out_of_range = {}
    for name, v in values.items():
        if name not in corpus.columns:
            continue
        col = pd.to_numeric(corpus[name], errors="coerce")
        col = col[col > 0]
        if len(col) == 0:
            continue
        lo, hi = float(col.min()), float(col.max())
        if v > 0 and (v < lo or v > hi):
            out_of_range[name] = {"value": float(v), "observed_min": lo,
                                  "observed_max": hi}

    shared = [c for c in values if c in corpus.columns]
    nn = float("nan")
    if shared:
        sub = corpus[shared].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        span = (sub.max() - sub.min()).replace(0, 1.0)
        target = pd.Series({c: values[c] for c in shared})
        nn = float((((sub - target) / span) ** 2).sum(axis=1).pow(0.5).min())

    return {"n_out_of_range": len(out_of_range),
            "out_of_range": out_of_range,
            "nearest_neighbour_distance": nn}
