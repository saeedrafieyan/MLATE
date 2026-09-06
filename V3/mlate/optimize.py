"""
Scaffold optimisation
=====================

Searches the formulation space for the composition and printing conditions that
maximise WSSQ, using the deployed classifiers as the objective.

Extracted from the Streamlit application deliberately. In the previous version
the objective function lived inside the interface, which meant it could not be
run headlessly, could not be tested, and could not produce a result the
manuscript was able to report. Here the interface calls this module rather than
containing it.

How a classifier becomes a continuous objective
-----------------------------------------------
Printability and Cell Response are ordinal classes, so a classifier returns a
probability vector rather than a number, and an optimiser handed the arg-max
would be climbing a step function with four or five levels. Instead each model's
full probability vector is collapsed to its EXPECTED CLASS VALUE,

    E[class] = sum_k  P(class = k) * label_k

which is continuous, respects the ordinal spacing of the labels, and carries the
model's uncertainty: a formulation the model is unsure about lands between
classes rather than being rounded confidently to one. Those two expected values
are then passed to WSSQ.

What is searched, and what is not
---------------------------------
Biomaterial concentrations, cell density and the printing parameters are
searched within user-supplied ranges. The CELL LINE IS NOT: it is chosen by the
user and held fixed for the run. This matters more than it looks. WSSQ falls
back to printability alone when Cell Response is 1, and an acellular
formulation therefore scores on printability only - so an optimiser allowed to
vary the cell line could raise its objective simply by removing the cells.
Fixing the line makes the acellular case a mode the user selects, not a
degenerate optimum the search can drift into.

Extrapolation is reported, not prevented
----------------------------------------
Nothing stops a user setting a range beyond anything in the corpus, and nothing
here silently clips it. `distance_report` instead states how far the returned
optimum sits outside the observed range of each variable and how close it is to
the nearest real formulation, so a candidate that the models are extrapolating
to is visible as such rather than presented with the same confidence as one
inside the training distribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from mlate import config as cfg
from mlate import wssq as wssq_mod

# Optuna's own logging is noisy inside a UI and uninformative in a script.
try:  # pragma: no cover - import guard only
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except Exception:  # pragma: no cover
    optuna = None


@dataclass
class Variable:
    """One searchable input, with the range and grid the user specified."""
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
    """Everything the optimiser may vary, plus the cell line it may not."""
    cell_line: str
    biomaterials: list[Variable] = field(default_factory=list)
    printing: list[Variable] = field(default_factory=list)
    cell_density: Variable | None = None

    @property
    def is_acellular(self) -> bool:
        return self.cell_line == cfg.ACELLULAR_TOKEN


def class_distribution(estimator, X: np.ndarray, classes: np.ndarray
                       ) -> tuple[np.ndarray, list[dict[float, float]]]:
    """
    Expected class values and the distributions they were taken over.

    Both come from a single `predict_proba` over the whole batch, because the
    two are wanted together and a second call would double the cost of a
    search. They carry different information and both are used. An expected
    value of 2.5 is what a confident prediction split evenly between classes 2
    and 3 produces, and also what a diffuse prediction over 1 and 4 produces;
    the optimiser needs only the number, while the protocol-generation prompt
    is given the distribution so that it can state which case it describes.

    Falls back to the predicted label when a model exposes no probabilities;
    five of the conventional classifiers do not, and for those the objective is
    a step function rather than a smooth one and the distributions are empty.
    """
    labels = np.asarray(classes, dtype=float)
    if not hasattr(estimator, "predict_proba"):
        idx = np.asarray(estimator.predict(X)).astype(int)
        return labels[idx], [{} for _ in idx]
    proba = np.asarray(estimator.predict_proba(X), dtype=float)
    return proba @ labels, [{float(l): float(pr) for l, pr in zip(labels, row)}
                            for row in proba]


def expected_class_value(estimator, X: np.ndarray,
                         classes: np.ndarray) -> np.ndarray:
    """
    Probability-weighted mean of the class labels.

    Falls back to the predicted label when a model exposes no probabilities;
    five of the conventional classifiers do not, and for those the objective is
    a step function rather than a smooth one. The application ranks models by
    benchmarked performance and the probability-capable models occupy the top of
    both leaderboards, so this path is a safeguard rather than the normal case.
    """
    labels = np.asarray(classes, dtype=float)
    if hasattr(estimator, "predict_proba"):
        return np.asarray(estimator.predict_proba(X), dtype=float) @ labels
    idx = np.asarray(estimator.predict(X)).astype(int)
    return labels[idx]


# Candidates scored per pass through the models. Thirty-two because the cost
# of an in-context model is the pass over its context and is flat in the batch
# size up to at least sixty-four, while a larger batch buys the sampler fewer
# updates for no further saving.
DEFAULT_BATCH_SIZE = 32


def _as_scorer(model):
    """
    Accept either a pickled classifier bundle or a `mlate.serving.Servable`.

    The three model families no longer share a storage format - a deep network
    is a state dictionary and a foundation model is a stored context - so the
    application passes the uniform wrapper from `mlate.serving`. A plain bundle
    is still accepted, because the analysis stages load their own artefacts
    directly and have no reason to go through the serving layer.
    """
    if isinstance(model, dict):
        return {"estimator": model["estimator"],
                "classes": np.asarray(model["classes"], dtype=float)}
    return {"estimator": model, "classes": np.asarray(model.classes,
                                                      dtype=float)}


class Objective:
    """
    WSSQ of one candidate formulation, as the optimiser sees it.

    Holds the preprocessor and both models so they are loaded once rather than
    per trial, and records every evaluated candidate so the search can be
    inspected afterwards instead of yielding only its arg-max.
    """

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
        """One row per candidate, in the raw feature layout the preprocessor
        expects."""
        batch = [values] if isinstance(values, dict) else list(values)
        rows = []
        for v in batch:
            row = {c: 0.0 for c in self.feature_columns}
            row.update({k: x for k, x in v.items() if k in row})
            row[cfg.CELL_COLS[0]] = self.space.cell_line
            rows.append(row)
        return pd.DataFrame(rows)[self.feature_columns]

    def evaluate_batch(self, batch: list[dict]) -> list[dict]:
        """
        Score many candidates in one pass through each model.

        This is the form the search uses, and the reason a foundation model can
        be offered at all. Their cost is dominated by the pass over the 2,646
        context rows, which is paid once per call however many candidates ride
        in it: on this corpus a batch of thirty-two costs what a single row
        costs. Scoring candidates one at a time, as the previous release did,
        charged that pass once per candidate and made an in-context model look
        two orders of magnitude more expensive than it is. The conventional
        classifiers gain from it too, for the ordinary reason that per-call
        Python overhead is then divided across the batch.
        """
        X = np.asarray(self.pre.transform(self.candidate_frame(batch)),
                       dtype=float)

        exp_p, proba_p = class_distribution(
            self.print_model["estimator"], X, self.print_model["classes"])

        # An acellular formulation has no cell response to predict. Asking the
        # model anyway would feed the optimiser a number with no referent; the
        # rating scale defines class 1 as "not applicable", so that is what is
        # used and WSSQ takes its printability-only branch.
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
        """Score one explicit formulation. Used to report the winner."""
        return self.evaluate_batch([values])[0]

    def suggest(self, trial) -> dict:
        """One candidate drawn from the search space, as parameter values."""
        values = {v.name: v.suggest(trial)
                  for v in self.space.biomaterials + self.space.printing}
        if self.space.cell_density is not None and not self.space.is_acellular:
            values[cfg.CELL_COLS[1]] = self.space.cell_density.suggest(trial)
        else:
            values[cfg.CELL_COLS[1]] = 0.0
        return values

    def record(self, values: dict, out: dict) -> None:
        """Append one evaluated candidate to the search history."""
        # history is exported to a spreadsheet; the two probability dictionaries
        # do not belong in it.
        self.history.append({**values, **{k: v for k, v in out.items()
                                          if not k.endswith("_proba")}})

    def __call__(self, trial) -> float:
        """Sequential form, kept for a caller that drives Optuna directly."""
        values = self.suggest(trial)
        out = self.evaluate(values)
        self.record(values, out)
        return out["wssq"]


def optimise(objective: Objective, n_trials: int = 100,
             seed: int = cfg.RANDOM_STATE, progress=None,
             batch_size: int = DEFAULT_BATCH_SIZE):
    """
    Run the search and return (best_values, best_score, study).

    Tree-structured Parzen estimation, which is the Bayesian method used in the
    published version: it models P(parameters | score) and samples where good
    scores are likely, rather than sampling the space uniformly. `multivariate`
    and `group` let it model interactions between parameters instead of treating
    each in isolation, which matters here because printability depends on
    combinations - a concentration that prints well at one nozzle diameter does
    not at another.

    Candidates are drawn and scored in batches through Optuna's ask-and-tell
    interface rather than one at a time. The trade is small and one-sided: the
    candidates within a batch are proposed from the same posterior, so the
    sampler sees `batch_size` fewer updates over a run, against a reduction in
    scoring cost of one to two orders of magnitude. It is what makes the
    in-context foundation models usable in a search at all - see
    `Objective.evaluate_batch` - and it leaves the conventional classifiers
    faster as well. Set `batch_size=1` to recover strictly sequential
    behaviour.
    """
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
    """
    How far outside the observed data the proposed formulation sits.

    Two numbers, because they answer different questions. `out_of_range` names
    the variables whose value exceeds anything recorded in the corpus - a
    concentration no one has published. `nearest_neighbour_distance` is the
    scaled Euclidean distance to the closest real formulation, which catches the
    subtler case where every individual value is ordinary but the combination
    has never been attempted.
    """
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
