"""
Cost-aware scheduling for the tuning grid
=========================================

The tuning grid is 33 models x 2 tasks x 2 protocols x 10 outer folds. Those
units are not remotely equal in cost: measured on a real fold, one fit ranges
from 0.004 s (Extra Tree) to 11.6 s (Stacking) - a factor of ~3,000. Handing
that to a worker pool in registry order wastes most of the machine, because the
expensive units are scattered and the pool drains into a long tail where one
worker finishes Stacking while fifty-one sit idle.

Longest-processing-time-first
-----------------------------
Units are dispatched most-expensive-first. LPT is the classic greedy makespan
heuristic and is guaranteed within 4/3 of optimal for identical machines; more
usefully here, it bounds the tail, because the last thing dispatched is by
construction the cheapest thing left rather than possibly the most expensive.
The cheap units backfill behind the expensive ones instead of the reverse.

Where the costs come from
-------------------------
Measured, not guessed - `04_machine_learning/calibrate.py` times one
single-threaded fit of every model on a real training fold and writes
fit_costs.xlsx. The registry's `cost` tag ("low"/"medium"/"high") is only a
coarse fallback for when that table is missing, and it is wrong often enough to
matter: the tag calls SVM (RBF) "high" at 0.81 s and CatBoost "medium" at
5.91 s.

What the estimate does NOT capture
----------------------------------
A tuning trial fits whatever hyper-parameters Optuna suggested, not the
registry's defaults, and the search spaces reach further than the defaults
(RandomForest to 500 trees, GradientBoosting to 400 x depth 20). The estimate is
therefore a lower bound with a roughly constant bias per model, which is all LPT
needs: ordering is preserved under a monotone transform, so the schedule is
unaffected even where the absolute predictions are low. Pruning cuts real cost
further and unevenly. Treat the totals as planning figures, not promises.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from mlate import config as cfg

# Coarse fallback, seconds per fit, used only when fit_costs.xlsx is absent.
# Deliberately pessimistic: over-estimating a cheap model costs a little
# ordering accuracy, under-estimating an expensive one costs a long tail.
FALLBACK = {"low": 0.2, "medium": 2.0, "high": 6.0}

# Inner-CV folds train on (k-1)/k of the outer training partition.
INNER_TRAIN_FRACTION = 0.9


@dataclass(frozen=True)
class Unit:
    """One (model, task, protocol, outer fold) job."""
    task: str
    protocol: str
    model: str
    fold_index: int
    fold_name: str = ""
    est_seconds: float = 0.0

    @property
    def key(self) -> str:
        return f"{self.task}__{self.protocol}__{self.model}__fold{self.fold_index}"

    @property
    def safe_key(self) -> str:
        """
        Filesystem-safe form of `key`.

        Model names carry spaces, parentheses, '+' and '->' ("Stacking
        (RF+XGB+LR -> LR)"), none of which belong in a Windows filename.
        """
        out = self.key
        for ch in " ()+->/\\":
            out = out.replace(ch, "_")
        return "_".join(part for part in out.split("_") if part)


def load_fit_costs(path=None) -> dict[tuple[str, str], float]:
    """
    Measured seconds per single-threaded fit, keyed by (task, model).

    Returns an empty mapping when the table has not been generated yet; callers
    fall back to the registry's cost tag.
    """
    path = path or (cfg.step_dir("04_machine_learning", "tables")
                    / "fit_costs.xlsx")
    if not path.exists():
        return {}
    df = pd.read_excel(path)
    return {(str(r.task), str(r.model)): float(r.seconds)
            for r in df.itertuples() if float(r.seconds) > 0}


def fit_seconds(model: str, task: str, costs: dict) -> float:
    """Best available estimate of one fit, in seconds."""
    if (task, model) in costs:
        return costs[(task, model)]
    # Any task's measurement beats the cost tag.
    same_model = [v for (t, m), v in costs.items() if m == model]
    if same_model:
        return sum(same_model) / len(same_model)
    from mlate import models as zoo
    tag = zoo.REGISTRY[model].cost if model in zoo.REGISTRY else "medium"
    return FALLBACK.get(tag, FALLBACK["medium"])


def estimate_unit(model: str, task: str, costs: dict, n_trials: int,
                  inner_folds: int) -> float:
    """
    Seconds for one tuning unit: every trial fits once per inner fold, then the
    winning configuration is refitted once on the whole outer training set.
    """
    from mlate import search_spaces as ss

    per_fit = fit_seconds(model, task, costs)
    if model in ss.BASELINES:
        return per_fit                     # fitted once, never searched
    if model in ss.META:
        return per_fit                     # composed from tuned bases, one fit
    return per_fit * (n_trials * inner_folds * INNER_TRAIN_FRACTION + 1.0)


def build_units(tasks, protocols, models, n_folds_by_cell: dict,
                n_trials: int, inner_folds: int,
                costs: dict | None = None) -> list[Unit]:
    """Every unit in the grid, ordered longest-processing-time-first."""
    costs = load_fit_costs() if costs is None else costs
    units = [
        Unit(task=task, protocol=protocol, model=model, fold_index=k,
             est_seconds=estimate_unit(model, task, costs, n_trials,
                                       inner_folds))
        for task in tasks
        for protocol in protocols
        for model in models
        for k in range(n_folds_by_cell[(task, protocol)])
    ]
    return sorted(units, key=lambda u: u.est_seconds, reverse=True)


def makespan(units: list[Unit], workers: int) -> dict:
    """
    Simulate LPT onto `workers` identical machines.

    Reports the predicted wall-clock and how much of the pool the schedule
    actually keeps busy, which is the number worth watching: a high total with
    low utilisation means the grid is tail-bound and wants smaller units, not
    more cores.
    """
    if not units:
        return {"n_units": 0, "core_seconds": 0.0, "makespan_seconds": 0.0,
                "utilisation": 0.0, "workers": workers}
    end = [0.0] * workers
    for u in units:                        # already sorted longest-first
        i = min(range(workers), key=lambda j: end[j])
        end[i] += u.est_seconds
    total = sum(u.est_seconds for u in units)
    span = max(end)
    return {
        "n_units": len(units),
        "core_seconds": total,
        "core_hours": total / 3600.0,
        "makespan_seconds": span,
        "makespan_hours": span / 3600.0,
        "utilisation": total / (span * workers) if span else 0.0,
        "workers": workers,
    }


def plan_table(units: list[Unit]) -> pd.DataFrame:
    """Per-model cost roll-up, for the run log and the supplement."""
    df = pd.DataFrame([{"model": u.model, "task": u.task,
                        "protocol": u.protocol, "est_seconds": u.est_seconds}
                       for u in units])
    out = (df.groupby("model")
             .agg(units=("est_seconds", "size"),
                  est_seconds_each=("est_seconds", "mean"),
                  est_core_hours=("est_seconds", lambda s: s.sum() / 3600.0))
             .sort_values("est_core_hours", ascending=False)
             .reset_index())
    out["share_of_total"] = out["est_core_hours"] / out["est_core_hours"].sum()
    return out
