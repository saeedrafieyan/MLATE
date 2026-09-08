from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from mlate import config as cfg

FALLBACK = {"low": 0.2, "medium": 2.0, "high": 6.0}

INNER_TRAIN_FRACTION = 0.9


@dataclass(frozen=True)
class Unit:
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
        out = self.key
        for ch in " ()+->/\\":
            out = out.replace(ch, "_")
        return "_".join(part for part in out.split("_") if part)


def load_fit_costs(path=None) -> dict[tuple[str, str], float]:
    path = path or (cfg.step_dir("04_machine_learning", "tables")
                    / "fit_costs.xlsx")
    if not path.exists():
        return {}
    df = pd.read_excel(path)
    return {(str(r.task), str(r.model)): float(r.seconds)
            for r in df.itertuples() if float(r.seconds) > 0}


def fit_seconds(model: str, task: str, costs: dict) -> float:
    if (task, model) in costs:
        return costs[(task, model)]
    same_model = [v for (t, m), v in costs.items() if m == model]
    if same_model:
        return sum(same_model) / len(same_model)
    from mlate import models as zoo
    tag = zoo.REGISTRY[model].cost if model in zoo.REGISTRY else "medium"
    return FALLBACK.get(tag, FALLBACK["medium"])


def estimate_unit(model: str, task: str, costs: dict, n_trials: int,
                  inner_folds: int) -> float:
    from mlate import search_spaces as ss

    per_fit = fit_seconds(model, task, costs)
    if model in ss.BASELINES:
        return per_fit
    if model in ss.META:
        return per_fit
    return per_fit * (n_trials * inner_folds * INNER_TRAIN_FRACTION + 1.0)


def build_units(tasks, protocols, models, n_folds_by_cell: dict,
                n_trials: int, inner_folds: int,
                costs: dict | None = None) -> list[Unit]:
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
    if not units:
        return {"n_units": 0, "core_seconds": 0.0, "makespan_seconds": 0.0,
                "utilisation": 0.0, "workers": workers}
    end = [0.0] * workers
    for u in units:
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
