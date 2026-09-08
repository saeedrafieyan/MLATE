from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from mlate import config as cfg
from mlate import models as zoo
from mlate import search_spaces as ss

TABLES = cfg.step_dir("04_machine_learning", "tables")
OUT = TABLES / "search_spaces.xlsx"


class RecordingTrial:

    def __init__(self):
        self.rows: list[dict] = []

    def suggest_int(self, name, low, high, step=1, log=False):
        self.rows.append({"parameter": name, "type": "int", "low": low,
                          "high": high, "step": step, "log": log,
                          "choices": ""})
        return low

    def suggest_float(self, name, low, high, step=None, log=False):
        self.rows.append({"parameter": name, "type": "float", "low": low,
                          "high": high, "step": step if step else "",
                          "log": log, "choices": ""})
        return low

    def suggest_categorical(self, name, choices):
        self.rows.append({"parameter": name, "type": "categorical", "low": "",
                          "high": "", "step": "", "log": "",
                          "choices": ", ".join(str(c) for c in choices)})
        return choices[0]


def main() -> None:
    rows = []
    for name in zoo.names():
        spec = zoo.REGISTRY[name]
        if name in ss.BASELINES:
            rows.append({"model": name, "family": spec.family,
                         "handling": "not tuned (baseline)",
                         "parameter": "-", "type": "", "low": "", "high": "",
                         "step": "", "log": "", "choices": "",
                         "pinned": ""})
            continue
        if name in ss.META:
            rows.append({"model": name, "family": spec.family,
                         "handling": "composed from tuned base learners",
                         "parameter": "-", "type": "", "low": "", "high": "",
                         "step": "", "log": "", "choices": "",
                         "pinned": ""})
            continue

        trial = RecordingTrial()
        ss.suggest(name, trial)
        pinned = ss.PINNED.get(name, {})
        for r in trial.rows:
            rows.append({"model": name, "family": spec.family,
                         "handling": "searched", **r,
                         "pinned": ""})
        for k, v in pinned.items():
            rows.append({"model": name, "family": spec.family,
                         "handling": "searched", "parameter": k,
                         "type": "fixed", "low": "", "high": "", "step": "",
                         "log": "", "choices": "", "pinned": str(v)})

    df = pd.DataFrame(rows)
    df.to_excel(OUT, index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 400)
    for model, g in df.groupby("model", sort=False):
        handling = g["handling"].iloc[0]
        if handling != "searched":
            print(f"\n### {model}  [{g['family'].iloc[0]}]  -- {handling}")
            continue
        print(f"\n### {model}  [{g['family'].iloc[0]}]  "
              f"({(g['parameter'] != '-').sum()} parameters)")
        for r in g.itertuples():
            if r.type == "categorical":
                print(f"    {r.parameter:28s} categorical  {{{r.choices}}}")
            elif r.type == "fixed":
                print(f"    {r.parameter:28s} PINNED       {r.pinned}")
            else:
                lg = " log" if r.log else ""
                st = f" step={r.step}" if r.step != "" else ""
                print(f"    {r.parameter:28s} {r.type:11s} "
                      f"[{r.low}, {r.high}]{st}{lg}")

    searched = df[df["handling"] == "searched"]
    n_params = searched[searched["type"] != "fixed"].shape[0]
    print(f"\n{searched['model'].nunique()} models searched, "
          f"{n_params} tunable parameters in total")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
