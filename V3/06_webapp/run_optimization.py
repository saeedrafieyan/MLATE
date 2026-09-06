"""
Headless scaffold optimisation, and optionally a generated protocol
===================================================================

    python 06_webapp/run_optimization.py --cell-line bMSCs --trials 200
    python 06_webapp/run_optimization.py --cell-line NoCellCultured
    python 06_webapp/run_optimization.py --cell-line HepG2 --protocol

Exists so that the optimisation described in the manuscript can be run,
reported and reproduced without a browser. In the previous release the
objective lived inside the Streamlit application, which meant no result from it
could appear in a paper except as a screenshot.

Writes the search history, the winning formulation, its distance from the
observed data, and any generated protocol to results/06_webapp/.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from mlate import artifacts
from mlate import config as cfg
from mlate import optimize as opt
from mlate import protocol as proto
from mlate import serving
from mlate.dataset import load_dataset

OUT = cfg.step_dir("06_webapp", "tables")

# A worked example, chosen to exercise the search rather than to recommend a
# formulation: an alginate-gelatin system with ionic crosslinking, which is the
# most common family in the corpus and therefore the best supported by data.
DEMO_SPACE = {
    "biomaterials": [("Alginate (%w/v)", 1.0, 8.0, 0.5),
                     ("Gelatin (%w/v)", 2.0, 12.0, 0.5),
                     ("CaCl2 (mM)", 50.0, 200.0, 10.0)],
    "printing": [("Physical Crosslinking Duration (s)", 0.0, 600.0, 10.0),
                 ("Photo Crosslinking Duration (s)", 0.0, 120.0, 5.0),
                 ("Extrusion Pressure (kPa)", 20.0, 200.0, 5.0),
                 ("Nozzle Movement Speed (mm/s)", 1.0, 20.0, 0.5),
                 ("Nozzle Diameter (µm)", 100.0, 600.0, 10.0),
                 ("Syringe Temperature (°C)", 18.0, 40.0, 0.5),
                 ("Substrate Temperature (°C)", 4.0, 40.0, 0.5)],
    "density": (1.0, 10.0, 0.5),
}


def pick_model(task: str, name: str | None):
    """
    Best benchmarked random-split model for a target, or a named one.

    Searches all three families through the serving layer, so `--llm`-style
    naming reaches TabICL and the deep networks as well as the pickled
    classifiers.
    """
    stubs = serving.discover(cfg.MODEL_DIR, task, "random")
    if not stubs:
        raise SystemExit(f"no deployable models for {task}")
    chosen = stubs[0] if not name else next(
        (s for s in stubs if s.name == name), None)
    if chosen is None:
        raise SystemExit(f"no model named {name!r} for {task}; available: "
                         + ", ".join(sorted(s.name for s in stubs)))
    print(f"  {task:14s} {chosen.name}  ({chosen.family}, "
          f"F1 {chosen.weighted_f1:.3f})")
    return chosen.open()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell-line", default="bMSCs")
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--cell-weight", type=float, default=0.7)
    ap.add_argument("--printability-model", default=None)
    ap.add_argument("--cell-response-model", default=None)
    ap.add_argument("--protocol", action="store_true",
                    help="also generate a fabrication protocol (needs a key)")
    ap.add_argument("--llm", default=proto.DEFAULT_MODEL)
    ap.add_argument("--batch-size", type=int,
                default=opt.DEFAULT_BATCH_SIZE,
                help="candidates scored per pass; 1 is strictly sequential")
    ap.add_argument("--tag", default="demo")
    args = ap.parse_args()

    pre, feature_columns, _, _ = artifacts.load_release()
    df, columns = load_dataset()
    print("models")
    pm = pick_model("printability", args.printability_model)
    cm = pick_model("cell_response", args.cell_response_model)

    acellular = args.cell_line == cfg.ACELLULAR_TOKEN
    space = opt.SearchSpace(
        cell_line=args.cell_line,
        biomaterials=[opt.Variable(*v) for v in DEMO_SPACE["biomaterials"]],
        printing=[opt.Variable(*v) for v in DEMO_SPACE["printing"]],
        cell_density=(None if acellular
                      else opt.Variable(cfg.CELL_COLS[1],
                                        *DEMO_SPACE["density"])))

    objective = opt.Objective(space, pre, feature_columns, pm, cm,
                              print_weight=1 - args.cell_weight,
                              cell_weight=args.cell_weight)

    print(f"\nsearching {len(space.biomaterials) + len(space.printing) + (0 if acellular else 1)} "
          f"variables over {args.trials} trials, cell line {args.cell_line}")
    best, score, study = opt.optimise(objective, n_trials=args.trials,
                                  batch_size=args.batch_size)
    detail = objective.evaluate(best)
    dist = opt.distance_report(best, df)

    print(f"\nbest WSSQ {score:.2f}%")
    print(f"  expected printability   {detail['expected_printability']:.3f}")
    if not acellular:
        print(f"  expected cell response  {detail['expected_cell_response']:.3f}")
    print(f"  parameters outside the observed range: {dist['n_out_of_range']}")
    print(f"  distance to nearest real formulation : "
          f"{dist['nearest_neighbour_distance']:.3f}")
    print("\noptimised formulation")
    for k, v in sorted(best.items()):
        print(f"  {k:44s} {v:>10.4g}")

    history = pd.DataFrame(objective.history)
    stem = f"optimisation_{args.tag}_{args.cell_line}"
    history.to_excel(OUT / f"{stem}_history.xlsx", index=False)

    record = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cell_line": args.cell_line, "n_trials": args.trials,
        "cell_weight": args.cell_weight, "batch_size": args.batch_size,
        "printability_model": pm.name, "cell_response_model": cm.name,
        "best_wssq": score,
        **{k: v for k, v in detail.items() if not k.endswith("_proba")},
        "printability_proba": detail.get("printability_proba"),
        "cell_response_proba": detail.get("cell_response_proba"),
        "best_params": best, "distance_report": dist,
    }

    if args.protocol:
        key = proto.api_key()
        if not key:
            print("\n! no OPENROUTER_API_KEY; skipping protocol generation")
        else:
            nb = proto.nearest_formulations(best, df, columns, n=3)
            form = proto.Formulation(
                cell_line=args.cell_line,
                biomaterials={k: v for k, v in best.items()
                              if k in set(columns.biomaterials)},
                printing={k: v for k, v in best.items()
                          if k in set(columns.print_params)},
                cell_density=best.get(cfg.CELL_COLS[1]),
                expected_printability=detail["expected_printability"],
                expected_cell_response=detail["expected_cell_response"],
                printability_proba=detail.get("printability_proba"),
                cell_response_proba=detail.get("cell_response_proba"),
                wssq=score, neighbours=nb, extrapolation=dist)
            print(f"\ngenerating protocol with {args.llm} …\n", flush=True)
            out = proto.generate(
                form, key=key, model=args.llm, n_records=len(df),
                on_chunk=lambda piece: print(piece, end="", flush=True),
                on_retry=lambda n, why: print(f"\n[retry {n}: {why}]\n",
                                              flush=True))
            print()
            (OUT.parent / f"{stem}_protocol.md").write_text(
                out["protocol"], encoding="utf-8")
            record["protocol"] = {k: v for k, v in out.items()
                                  if k != "protocol"}
            print(f"  {out['completion_tokens']} completion tokens, "
                  f"prompt {out['prompt_sha256'][:12]}")

    (OUT / f"{stem}.json").write_text(json.dumps(record, indent=2, default=str),
                                      encoding="utf-8")
    print(f"\n-> {OUT / f'{stem}.json'}")


if __name__ == "__main__":
    main()
