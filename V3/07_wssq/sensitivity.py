"""
WSSQ sensitivity analysis
=========================

    python 07_wssq/sensitivity.py
    python 07_wssq/sensitivity.py --labels-only   # no deployment bundle needed

Answers two referee comments that ask different questions about the same
formula, and are worth keeping apart because only one of the constants involved
is actually fixed.

  R2-4  "On page 9, Equation 6 uses (HWM+WMC)/2. Why are HWM and WMC weighted
        equally? ... justify this choice or provide a sensitivity analysis
        using different weights."

        This is the `blend` parameter, and it is the only number in WSSQ that a
        user cannot change - it is not exposed in the application, so every
        score anyone has ever computed used 0.5. It therefore has to be
        defended on its own terms. Two things are reported: the structural
        argument (blend selects severity inside a bounded family of conjunctive
        means, verified numerically rather than asserted), and how far the
        ranking of candidates moves across the whole blend range.

  R1-3  "include a sensitivity analysis showing how rankings of candidate
        formulations change under different printability/cell-response
        weightings."

        These weights ARE user-controlled: the application exposes cell weight
        as a slider from 0 to 100% in steps of 5, with print weight taking the
        remainder. So this is not a hidden design choice needing a defence - it
        is a control surface, and the useful analysis is a map of how much a
        user's own choice moves the answer. Reported over the slider's real
        positions rather than over a continuum nobody can select.

What "ranking of candidate formulations" means here
---------------------------------------------------
Every formulation in the corpus is a candidate, ranked by WSSQ. Sensitivity is
measured against the shipped default (cell weight 0.7, blend 0.5) with
Kendall's tau and Spearman's rho over the full ranking, and with overlap of the
top 10, 50 and 100 - because WSSQ exists to drive an optimiser toward its
maximum, so a reordering deep in the tail matters far less than a change in
which candidates finish on top.

Scored on what the optimiser actually sees
------------------------------------------
The primary arm uses the EXPECTED CLASS VALUE, sum_k P(class k) * label_k -
`expected_class_value` in the deployed application, and the reason its
predictions carry model confidence rather than a hard class. This matters more
than it sounds. Printability has four levels and Cell Response five, so scoring
WSSQ on classes puts every candidate on a lattice of twenty possible positions;
on that lattice the harmonic and geometric means agree, and `blend` looks
strictly irrelevant. The optimiser never traverses that lattice - it climbs a
continuous surface of thousands of distinct values, where the two means do not
agree. An analysis run on labels alone would report that `blend` cannot matter,
which is true of the labels and false of the tool. Both discrete arms are
therefore kept as controls, and the difference between them and the primary arm
is itself part of the answer.

The same distinction governs the boundary rules. Both bypass the weighted means
entirely, so a candidate with Printability 0, or Cell Response 1, scores the
same under every weighting. On observed labels that covers 61% of the corpus;
on expected values it covers 0.2%, because an expected class value is almost
never exactly 1. Headline correlations are computed on the responsive subset in
every arm so the two are comparable.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from mlate import config as cfg
from mlate import wssq
from mlate.dataset import load_dataset, target_frame

warnings.filterwarnings("ignore")

TABLES = cfg.step_dir("07_wssq", "tables")
BLENDS = tuple(round(b, 2) for b in np.arange(0.0, 1.01, 0.05))
TOP_K = (10, 50, 100)


def rank_agreement(base: np.ndarray, other: np.ndarray) -> dict:
    """Agreement between two scorings of the same candidates."""
    out = {}
    if len(base) > 1 and np.ptp(base) > 0 and np.ptp(other) > 0:
        out["kendall_tau"] = float(kendalltau(base, other).statistic)
        out["spearman_rho"] = float(spearmanr(base, other).statistic)
    else:
        out["kendall_tau"] = out["spearman_rho"] = np.nan
    order_b = np.argsort(-base, kind="stable")
    order_o = np.argsort(-other, kind="stable")
    for k in TOP_K:
        k = min(k, len(base))
        out[f"top{k}_overlap"] = len(
            set(order_b[:k].tolist()) & set(order_o[:k].tolist())) / k
    out["max_abs_score_shift"] = float(np.max(np.abs(base - other)))
    return out


def structural_check(p: np.ndarray, c: np.ndarray) -> pd.DataFrame:
    """
    Verify the inequality the justification of `blend` rests on.

    WSSQ mixes a weighted harmonic and a weighted geometric mean. If
    HWM <= WMC <= arithmetic holds on this data, then `blend` interpolates
    inside a bounded interval whose upper end is still stricter than the
    compensatory arithmetic mean the metric deliberately avoids - which makes
    0.5 a midpoint of a principled range rather than an arbitrary constant.
    """
    rows = []
    for wc in (0.1, 0.3, 0.5, 0.7, 0.9):
        comp = wssq.components(p, c, 1 - wc, wc)
        h, m, a = comp["harmonic"], comp["multiplicative"], comp["arithmetic"]
        rows.append({
            "cell_weight": wc,
            "HWM <= WMC": bool(np.all(h <= m + 1e-9)),
            "WMC <= AM": bool(np.all(m <= a + 1e-9)),
            "mean_HWM": float(h.mean()), "mean_WMC": float(m.mean()),
            "mean_AM": float(a.mean()),
            "mean_gap_WMC_minus_HWM": float((m - h).mean()),
            "max_gap_WMC_minus_HWM": float((m - h).max()),
        })
    return pd.DataFrame(rows)


def lattice_check() -> pd.DataFrame:
    """
    Does `blend` reorder candidates on the label lattice, ignoring how often
    each combination actually occurs?

    This is the control for the headline blend result, and it is reported
    because it refuses to let that result be overstated. Printability has four
    levels and Cell Response five, so WSSQ can take at most twenty distinct
    values; it is tempting to conclude that the harmonic and geometric means
    must agree on so coarse a lattice and that `blend` is therefore
    order-irrelevant by construction.

    They do not agree. Counting each of the twenty combinations once, the two
    means induce different orderings at many weightings. The invariance
    measured on the corpus is thus a property of how this dataset is
    DISTRIBUTED over the lattice - the combinations where the means disagree
    are rare or absent here - and not a theorem about the formula. Said
    plainly: the choice of 0.5 is immaterial for these candidates, and would
    not automatically remain so for a differently distributed corpus.
    """
    P, C = map(np.ravel, np.meshgrid(np.arange(0, 4.), np.arange(1, 6.)))
    resp = ~wssq.is_weight_invariant(P, C)
    rows = []
    for wc in np.arange(0, 1.001, 0.01):
        comp = wssq.components(P[resp], C[resp], 1 - wc, wc)
        tau = kendalltau(comp["harmonic"], comp["multiplicative"]).statistic
        rows.append({"cell_weight": round(float(wc), 2),
                     "kendall_tau_HWM_vs_WMC": float(tau)})
    return pd.DataFrame(rows)


def analyse(p: np.ndarray, c: np.ndarray, label: str) -> dict:
    invariant = wssq.is_weight_invariant(p, c)
    resp = ~invariant
    n, n_inv = len(p), int(invariant.sum())

    base = wssq.compute_wssq(p, c, 1 - wssq.DEFAULT_CELL_WEIGHT,
                             wssq.DEFAULT_CELL_WEIGHT, wssq.DEFAULT_BLEND)

    # ---- R2-4: blend, the fixed constant -------------------------------
    blend_rows = []
    for b in BLENDS:
        s = wssq.compute_wssq(p, c, 1 - wssq.DEFAULT_CELL_WEIGHT,
                              wssq.DEFAULT_CELL_WEIGHT, b)
        blend_rows.append({"blend": b, "scope": "responsive candidates",
                           "n": int(resp.sum()), "mean_wssq": float(s[resp].mean()),
                           **rank_agreement(base[resp], s[resp])})

    # ---- R1-3: the user-facing weight slider ---------------------------
    weight_rows = []
    for wc in wssq.SLIDER_CELL_WEIGHTS:
        s = wssq.compute_wssq(p, c, 1 - wc, wc, wssq.DEFAULT_BLEND)
        weight_rows.append({"cell_weight": wc, "print_weight": round(1 - wc, 2),
                            "n": int(resp.sum()), "mean_wssq": float(s[resp].mean()),
                            **rank_agreement(base[resp], s[resp])})

    # ---- joint grid: does blend matter more or less than the slider? ---
    grid = []
    for wc in wssq.SLIDER_CELL_WEIGHTS:
        for b in BLENDS:
            s = wssq.compute_wssq(p, c, 1 - wc, wc, b)
            a = rank_agreement(base[resp], s[resp])
            grid.append({"cell_weight": wc, "blend": b,
                         "kendall_tau": a["kendall_tau"],
                         "top10_overlap": a["top10_overlap"],
                         "mean_wssq": float(s[resp].mean())})

    # How much does blend move things AT A FIXED weight - i.e. the part of the
    # variation a user cannot control?
    blend_only = []
    for wc in wssq.SLIDER_CELL_WEIGHTS:
        ref = wssq.compute_wssq(p, c, 1 - wc, wc, wssq.DEFAULT_BLEND)
        taus, tops = [], []
        for b in BLENDS:
            s = wssq.compute_wssq(p, c, 1 - wc, wc, b)
            a = rank_agreement(ref[resp], s[resp])
            taus.append(a["kendall_tau"]); tops.append(a["top10_overlap"])
        blend_only.append({"cell_weight": wc, "min_kendall_tau": min(taus),
                           "min_top10_overlap": min(tops)})

    return {
        "label": label,
        "n_candidates": n, "n_weight_invariant": n_inv,
        "pct_weight_invariant": 100 * n_inv / n,
        "structural": structural_check(p[resp], c[resp]),
        "blend": pd.DataFrame(blend_rows),
        "weights": pd.DataFrame(weight_rows),
        "grid": pd.DataFrame(grid),
        "blend_only": pd.DataFrame(blend_only),
        "lattice": lattice_check(),
    }


MODEL_FILES = (("printability", "printability__bagged_trees.pkl"),
               ("cell_response", "cell_response__stacking_rf_xgb_lr_lr.pkl"))


def _model_outputs(mode: str):
    """
    Targets as the application computes them.

    mode "expected"  the probability-weighted expected class value,
                     sum_k P(class k) * label_k. This is what the deployed
                     optimiser maximises - `expected_class_value` in the
                     Streamlit app - and it is CONTINUOUS.
    mode "argmax"    the predicted class. Discrete.

    The distinction is the whole reason this analysis has three arms. Scoring
    WSSQ on integers puts every candidate on a 4x5 lattice of twenty possible
    positions, where the harmonic and geometric means happen to agree and the
    blend parameter looks irrelevant. The optimiser never sees that lattice: it
    traverses a continuous surface with thousands of distinct values, and the
    two means do not agree there. A sensitivity analysis run on labels alone
    would report that `blend` cannot matter, which is true of the labels and
    false of the tool.
    """
    import joblib

    from mlate import artifacts
    pre, _, _, _ = artifacts.load_release()
    df, columns = load_dataset()
    out = {}
    for task, fname in MODEL_FILES:
        path = cfg.MODEL_DIR / "classifiers" / "random" / fname
        if not path.exists():
            print(f"  ! missing {path.name}; "
                  f"run 06_webapp/export_deployment.py first")
            return None
        sub, _ = target_frame(df, task)
        X = np.asarray(pre.transform(sub[columns.predictors]), dtype=float)
        obj = joblib.load(path)
        labels = np.asarray(obj["classes"], dtype=float)
        est = obj["estimator"]
        if mode == "expected":
            out[task] = est.predict_proba(X) @ labels
        else:
            out[task] = labels[np.asarray(est.predict(X)).astype(int)]
    return out["printability"], out["cell_response"]


def report(res: dict) -> None:
    print("\n" + "=" * 78)
    print(f"{res['label']}  |  {res['n_candidates']:,} candidate formulations")
    print("=" * 78)
    print(f"weight-invariant by construction: {res['n_weight_invariant']:,} "
          f"({res['pct_weight_invariant']:.1f}%) - Printability 0 or "
          f"Cell Response 1")
    print(f"responsive to weighting:          "
          f"{res['n_candidates'] - res['n_weight_invariant']:,}")

    print("\n-- structural basis for `blend` (R2-4) "
          "-------------------------------")
    print(res["structural"].round(3).to_string(index=False))

    print("\n-- R2-4: varying `blend`, the FIXED constant, at default weights")
    b = res["blend"]
    show = b[b.blend.isin([0.0, 0.25, 0.5, 0.75, 1.0])]
    print(show[["blend", "mean_wssq", "kendall_tau", "spearman_rho",
                "top10_overlap", "top50_overlap",
                "max_abs_score_shift"]].round(4).to_string(index=False))
    worst = b.loc[b.kendall_tau.idxmin()]
    print(f"   worst case over the whole blend range: tau={worst.kendall_tau:.4f} "
          f"at blend={worst.blend}, top-10 overlap={worst.top10_overlap:.2f}")

    print("\n-- R1-3: varying the USER-CONTROLLED weight slider")
    w = res["weights"]
    show = w[w.cell_weight.isin([0.0, 0.25, 0.5, 0.7, 0.75, 1.0])]
    print(show[["cell_weight", "print_weight", "mean_wssq", "kendall_tau",
                "spearman_rho", "top10_overlap",
                "top50_overlap"]].round(4).to_string(index=False))
    worst = w.loc[w.kendall_tau.idxmin()]
    print(f"   worst case across the slider: tau={worst.kendall_tau:.4f} at "
          f"cell weight={worst.cell_weight}, "
          f"top-10 overlap={worst.top10_overlap:.2f}")

    lat = res["lattice"]
    dis = lat[lat.kendall_tau_HWM_vs_WMC < 1 - 1e-12]
    print("\n-- control: is that invariance a property of the FORMULA?  No.")
    print(f"   on a uniform label lattice the two means disagree at "
          f"{len(dis)} of {len(lat)} weightings "
          f"(min tau {lat.kendall_tau_HWM_vs_WMC.min():.4f}).")
    print("   the corpus invariance therefore reflects how this dataset is "
          "distributed,")
    print("   not a theorem - a finding about these candidates, not about "
          "WSSQ itself.")

    print("\n-- how much of the movement is the part users CANNOT control?")
    bo = res["blend_only"]
    print(f"   across all 21 slider positions, sweeping blend 0->1 never "
          f"drops Kendall tau below {bo.min_kendall_tau.min():.4f}")
    print(f"   and never drops top-10 overlap below "
          f"{bo.min_top10_overlap.min():.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels-only", action="store_true",
                    help="skip the model arms (no deployment bundle needed)")
    args = ap.parse_args()

    df, _ = load_dataset()
    p = df[cfg.TARGETS[0]].to_numpy(dtype=float)
    c = df[cfg.TARGETS[1]].to_numpy(dtype=float)

    # Primary arm first: the continuous surface the optimiser actually climbs.
    results = []
    if not args.labels_only:
        exp = _model_outputs("expected")
        if exp is not None:
            results.append(analyse(exp[0], exp[1],
                                   "expected class values (what the "
                                   "optimiser maximises)"))
        pred = _model_outputs("argmax")
        if pred is not None:
            results.append(analyse(pred[0], pred[1],
                                   "predicted classes (control: discrete)"))
    results.append(analyse(p, c, "observed labels (control: discrete)"))

    for res in results:
        report(res)

    dest = TABLES / "wssq_sensitivity.xlsx"
    with pd.ExcelWriter(dest) as xl:
        for res in results:
            tag = ("expected" if "expected" in res["label"]
                   else "predicted" if "predicted" in res["label"]
                   else "labels")
            pd.DataFrame([{
                "candidates": res["n_candidates"],
                "weight_invariant": res["n_weight_invariant"],
                "pct_weight_invariant": round(res["pct_weight_invariant"], 2),
            }]).to_excel(xl, sheet_name=f"summary_{tag}", index=False)
            res["structural"].to_excel(xl, sheet_name=f"structural_{tag}",
                                       index=False)
            res["blend"].to_excel(xl, sheet_name=f"blend_{tag}", index=False)
            res["weights"].to_excel(xl, sheet_name=f"weights_{tag}",
                                    index=False)
            res["grid"].to_excel(xl, sheet_name=f"grid_{tag}", index=False)
            res["lattice"].to_excel(xl, sheet_name=f"lattice_{tag}", index=False)
    print(f"\n-> {dest}")


if __name__ == "__main__":
    main()
