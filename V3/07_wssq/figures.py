"""
WSSQ sensitivity figure
=======================

    python 07_wssq/figures.py

One figure, two panels, sharing a y-axis. The shared axis is the point: the
entire argument to both referees is an ASYMMETRY between the two parameters,
and putting them on separate scales would hide exactly what the figure exists
to show.

  (A)  `blend` - the constant a user cannot change. Flat.
  (B)  the cell/print weight slider - the control a user sets. Steep at the
       extremes.

Read together they say: the parameter we fixed has second-order influence and
never changes which candidates finish on top; the parameter that does steer the
outcome is exposed to the user rather than chosen by us. That is the defence of
equation 6, and it is a stronger one than any argument about the algebra.

Both panels plot Kendall's tau against the shipped default and top-10 overlap,
on the expected-class-value arm - the continuous surface the optimiser climbs,
not the twenty-point label lattice.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from mlate import config as cfg
from mlate import style as ms

warnings.filterwarnings("ignore")

TABLES = cfg.step_dir("07_wssq", "tables")
BOOK = TABLES / "wssq_sensitivity.xlsx"


def _panel(ax, x, tau, top10, xlabel, default_x, title):
    ax.plot(x, tau, "-o", color=ms.NAVY, markersize=3.4, linewidth=1.6,
            label=r"Kendall $\tau$ (full ranking)", zorder=3)
    ax.plot(x, top10, "-s", color=ms.RUST, markersize=3.2, linewidth=1.4,
            label="top-10 overlap", zorder=3)
    ax.axvline(default_x, color=ms.MUTED, linewidth=1.0, linestyle=(0, (4, 3)),
               zorder=1)
    ax.annotate("shipped\ndefault", xy=(default_x, 0.045),
                xytext=(default_x, 0.045), ha="center", va="bottom",
                fontsize=6.4, color=ms.MUTED,
                path_effects=ms.halo(1.6))
    ax.set_xlabel(xlabel)
    ax.set_ylim(0.0, 1.045)
    ax.set_title(title, loc="left", fontsize=8.2, pad=6)
    ms.grid_axis(ax, "y")
    ms.despine(ax)


def main() -> None:
    if not BOOK.exists():
        raise SystemExit(f"{BOOK.name} not found - run 07_wssq/sensitivity.py")
    blend = pd.read_excel(BOOK, sheet_name="blend_expected")
    weights = pd.read_excel(BOOK, sheet_name="weights_expected")

    ms.apply()
    fig, axes = ms.figure(width="double", height=3.1, ncols=2,
                          sharey=True)

    _panel(axes[0], blend["blend"], blend["kendall_tau"],
           blend["top10_overlap"],
           "blend  (weight on the harmonic mean)", 0.5,
           r"$\bf{A}$   Fixed in the tool: the HWM/WMC blend")
    _panel(axes[1], weights["cell_weight"], weights["kendall_tau"],
           weights["top10_overlap"],
           "cell-response weight  (print weight = 1 - this)", 0.7,
           r"$\bf{B}$   Set by the user: the weighting slider")

    axes[0].set_ylabel("agreement with the shipped default")
    axes[1].legend(loc="lower center", frameon=False, fontsize=6.8,
                   handlelength=1.6, borderaxespad=0.6)

    fig.suptitle("Sensitivity of WSSQ candidate rankings to its two "
                 "parameters", x=0.008, ha="left", fontsize=9.4, y=0.995)
    fig.text(0.008, 0.902,
             "scored on expected class values, the continuous surface the "
             "optimiser maximises; n = 2,640 responsive candidates",
             ha="left", fontsize=6.9, color=ms.MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.855))

    paths = ms.save(fig, "figS14_wssq_sensitivity", step="07_wssq")
    print("  " + ", ".join(p.name for p in paths))
    for name, d, col in (("blend", blend, "blend"),
                         ("weight", weights, "cell_weight")):
        w = d.loc[d["kendall_tau"].idxmin()]
        print(f"  {name:6s} worst: tau={w.kendall_tau:.4f} at {col}="
              f"{w[col]}, top-10 overlap={w.top10_overlap:.2f}")


if __name__ == "__main__":
    main()
