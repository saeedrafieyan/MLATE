from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from mlate import config as cfg
from mlate import style as ms

TABLES = cfg.step_dir("02_preprocessing", "tables")

SHORT = {
    "Physical Crosslinking Duration (s)": "Physical crosslink (s)",
    "Photo Crosslinking Duration (s)": "Photo crosslink (s)",
    "Extrusion Pressure (kPa)": "Extrusion pressure (kPa)",
    "Nozzle Movement Speed (mm/s)": "Nozzle speed (mm/s)",
    "Nozzle Diameter (µm)": "Nozzle diameter (µm)",
    "Syringe Temperature (°C)": "Syringe temp (°C)",
    "Substrate Temperature (°C)": "Substrate temp (°C)",
}
STACK = {"reported": ms.NAVY, "tier1_within_study": ms.TEAL,
         "tier2_global": ms.SLATE}
STACK_LABEL = {"reported": "reported in the source",
               "tier1_within_study": "filled from the same study",
               "tier2_global": "filled globally"}


def _short(cols) -> list[str]:
    return [SHORT.get(c, c) for c in cols]


def main() -> None:
    ms.apply()
    structure = pd.read_excel(TABLES / "study_structure.xlsx")
    overall = pd.read_excel(TABLES / "imputation_validation.xlsx",
                            sheet_name="overall_ranking").sort_values("nMAE")
    per_col = pd.read_excel(TABLES / "imputation_validation.xlsx",
                            sheet_name="per_column")
    tiers = pd.read_excel(TABLES / "imputation_tiers.xlsx",
                          sheet_name="summary")

    order = overall["strategy"].tolist()
    cols = [c for c in cfg.PRINT_PARAMS if c in set(per_col["column"])]

    fig = ms.plt.figure(figsize=(ms.WIDTHS["double"], 6.9))
    gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.52,
                          height_ratios=[1.0, 1.12])

    ax = fig.add_subplot(gs[0, 0])
    s = structure.set_index("column").reindex(cols)
    y = np.arange(len(cols))
    ax.barh(y - 0.19, s["pct_var_between_studies"], height=0.36,
            color=ms.NAVY, label="variance between studies")
    ax.barh(y + 0.19, s["pct_studies_single_value"], height=0.36,
            color=ms.GOLD, label="studies reporting one value")
    for yi, (a, b) in enumerate(zip(s["pct_var_between_studies"],
                                    s["pct_studies_single_value"])):
        ax.text(a + 1.5, yi - 0.19, f"{a:.0f}", va="center", size=5.6,
                color=ms.MUTED)
        ax.text(b + 1.5, yi + 0.19, f"{b:.0f}", va="center", size=5.6,
                color=ms.MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels(_short(cols), size=6)
    ax.invert_yaxis()
    ax.set_xlim(0, 118)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("% of total")
    ms.grid_axis(ax, "x")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=1, fontsize=5.8)
    ms.panel_tag(ax, "A", dx=-0.42, dy=1.16)

    ax = fig.add_subplot(gs[0, 1])
    o = overall
    nmae = o["nMAE"].to_numpy(dtype=float)

    ceiling = float(np.sort(nmae)[-2]) * 1.42
    broken = nmae > ceiling
    stop = ceiling * 0.82
    colours = [ms.TEAL if st == order[0] else ms.NAVY for st in o["strategy"]]
    ax.barh(range(len(o)), np.where(broken, stop, nmae), color=colours,
            height=0.44)

    for i, (v, cut) in enumerate(zip(nmae, broken)):
        if cut:
            for dx in (-0.055, -0.018):
                ax.plot([stop + dx * ceiling,
                         stop + (dx + 0.038) * ceiling],
                        [i - 0.34, i + 0.34], color="white", lw=2.6,
                        solid_capstyle="butt", zorder=3, clip_on=False)
            ax.text(stop + 0.035 * ceiling, i, f"{v:.2f}", va="center",
                    ha="left", size=6, color=ms.MUTED)
        else:
            ax.text(v + ceiling * 0.015, i, f"{v:.2f}", va="center",
                    ha="left", size=6, color=ms.MUTED)

    ax.set_yticks(range(len(o)))
    ax.set_yticklabels(o["strategy"], size=6)
    ax.set_ylim(len(o) - 0.35, -0.62)
    ax.set_xlim(0, ceiling)
    ax.set_xlabel("mean normalized MAE")
    ms.grid_axis(ax, "x")
    ms.panel_tag(ax, "B", dx=-0.40, dy=1.16)

    ax = fig.add_subplot(gs[1, 0])
    grid = (per_col.pivot(index="strategy", columns="column", values="nMAE")
                   .reindex(index=order, columns=cols))
    v = grid.to_numpy(dtype=float)
    im = ax.imshow(v, cmap=ms.SEQ_NAVY, aspect="auto", vmin=0,
                   vmax=float(np.nanpercentile(v, 95)))
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(_short(cols), size=5.2, rotation=38, ha="right")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, size=6)
    ax.grid(False)
    ms.despine(ax, keep=())
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            if np.isnan(v[i, j]):
                continue
            ax.text(j, i, f"{v[i, j]:.2f}", ha="center", va="center", size=5,
                    color="white" if v[i, j] > im.norm.vmax * 0.55 else ms.TEXT)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("MAE / IQR", size=6)
    cb.ax.tick_params(labelsize=5.5)
    cb.outline.set_visible(False)
    ms.panel_tag(ax, "C", dx=-0.42, dy=1.10)

    ax = fig.add_subplot(gs[1, 1])
    t = tiers.set_index("column").reindex(cols)
    total = float(t[list(STACK)].sum(axis=1).iloc[0])
    left = np.zeros(len(cols))
    for key, colour in STACK.items():
        w = 100 * t[key].to_numpy(dtype=float) / total
        ax.barh(range(len(cols)), w, left=left, color=colour, height=0.62,
                label=STACK_LABEL[key])
        for i, (wi, li) in enumerate(zip(w, left)):
            if wi >= 7:
                ax.text(li + wi / 2, i, f"{wi:.0f}", ha="center", va="center",
                        size=5.4, color="white")
        left += w
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(_short(cols), size=6)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of 2,646 samples")
    ms.grid_axis(ax, "x")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=1, fontsize=5.8)
    ms.panel_tag(ax, "D", dx=-0.42, dy=1.16)

    paths = ms.save(fig, "figS7_imputation_benchmark", step="02_preprocessing")
    print("written:", *[p.name for p in paths])


if __name__ == "__main__":
    main()
