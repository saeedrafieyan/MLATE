from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import re
import textwrap
from pathlib import Path

import numpy as np
from matplotlib.patches import Rectangle
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from mlate import style as ms

from mlate import config as cfg

DATA = cfg.DATASET
TAXONOMY = cfg.TAXONOMY
TABLE_DIR = cfg.step_dir("01_data", "tables")

V2 = {"Studies": 88, "Biomaterials": 60, "Cell lines": 49, "Samples": 1171}

CLASS_ORDER = [
    "modified_natural_polymer", "synthetic_polymer", "natural_polymer",
    "crosslinker", "initiator", "nanomaterial", "bioactive_material",
    "small_molecule", "ECM_derived", "bioceramics", "enzyme",
    "serum_buffer_plasma", "clay",
]

PRINTABILITY_LABELS = {
    0: "0  not extruded", 1: "1  liquid / beading",
    2: "2  extrudable", 3: "3  extrudable, optimised",
}
RESPONSE_LABELS = {
    1: "1  no cells", 2: "2  poor short-term",
    3: "3  good short-term",
    4: "4  good short, poor long", 5: "5  good short + long",
}

ACELLULAR = "NoCellCultured"


def pretty(name: str) -> str:
    return name.replace("_", " ")


def load() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    df = pd.read_excel(DATA)
    tax = pd.read_csv(TAXONOMY)
    cols = list(df.columns)
    biomaterials = cols[5:cols.index("Cell Line")]
    assert set(tax["column"]) == set(biomaterials), \
        "taxonomy and dataset disagree on the biomaterial columns"
    return df, tax, biomaterials


def fig_biomaterial_taxonomy(tax) -> None:
    t = tax.copy()
    t["material_class"] = pd.Categorical(t["material_class"],
                                         CLASS_ORDER, ordered=True)
    t = t.sort_values(["material_class", "display_name"]).reset_index(drop=True)

    n = len(t)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    t["theta"] = theta
    step = 2 * np.pi / n

    w = ms.WIDTHS["double"]
    fig = plt.figure(figsize=(w, w * 1.10))
    ax = fig.add_axes([0.04, 0.13, 0.92, 0.85], projection="polar")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.axis("off")

    R_ROOT, R_NODE, R_LEAF, R_TIP = 0.0, 0.30, 0.36, 1.0

    for cls, block in t.groupby("material_class", observed=True):
        if block.empty:
            continue
        colour = ms.MATERIAL_CLASS_COLORS[cls]
        a0, a1 = block["theta"].min(), block["theta"].max()

        arc = np.linspace(a0 - step / 2, a1 + step / 2, 64)
        ax.plot(arc, np.full_like(arc, R_NODE), color=colour, lw=2.0,
                solid_capstyle="butt", zorder=2)

        mid = (a0 + a1) / 2
        ax.plot([mid, mid], [R_ROOT, R_NODE], color=colour, lw=1.4,
                solid_capstyle="butt", zorder=2)

    for _, row in t.iterrows():
        colour = ms.MATERIAL_CLASS_COLORS[row["material_class"]]
        ang = row["theta"]
        ax.plot([ang, ang], [R_NODE, R_LEAF], color=colour, lw=0.9,
                solid_capstyle="butt", alpha=0.85, zorder=1)
        ax.plot([ang, ang], [R_LEAF, R_TIP], color=colour, lw=1.5,
                solid_capstyle="round", alpha=0.9, zorder=1)
        screen = (90 - np.rad2deg(ang)) % 360
        flip = 90 < screen <= 270
        ax.text(ang, R_TIP + 0.035, row["display_name"],
                rotation=screen + (180 if flip else 0),
                rotation_mode="anchor",
                ha="right" if flip else "left", va="center",
                size=4.6, color=ms.TEXT)

    ax.set_ylim(0, 1.34)

    ax.add_patch(Circle((0, 0), 0.185, transform=ax.transData._b,
                        facecolor="white", edgecolor=ms.GRIDLINE,
                        lw=0.8, zorder=3))
    ax.text(0, 0, f"{n}\nbiomaterials", ha="center", va="center",
            size=8, color=ms.TEXT, transform=ax.transData, zorder=4,
            linespacing=1.35)

    counts = t["material_class"].value_counts().reindex(CLASS_ORDER)
    handles = [plt.Line2D([], [], color=ms.MATERIAL_CLASS_COLORS[c], lw=2.6,
                          label=f"{pretty(c)}  ({counts[c]})")
               for c in CLASS_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, 0.005), fontsize=6.4,
               columnspacing=1.4, handlelength=1.1, labelspacing=0.55)

    ms.save(fig, step="01_data", name="fig2_biomaterial_taxonomy")


def fig_overview(df) -> None:
    fig = plt.figure(figsize=(ms.WIDTHS["double"], 4.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.45, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    with_outcome = int((df["Cell Response"] != 1).sum())
    vals = [with_outcome, len(df) - with_outcome]
    wedges, _ = ax.pie(vals, colors=[ms.TEAL, ms.SLATE], startangle=90,
                       radius=0.86,
                       wedgeprops=dict(width=0.36, edgecolor="white", lw=1.2))
    ax.text(0, 0, f"{len(df):,}\nsamples", ha="center", va="center", size=8)
    for w, lab, v in zip(wedges, ["bioprinted", "3D printed"], vals):
        a = np.deg2rad((w.theta1 + w.theta2) / 2)
        x, y = 1.02 * np.cos(a), 1.02 * np.sin(a)
        ax.text(x, y, f"{lab}\n{v:,}  ({100*v/len(df):.1f}%)",
                ha="left" if x >= 0 else "right", va="center",
                size=7, color=ms.TEXT)
    ax.set(aspect="equal")
    ax.set_xlim(-1.9, 1.9)
    ax.set_ylim(-1.15, 1.15)
    ms.panel_tag(ax, "A", dx=-0.02, dy=1.06)

    ax = fig.add_subplot(gs[0, 1])
    top = (df.loc[df["Cell Line"] != ACELLULAR, "Cell Line"]
             .value_counts().head(12).iloc[::-1])
    bars = ax.barh(range(len(top)), top.values, color=ms.NAVY, height=0.68)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index, size=6)
    ax.set_xlabel("samples")
    ax.set_xlim(0, top.max() * 1.16)
    ms.grid_axis(ax, "x")
    ms.label_bars(ax, bars, top.values, horizontal=True, size=6)
    ax.set_xticks([])
    ms.despine(ax, keep=("left",))
    ms.panel_tag(ax, "B", dx=-0.42, dy=1.02)

    for j, (col, labels, cmap) in enumerate([
        ("Printability", PRINTABILITY_LABELS, ms.SEQ_NAVY),
        ("Cell Response", RESPONSE_LABELS, ms.SEQ_TEAL),
    ]):
        ax = fig.add_subplot(gs[1, j])
        counts = df[col].value_counts().sort_index()
        shades = [cmap(0.32 + 0.62 * i / max(1, len(counts) - 1))
                  for i in range(len(counts))]
        bars = ax.bar(range(len(counts)), counts.values, color=shades,
                      width=0.68)
        ms.label_bars(ax, bars, counts.values, pad=0.025)
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(
            [textwrap.fill(labels[k], 12) for k in counts.index], size=5.8)
        ax.set_ylabel("samples")
        ax.set_ylim(0, counts.max() * 1.18)
        ms.grid_axis(ax, "y")
        ms.panel_tag(ax, "C" if j == 0 else "D", dx=-0.16, dy=1.06)
        ax.set_title(col, size=8, color=ms.MUTED, pad=4)

    ms.save(fig, step="01_data", name="fig3_overview")


def fig_tissue_composition(df) -> None:
    counts = df["target_tissue"].value_counts()
    fig, ax = ms.figure("onehalf", height=3.4)

    order = counts.iloc[::-1]
    colours = [ms.SLATE if t in {"acellular", "general_biocompatibility",
                                 "undifferentiated_stem_cell", "non_mammalian"}
               else ms.NAVY for t in order.index]
    bars = ax.barh(range(len(order)), order.values, color=colours, height=0.7)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([pretty(t) for t in order.index], size=6.5)
    ax.set_xlabel("samples")
    ax.set_xlim(0, order.max() * 1.14)
    ms.grid_axis(ax, "x")
    ms.label_bars(ax, bars, order.values, horizontal=True, size=6)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=l) for c, l in
               [(ms.NAVY, "organ-specific"), (ms.SLATE, "not organ-specific")]]
    ax.legend(handles=handles, loc="lower right", fontsize=6.5)
    ms.save(fig, step="01_data", name="fig4_tissue_composition")


def fig_material_classes(tax) -> None:
    counts = tax["material_class"].value_counts().reindex(CLASS_ORDER).iloc[::-1]
    fig, ax = ms.figure("onehalf", height=2.8)
    bars = ax.barh(range(len(counts)), counts.values,
                   color=[ms.MATERIAL_CLASS_COLORS[c] for c in counts.index],
                   height=0.68)
    ax.set_yticks(range(len(counts)))
    ax.set_yticklabels([pretty(c) for c in counts.index], size=7)
    ax.set_xlabel("number of distinct materials")
    ax.set_xlim(0, counts.max() * 1.14)
    ms.grid_axis(ax, "x")
    ms.label_bars(ax, bars, counts.values, horizontal=True)
    ms.save(fig, step="01_data", name="fig7_material_classes")


def normalise_doi(doi: str) -> str:
    d = str(doi).strip().lower()
    d = d.replace("%2f", "/").replace("https://doi.org/", "")
    return d.replace("doi.org/", "").rstrip(".").strip()


def tidy_reference(ref: str) -> str:
    s = " ".join(str(ref).split())
    year = ""
    m = re.search(r"(1[89]\d{2}|20\d{2})", s)
    if m:
        year, s = m.group(1), s[:m.start()]
    s = re.sub(r"[.,;:&\s]+$", "", s).strip()

    tail = ""
    m = re.search(r"\bet\.?\s*al\b\.?", s, flags=re.I)
    if m:
        tail, s = "et al.", s[:m.start()]
    elif re.search(r"\band\b", s, flags=re.I):
        parts = re.split(r"\band\b", s, flags=re.I)
        names = [p.strip(" .,;").split()[-1] for p in parts if p.strip()]
        return f"{' & '.join(names)} {year}".strip()

    tokens = re.sub(r"[.,;:\s]+$", "", s).strip().split()
    surname = tokens[-1].strip(" .,;") if tokens else s
    return " ".join(x for x in (surname, tail, year) if x)


def squarify(values, x, y, dx, dy):
    total = sum(values)
    scaled = [v * dx * dy / total for v in values]
    rects, i = [], 0

    def worst(row, side):
        s = sum(row)
        if s == 0 or side == 0:
            return float("inf")
        return max(max((side * side * r) / (s * s) for r in row),
                   max((s * s) / (side * side * r) for r in row))

    while i < len(scaled):
        row, side = [], dy if dx >= dy else dx
        while i < len(scaled):
            trial = row + [scaled[i]]
            if row and worst(trial, side) > worst(row, side):
                break
            row, i = trial, i + 1
        s = sum(row)
        if dx >= dy:
            w = s / dy if dy else 0
            oy = y
            for v in row:
                h = v / w if w else 0
                rects.append((x, oy, w, h))
                oy += h
            x, dx = x + w, dx - w
        else:
            h = s / dx if dx else 0
            ox = x
            for v in row:
                w = v / h if h else 0
                rects.append((ox, y, w, h))
                ox += w
            y, dy = y + h, dy - h
    return rects


def study_contributions(df) -> pd.DataFrame:
    d = df.copy()
    d["_doi"] = d["DOI"].map(normalise_doi)
    g = (d.groupby("_doi")
          .agg(samples=("DOI", "size"),
               reference=("Reference", lambda s: s.mode().iloc[0]),
               tissue=("target_tissue", lambda s: s.mode().iloc[0]),
               bioprinted=("Cell Line", lambda s: int((s != ACELLULAR).sum())))
          .sort_values("samples", ascending=False))
    g["label"] = g.reference.map(tidy_reference)

    dup = g.label.duplicated(keep=False)
    for lab in g.loc[dup, "label"].unique():
        for k, idx in enumerate(g.index[g.label == lab]):
            g.loc[idx, "label"] = f"{lab}{chr(ord('a') + k)}"

    g.insert(0, "rank", range(1, len(g) + 1))
    g["share_pct"] = (100 * g.samples / g.samples.sum()).round(2)
    g["cumulative_pct"] = g.share_pct.cumsum().round(2)
    return g.reset_index().rename(columns={"_doi": "doi"})


STUDY_TOP_N = 30


def fig_study_distribution(df) -> None:
    g = study_contributions(df)
    total = g.samples.sum()

    fig = plt.figure(figsize=(ms.WIDTHS["double"], 5.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.12],
                          height_ratios=[1.32, 1.0], wspace=0.30, hspace=0.42)

    axA = fig.add_subplot(gs[:, 0])
    top = g.head(STUDY_TOP_N).iloc[::-1]
    bars = axA.barh(range(len(top)), top.samples.values, color=ms.NAVY,
                    height=0.72)
    axA.set_yticks(range(len(top)))
    axA.set_yticklabels(top.label.values, size=6.0)
    axA.set_xlabel("samples contributed")
    axA.set_xlim(0, top.samples.max() * 1.16)
    axA.set_ylim(-0.8, len(top) - 0.2)
    ms.grid_axis(axA, "x")
    ms.label_bars(axA, bars, top.samples.values, horizontal=True, size=5.6)
    ms.panel_tag(axA, "A", dx=-0.42, dy=1.03)

    axB = fig.add_subplot(gs[0, 1])
    side = 100.0
    rects = squarify(list(g.samples.values), 0, 0, side, side)
    pending = []
    for k, ((x, y, w, h), (_, row)) in enumerate(zip(rects, g.iterrows())):
        axB.add_patch(Rectangle((x, y), w, h,
                                facecolor=ms.NAVY if k < STUDY_TOP_N
                                else ms.SLATE,
                                edgecolor="white", linewidth=0.35))
        if k < 12:
            pending.append((x + w / 2, y + h / 2, w, h,
                            row.label.replace(" et al.", "")))
    axB.set_xlim(0, side)
    axB.set_ylim(side, 0)
    axB.set_aspect("equal")
    axB.axis("off")

    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = axB.transData.inverted()

    def fits(t, w, h):
        bb = t.get_window_extent(rend)
        (x0, y0), (x1, y1) = inv.transform(((bb.x0, bb.y0), (bb.x1, bb.y1)))
        return (x1 - x0) <= w * 0.92 and abs(y1 - y0) <= h * 0.85

    for rank, (cx, cy, w, h, text) in enumerate(pending, start=1):
        t = axB.text(cx, cy, text, ha="center", va="center", size=4.6,
                     color="white")
        if fits(t, w, h):
            continue
        t.remove()
        t = axB.text(cx, cy, str(rank), ha="center", va="center", size=4.6,
                     color="white")
        if not fits(t, w, h):
            t.remove()

    axB.legend(handles=[Rectangle((0, 0), 1, 1, color=ms.NAVY,
                                  label=f"the {STUDY_TOP_N} studies named in A"),
                        Rectangle((0, 0), 1, 1, color=ms.SLATE,
                                  label=f"remaining {len(g) - STUDY_TOP_N} studies")],
               loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2,
               fontsize=5.8, handlelength=1.0)
    axB.set_title(f"all {len(g)} studies, area proportional to samples",
                  size=7, color=ms.MUTED, pad=3)
    ms.panel_tag(axB, "B", dx=-0.04, dy=1.12)

    axC = fig.add_subplot(gs[1, 1])
    cum = g.cumulative_pct.values
    xs = np.arange(1, len(g) + 1)
    axC.plot(xs, cum, color=ms.NAVY, lw=1.3)
    axC.fill_between(xs, cum, color=ms.NAVY, alpha=0.10)
    for k, colour in ((10, ms.RUST), (STUDY_TOP_N, ms.TEAL)):
        axC.plot([k, k], [0, cum[k - 1]], color=colour, lw=0.9, ls="--")
        axC.plot([0, k], [cum[k - 1]] * 2, color=colour, lw=0.9, ls="--")
        axC.plot([k], [cum[k - 1]], "o", color=colour, ms=2.6)
        axC.text(k + 5, cum[k - 1] - 7.0,
                 f"top {k} studies → {cum[k - 1]:.0f}%",
                 size=6.0, color=colour, va="top", ha="left")
    axC.text(len(g) * 0.98, 6,
             f"median {int(g.samples.median())} samples per study\n"
             f"{int((g.samples == 1).sum())} studies contribute one",
             size=5.6, color=ms.MUTED, ha="right", va="bottom", linespacing=1.5)
    axC.set_xlabel(f"studies, ranked by contribution  (n = {len(g)})")
    axC.set_ylabel("cumulative share of samples (%)")
    axC.set_xlim(0, len(g))
    axC.set_ylim(0, 102)
    ms.grid_axis(axC, "y")
    ms.panel_tag(axC, "C", dx=-0.20, dy=1.10)

    ms.save(fig, step="01_data", name="figS1_study_distribution")

    g.drop(columns=["reference"]).to_excel(
        TABLE_DIR / "tableS4_study_contributions.xlsx", index=False)


def fig_cooccurrence(df, tax, biomaterials, top_n: int = 22) -> None:
    present = (df[biomaterials].fillna(0) != 0)
    top = present.sum().sort_values(ascending=False).head(top_n).index.tolist()
    m = present[top].astype(int)
    co = m.T @ m
    np.fill_diagonal(co.values, 0)

    names = (tax.set_index("column").loc[top, "display_name"]
                .str.slice(0, 20).tolist())

    fig, ax = ms.figure("double", height=6.2)
    im = ax.imshow(co.values, cmap=ms.SEQ_NAVY, aspect="equal")
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(names, rotation=90, size=6)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(names, size=6)
    ax.grid(False)
    ms.despine(ax, keep=())

    for i in range(len(top)):
        for j in range(len(top)):
            v = co.values[i, j]
            if v > 0:
                ax.text(j, i, str(v), ha="center", va="center", size=4.4,
                        color="white" if v > co.values.max() * 0.55 else ms.TEXT)

    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("co-formulated samples", size=7)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6)
    ms.save(fig, step="01_data", name="figS2_material_cooccurrence")


def fig_cell_density(df) -> None:
    d = df[(df["Cell Line"] != ACELLULAR)
           & df["Cell Density (million cells/mL)"].notna()]
    top = d["Cell Line"].value_counts().head(15).index.tolist()
    data = [d.loc[d["Cell Line"] == c,
                  "Cell Density (million cells/mL)"].values for c in top]

    fig, ax = ms.figure("onehalf", height=3.6)
    bp = ax.boxplot(data[::-1], vert=False, widths=0.6, patch_artist=True,
                    medianprops=dict(color=ms.RUST, lw=1.1),
                    flierprops=dict(marker="o", markersize=1.8,
                                    markerfacecolor=ms.MUTED,
                                    markeredgecolor="none", alpha=0.6),
                    whiskerprops=dict(color=ms.MUTED, lw=0.6),
                    capprops=dict(color=ms.MUTED, lw=0.6))
    for patch in bp["boxes"]:
        patch.set(facecolor=ms.TEAL, alpha=0.32, edgecolor=ms.TEAL, lw=0.7)

    ax.set_yticklabels(top[::-1], size=6.5)
    ax.set_xlabel("cell density (million cells mL$^{-1}$)")
    ax.set_xscale("symlog", linthresh=1)
    ms.grid_axis(ax, "x")
    ms.save(fig, step="01_data", name="figS3_cell_density")


def fig_printing_parameters(df) -> None:
    params = [c for c in df.columns if c in {
        "Physical Crosslinking Duration (s)", "Photo Crosslinking Duration (s)",
        "Extrusion Pressure (kPa)", "Nozzle Movement Speed (mm/s)",
        "Nozzle Diameter (µm)", "Syringe Temperature (°C)",
        "Substrate Temperature (°C)"}]

    fig, axes = ms.figure("double", height=3.6, nrows=2, ncols=4)
    for ax, col in zip(axes.ravel(), params):
        s = df[col].dropna()
        ax.hist(s, bins=28, color=ms.NAVY, edgecolor="white", linewidth=0.3)
        ax.set_title(textwrap.fill(col, 26), size=6.5, pad=4)
        ax.set_ylabel("samples", size=6.5)
        ax.tick_params(labelsize=6)
        pct = 100 * df[col].isna().mean()
        ax.text(0.97, 0.93, f"{pct:.0f}% missing", transform=ax.transAxes,
                ha="right", va="top", size=5.8, color=ms.RUST)
        ms.grid_axis(ax, "y")

    axes.ravel()[-1].axis("off")
    fig.tight_layout()
    ms.save(fig, step="01_data", name="figS4_printing_parameters")


def fig_missingness(df, biomaterials) -> None:
    cols = [c for c in df.columns
            if c not in biomaterials
            and c not in {"Reference", "DOI", "target_tissue",
                          "target_tissue_all", "is_cancer_model",
                          "dup_group_id", "label_conflict"}]
    miss = (df[cols].isna().mean() * 100).sort_values()
    miss = miss[miss > 0]

    fig, ax = ms.figure("onehalf", height=2.3)
    bars = ax.barh(range(len(miss)), miss.values, color=ms.RUST, height=0.62)
    ax.set_yticks(range(len(miss)))
    ax.set_yticklabels([textwrap.fill(c, 30) for c in miss.index], size=6.5)
    ax.set_xlabel("% of samples where the value was not reported")
    ax.set_xlim(0, 100)
    ms.grid_axis(ax, "x")
    ms.label_bars(ax, bars, miss.values, fmt="{:.0f}%", horizontal=True, size=6)
    ms.save(fig, step="01_data", name="figS5_missingness")


def fig_study_clustering(df, biomaterials) -> None:
    fig, axes = ms.figure("double", height=2.5, ncols=3)

    ax = axes[0]
    shares = []
    for t in ["Printability", "Cell Response"]:
        g = df.groupby("DOI")[t]
        between = g.mean().sub(df[t].mean()).pow(2).mul(g.size()).sum()
        shares.append(100 * between / df[t].sub(df[t].mean()).pow(2).sum())
    bars = ax.bar([0, 1], shares, width=0.55, color=[ms.NAVY, ms.TEAL])
    ms.label_bars(ax, bars, shares, fmt="{:.0f}%", pad=0.03)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Printability", "Cell\nResponse"], size=6.5)
    ax.set_ylabel("% of label variance\nbetween studies")
    ax.set_ylim(0, 100)
    ms.grid_axis(ax, "y")
    ms.panel_tag(ax, "A", dx=-0.3)

    ax = axes[1]
    pattern = (df[biomaterials].fillna(0) != 0).astype(int).astype(str).agg("".join, axis=1)
    frames = {
        "materials": pattern,
        "materials\n+ cell line": pattern + "|" + df["Cell Line"].astype(str),
    }
    vals = []
    for key in frames:
        t = pd.DataFrame({"p": frames[key], "doi": df["DOI"]})
        uniq = t.groupby("p")["doi"].nunique().loc[lambda s: s == 1].index
        vals.append(100 * t.groupby("p").size()[uniq].sum() / len(df))
    bars = ax.bar(range(2), vals, width=0.55, color=[ms.SLATE, ms.RUST])
    ms.label_bars(ax, bars, vals, fmt="{:.0f}%", pad=0.03)
    ax.set_xticks(range(2))
    ax.set_xticklabels(list(frames), size=6.5)
    ax.set_ylabel("% of samples whose signature\nidentifies one study")
    ax.set_ylim(0, 100)
    ms.grid_axis(ax, "y")
    ms.panel_tag(ax, "B", dx=-0.3)

    ax = axes[2]
    per = df.groupby("DOI").size()
    fracs = [100 * sum(k * (1 - (1 - f) ** (k - 1)) for k in per) / len(df)
             for f in (0.1, 0.2, 0.3)]
    bars = ax.bar(range(3), fracs, width=0.55, color=ms.NAVY)
    ms.label_bars(ax, bars, fracs, fmt="{:.0f}%", pad=0.03)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["10%", "20%", "30%"], size=6.5)
    ax.set_xlabel("random test-set size")
    ax.set_ylabel("% of test rows with a\nsame-study row in training")
    ax.set_ylim(0, 105)
    ms.grid_axis(ax, "y")
    ms.panel_tag(ax, "C", dx=-0.3)

    fig.tight_layout()
    ms.save(fig, step="01_data", name="figS6_study_clustering")


def write_tables(df, tax, biomaterials) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for _, r in tax.iterrows():
        s = df[r["column"]]
        present = s.fillna(0) != 0
        v = s[present]
        rows.append({
            "Biomaterial": r["display_name"],
            "Unit": r["unit"],
            "Class": pretty(r["material_class"]),
            "Samples": int(present.sum()),
            "Studies": int(df.loc[present, "DOI"].nunique()),
            "Min": v.min(), "Median": v.median(),
            "Mean": round(v.mean(), 4) if len(v) else np.nan,
            "Max": v.max(),
            "SD": round(v.std(), 4) if len(v) > 1 else np.nan,
        })
    s1 = pd.DataFrame(rows).sort_values("Samples", ascending=False)
    s1.to_excel(TABLE_DIR / "tableS1_biomaterials.xlsx", index=False)

    d = df[df["Cell Line"] != ACELLULAR]
    dens = d.groupby("Cell Line")["Cell Density (million cells/mL)"]
    s2 = pd.DataFrame({
        "Samples": d.groupby("Cell Line").size(),
        "Studies": d.groupby("Cell Line")["DOI"].nunique(),
        "Tissue": d.groupby("Cell Line")["target_tissue"]
                   .agg(lambda x: x.mode().iat[0] if len(x.mode()) else ""),
        "Min density": dens.min(), "Median density": dens.median(),
        "Max density": dens.max(),
    }).sort_values("Samples", ascending=False).reset_index()
    s2.to_excel(TABLE_DIR / "tableS2_cell_lines.xlsx", index=False)

    s3 = pd.DataFrame({
        "Samples": df.groupby("target_tissue").size(),
        "Studies": df.groupby("target_tissue")["DOI"].nunique(),
        "Cell lines": df.groupby("target_tissue")["Cell Line"].nunique(),
        "Bioprinted": df[df["Cell Line"] != ACELLULAR]
                        .groupby("target_tissue").size(),
        "Mean printability": df.groupby("target_tissue")["Printability"].mean()
                               .round(2),
        "Mean cell response": df.groupby("target_tissue")["Cell Response"]
                                .mean().round(2),
    }).fillna(0).sort_values("Samples", ascending=False).reset_index()
    s3["% of dataset"] = (100 * s3["Samples"] / len(df)).round(1)
    s3.to_excel(TABLE_DIR / "tableS3_tissue_composition.xlsx", index=False)

    summary = pd.DataFrame([
        ("Samples", len(df), V2["Samples"]),
        ("Studies (unique DOI)", df["DOI"].nunique(), V2["Studies"]),
        ("Biomaterials", len(biomaterials), V2["Biomaterials"]),
        ("Cell lines", df["Cell Line"].nunique(), V2["Cell lines"]),
        ("Tissue categories", df["target_tissue"].nunique(), 0),
        ("Bioprinted samples", int((df["Cell Line"] != ACELLULAR).sum()), 0),
        ("Printing parameters", 7, 7),
    ], columns=["Item", "MLATE V3", "MLATE V2"])
    summary["Growth"] = (summary["MLATE V3"] / summary["MLATE V2"]
                         .replace(0, np.nan)).round(2)
    summary.to_excel(TABLE_DIR / "table_dataset_summary.xlsx", index=False)

    print(f"\ntables -> {TABLE_DIR}")
    print(summary.to_string(index=False))


def main() -> None:
    ms.apply()
    df, tax, biomaterials = load()
    print(f"{len(df)} samples x {len(biomaterials)} biomaterials "
          f"x {df['DOI'].nunique()} studies")

    for fn, args in [
        (fig_biomaterial_taxonomy, (tax,)),
        (fig_overview, (df,)),
        (fig_tissue_composition, (df,)),
        (fig_material_classes, (tax,)),
        (fig_study_distribution, (df,)),
        (fig_cooccurrence, (df, tax, biomaterials)),
        (fig_cell_density, (df,)),
        (fig_printing_parameters, (df,)),
        (fig_missingness, (df, biomaterials)),
        (fig_study_clustering, (df, biomaterials)),
    ]:
        fn(*args)
        print(f"  ok  {fn.__name__}")

    write_tables(df, tax, biomaterials)
    print(f"\nfigures -> {ms.OUT_DIR}")


if __name__ == "__main__":
    main()
