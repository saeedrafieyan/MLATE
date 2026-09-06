"""
MLATE V3 — figure design system
===============================

One source of truth for the look of every figure in the paper. Import this
before plotting; never set colours, fonts or sizes ad hoc in a figure script.

    from mlate import style as ms
    ms.apply()
    fig, ax = ms.figure(width="single")
    ...
    ms.save(fig, "fig1_dataset_growth")

This module is the project-specific instance of a portable design system kept
as a skill at ~/.claude/skills/scientific-figures/. That copy carries the same
palette, geometry and helpers with no project imports, plus reference/principles.md
- the rules about honest projection, shared coordinate systems and plotting the
criterion that actually made the decision. Port improvements in both directions.

Design criteria
---------------
minimal    no chartjunk. No frames, no shadows, no gradients, no 3-D. Grid is
           a faint single-axis reference, drawn behind the data. Legends have
           no box. Only the two spines that carry information are kept.
chic       restrained palette, generous whitespace, size and weight used
           sparingly to build hierarchy instead of colour and bold.
publication serif type matching the manuscript body, column-accurate widths,
           vector PDF plus 600-dpi PNG, every glyph legible at print size.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ─────────────────────────────────────────────────────────────────────────────
# PALETTE
# Extends the navy/teal already used in the submitted figures so the revision
# stays visually continuous with what the reviewers have seen. Ordered by
# perceptual distinctness: the first three read clearly in greyscale.
# ─────────────────────────────────────────────────────────────────────────────
NAVY = "#22405C"
TEAL = "#2E7D6F"
RUST = "#C0663F"
GOLD = "#D9A441"
SLATE = "#7B8794"
PLUM = "#6B5B8A"
SAGE = "#8FA88B"
CLAY = "#A8756A"

CATEGORICAL = [NAVY, TEAL, RUST, GOLD, PLUM, SAGE, CLAY, SLATE]

# Eleven biomaterial classes need eleven stable colours. Fixed here so a class
# keeps its colour across Figure 2, Figure 7 and every supplementary panel.
MATERIAL_CLASS_COLORS = {
    "natural_polymer":          NAVY,
    "modified_natural_polymer": TEAL,
    "synthetic_polymer":        RUST,
    "crosslinker":              GOLD,
    "small_molecule":           PLUM,
    "bioactive_material":       SAGE,
    "nanomaterial":             CLAY,
    "serum_buffer_plasma":      SLATE,
    "bioceramics":              "#4F6D8C",
    "enzyme":                   "#9C8AA5",
    "clay":                     "#BFA980",
    "initiator":                "#E0A76B",   # split out of crosslinker
    "ECM_derived":              "#3F6B5E",   # split out of natural_polymer
}

# ─────────────────────────────────────────────────────────────────────────────
# ENCODINGS
# Fixed here, not in the figure scripts, so a reader can carry one key across
# every panel of the paper and the supplement.
# ─────────────────────────────────────────────────────────────────────────────

# Validation protocol. Navy is the interpolation estimate the main text leads
# with; slate is the study-grouped estimate reported alongside it.
PROTOCOL_COLORS = {"random": NAVY, "doi": SLATE, "tissue": RUST}
PROTOCOL_LABELS = {"random": "random split",
                   "doi": "study-grouped (DOI)",
                   "tissue": "leave-one-tissue-out"}

# Broad model group, for figures that put all three benchmarks on one axis.
GROUP_COLORS = {"ml": NAVY, "dl": TEAL, "foundation": RUST,
                "baseline": SLATE}
GROUP_LABELS = {"ml": "conventional ML", "dl": "deep learning",
                "foundation": "tabular foundation", "baseline": "baseline"}

# ─────────────────────────────────────────────────────────────────────────────
# ALGORITHM FAMILY — the mapping already used in Figures S9-S11
#
# Moved here from 04_machine_learning/figures_by_protocol.py so that the deep
# and foundation figures can use the same key. A family must keep its colour
# across every panel of the paper; two figures that disagree about what teal
# means are worse than two figures with no colour at all.
#
# Eleven conventional families exhaust the eight palette hues, so three are
# tinted from existing ones rather than pulled from a second palette. The two
# additions at the end cover the models that step 05 introduces.
# ─────────────────────────────────────────────────────────────────────────────
FAMILY_COLORS = {
    "baseline": SLATE,
    "linear": NAVY,
    "discriminant": PLUM,
    "naive_bayes": TEAL,
    "neighbours": SAGE,
    "svm": GOLD,
    "tree": CLAY,
    "bagging": RUST,
    "boosting": "#1C5E72",      # deep cyan-navy
    "neural": "#8E6FA8",        # light plum
    "meta": "#5C7A54",          # deep sage
    "transformer": "#C08A5E",   # warm clay-gold, for FT-Transformer/TabNet
    "foundation": "#9C4F3F",    # deep rust, for TabPFN/TabICL
}

FAMILY_LABELS = {
    "baseline": "baseline", "linear": "linear",
    "discriminant": "discriminant", "naive_bayes": "naive Bayes",
    "neighbours": "neighbours", "svm": "SVM", "tree": "tree",
    "bagging": "bagging", "boosting": "boosting", "neural": "neural",
    "meta": "meta-ensemble", "transformer": "transformer",
    "foundation": "foundation model",
}

# Abbreviations used on figure axes, where a full registry name would eat the
# row-label budget. Kept here so every figure shortens a name the same way.
SHORT_NAMES = {
    "Logistic Regression": "Logistic Reg.",
    "Logistic Regression (balanced)": "Logistic Reg. (bal.)",
    "Random Forest (balanced)": "Random Forest (bal.)",
    "Quadratic Discriminant": "Quadratic Discrim.",
    "Linear Discriminant": "Linear Discrim.",
    "Hist Gradient Boosting": "Hist Grad. Boosting",
    "Gradient Boosting": "Grad. Boosting",
    "k-Nearest Neighbours": "k-NN",
    "k-NN (distance weighted)": "k-NN (dist. weighted)",
    "Bernoulli Naive Bayes": "Bernoulli NB",
    "Gaussian Naive Bayes": "Gaussian NB",
    "Soft Voting (RF+XGB+LR)": "Soft Voting",
    "Stacking (RF+XGB+LR -> LR)": "Stacking",
    "Dummy (majority)": "Dummy - majority",
    "Dummy (stratified)": "Dummy - stratified",
    "MLP (256-128)": "MLP",
    "FT_Transformer": "FT-Transformer",
    "TabNet_Lite": "TabNet-Lite",
    "NODE_Lite": "NODE-Lite",
    "1D_CNN": "1D-CNN",
    "TabPFN (thinking)": "TabPFN (thinking)",
}


def short(name: str) -> str:
    """Figure-axis label for a model, abbreviated consistently."""
    return SHORT_NAMES.get(name, name.replace("_", " "))

# Train/test partition, for the overfitting diagnostic. Test is the reported
# quantity and takes the lead hue; train is a muted wash, because a training
# score is a diagnostic and must never read as a result.
SPLIT_COLORS = {"test": NAVY, "train": SLATE}

# ─────────────────────────────────────────────────────────────────────────────
# METRIC ORDER — one canonical sequence for every panel in the paper
#
# Referee 1, comment 5: "the metric ordering in the Figure 6 heatmaps is not
# consistent across panels - the conventional-ML panels end with AUC, MCC,
# Kappa, whereas the DL/zero-shot panels end with Kappa, MCC, AUC. A single
# consistent metric order across all panels would aid comparison."
#
# The order below is that of the submitted manuscript's conventional-ML panels,
# so the fix moves the minority of panels rather than retraining every reader,
# and it groups sensibly: overall accuracy, then the precision/recall/F1 trio,
# then ranking ability, then the two chance-corrected agreement statistics.
#
# Every figure and table script must take its column order from here. Hard-
# coding a list in a figure script is what produced the inconsistency the
# referee found.
# ─────────────────────────────────────────────────────────────────────────────
METRIC_ORDER = ("accuracy", "weighted_precision", "weighted_recall",
                "weighted_f1", "roc_auc_weighted_ovr", "mcc", "kappa")

# The wider panel, for supplementary tables that also carry the imbalance-aware
# and ordinal metrics. The first seven are METRIC_ORDER, unchanged, so a reader
# moving between a main-text figure and a supplementary table reads the same
# sequence and then continues.
METRIC_ORDER_FULL = METRIC_ORDER + (
    "balanced_accuracy", "macro_f1", "quadratic_kappa")

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "weighted_precision": "Precision",
    "weighted_recall": "Recall",
    "weighted_f1": "F1",
    "roc_auc_weighted_ovr": "AUC",
    "mcc": "MCC",
    "kappa": "Kappa",
    "balanced_accuracy": "Balanced acc.",
    "macro_f1": "Macro F1",
    "quadratic_kappa": "Quadratic κ",
}


def metric_columns(full: bool = False) -> list[str]:
    """The canonical metric sequence. Never hard-code this in a figure script."""
    return list(METRIC_ORDER_FULL if full else METRIC_ORDER)


def metric_headers(full: bool = False) -> list[str]:
    """Display labels for `metric_columns`, in the same order."""
    return [METRIC_LABELS[m] for m in metric_columns(full)]


# Ordinal target scales: low-to-high, single-hue so magnitude reads directly.
SEQ_NAVY = LinearSegmentedColormap.from_list("mlate_navy", ["#E8EDF2", NAVY])
SEQ_TEAL = LinearSegmentedColormap.from_list("mlate_teal", ["#E9F1EF", TEAL])
DIVERGING = LinearSegmentedColormap.from_list("mlate_div", [NAVY, "#F2F0EC", RUST])

# Washed tints of the two lead hues, for filled areas that must sit behind a
# stroked outline of the same colour - box faces, confidence bands, the shaded
# half of a paired comparison. Defined here because a figure that needs a pale
# navy should not invent one: 04_machine_learning/figures.py previously carried
# `ms.NAVY_WASH if hasattr(ms, "NAVY_WASH") else "#EAF0F6"`, which is precisely
# the ad-hoc colour this module exists to prevent.
NAVY_WASH = "#EAF0F6"
TEAL_WASH = "#E9F1EF"
RUST_WASH = "#F7ECE7"

MISSING = "#D9DCE0"   # neutral fill for "not reported"
GRIDLINE = "#D6D9DD"
TEXT = "#1A1A1A"
MUTED = "#6E747C"

# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRY — IOP / Biofabrication column widths, in inches
# ─────────────────────────────────────────────────────────────────────────────
WIDTHS = {"single": 3.46, "onehalf": 5.12, "double": 7.09}
DPI = 600
from mlate import config as cfg

OUT_DIR = cfg.RESULTS_DIR   # overridden per step via save(..., step=)

# Serif, to match the manuscript body text. Flip SERIF to False for a
# sans-serif build if a journal ever asks for it.
SERIF = True
_FONT_STACK = (["Times New Roman", "Nimbus Roman", "DejaVu Serif"] if SERIF
               else ["Arial", "Helvetica", "DejaVu Sans"])


def apply() -> None:
    """Install the house style. Call once, before any plotting."""
    mpl.rcParams.update({
        # type
        "font.family":        "serif" if SERIF else "sans-serif",
        ("font.serif" if SERIF else "font.sans-serif"): _FONT_STACK,
        "font.size":          8,
        "axes.titlesize":     9,
        "axes.labelsize":     8,
        "xtick.labelsize":    7,
        "ytick.labelsize":    7,
        "legend.fontsize":    7,
        "figure.titlesize":   10,
        "mathtext.fontset":   "stix",

        # colour
        "text.color":         TEXT,
        "axes.labelcolor":    TEXT,
        "axes.edgecolor":     MUTED,
        "xtick.color":        MUTED,
        "ytick.color":        MUTED,
        "axes.prop_cycle":    mpl.cycler(color=CATEGORICAL),

        # minimal frame - only the informative spines survive
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.linewidth":     0.6,
        "xtick.major.width":  0.6,
        "ytick.major.width":  0.6,
        "xtick.major.size":   2.5,
        "ytick.major.size":   2.5,
        "xtick.direction":    "out",
        "ytick.direction":    "out",

        # faint reference grid, always behind the data
        "axes.grid":          True,
        "axes.axisbelow":     True,
        "grid.color":         GRIDLINE,
        "grid.linewidth":     0.5,
        "grid.alpha":         0.7,

        # unboxed legend
        "legend.frameon":     False,
        "legend.handlelength": 1.2,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.2,
        "legend.borderaxespad": 0.0,

        # output
        "figure.dpi":         150,
        "savefig.dpi":        DPI,
        "figure.facecolor":   "white",
        "savefig.facecolor":  "white",
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype":       42,   # embed as TrueType, keeps text selectable
        "ps.fonttype":        42,
        "svg.fonttype":       "none",

        "lines.linewidth":    1.2,
        "lines.markersize":   3.5,
        "patch.linewidth":    0.0,
    })


def figure(width: str = "single", height: float | None = None,
           nrows: int = 1, ncols: int = 1, **kw):
    """
    Create a correctly-sized figure.

    width   'single' | 'onehalf' | 'double', or a number in inches
    height  inches; defaults to a 0.68 aspect per axes row
    """
    w = WIDTHS.get(width, width) if isinstance(width, str) else width
    h = height if height is not None else w * 0.68 * nrows
    return plt.subplots(nrows, ncols, figsize=(w, h), **kw)


def grid_axis(ax, axis: str = "y") -> None:
    """Restrict the grid to the single axis that carries magnitude."""
    ax.grid(False)
    ax.grid(True, axis=axis)


def despine(ax, keep=("left", "bottom")) -> None:
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


def label_bars(ax, bars, values, fmt="{:.0f}", pad=0.01,
               horizontal=False, size=6.5) -> None:
    """Direct-label bars so the reader never has to trace back to an axis."""
    span = (ax.get_xlim()[1] if horizontal else ax.get_ylim()[1]) or 1
    for bar, v in zip(bars, values):
        if horizontal:
            ax.text(bar.get_width() + pad * span,
                    bar.get_y() + bar.get_height() / 2,
                    fmt.format(v), va="center", ha="left",
                    size=size, color=MUTED)
        else:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + pad * span,
                    fmt.format(v), ha="center", va="bottom",
                    size=size, color=MUTED)


def halo(width: float = 2.0, colour: str = "white") -> list:
    """
    A white outline for text sitting on top of dense marks.

    Cluster labels land in the middle of a scatter, where plain text on either
    a light or a dark patch of points is unreadable. A stroke behind the glyph
    keeps the label legible without a filled box, which would hide the data it
    is annotating.
    """
    from matplotlib import patheffects
    return [patheffects.withStroke(linewidth=width, foreground=colour)]


def panel_tag(ax, tag: str, dx: float = -0.14, dy: float = 1.04) -> None:
    """Panel letter, upper-left, unbolded - hierarchy comes from position."""
    ax.text(dx, dy, tag, transform=ax.transAxes,
            size=9.5, weight="semibold", va="top", ha="left", color=TEXT)


def save(fig, name: str, step: str = "01_data",
         formats=("png", "pdf")) -> list[Path]:
    """Write vector + raster copies under results/<step>/figures/."""
    out = cfg.step_dir(step, "figures")
    written = []
    for ext in formats:
        path = out / f"{name}.{ext}"
        fig.savefig(path, format=ext)
        written.append(path)
    plt.close(fig)
    return written
