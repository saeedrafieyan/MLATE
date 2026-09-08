from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

NAVY = "#22405C"
TEAL = "#2E7D6F"
RUST = "#C0663F"
GOLD = "#D9A441"
SLATE = "#7B8794"
PLUM = "#6B5B8A"
SAGE = "#8FA88B"
CLAY = "#A8756A"

CATEGORICAL = [NAVY, TEAL, RUST, GOLD, PLUM, SAGE, CLAY, SLATE]

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
    "initiator":                "#E0A76B",
    "ECM_derived":              "#3F6B5E",
}


PROTOCOL_COLORS = {"random": NAVY, "doi": SLATE, "tissue": RUST}
PROTOCOL_LABELS = {"random": "random split",
                   "doi": "study-grouped (DOI)",
                   "tissue": "leave-one-tissue-out"}

GROUP_COLORS = {"ml": NAVY, "dl": TEAL, "foundation": RUST,
                "baseline": SLATE}
GROUP_LABELS = {"ml": "conventional ML", "dl": "deep learning",
                "foundation": "tabular foundation", "baseline": "baseline"}

FAMILY_COLORS = {
    "baseline": SLATE,
    "linear": NAVY,
    "discriminant": PLUM,
    "naive_bayes": TEAL,
    "neighbours": SAGE,
    "svm": GOLD,
    "tree": CLAY,
    "bagging": RUST,
    "boosting": "#1C5E72",
    "neural": "#8E6FA8",
    "meta": "#5C7A54",
    "transformer": "#C08A5E",
    "foundation": "#9C4F3F",
}

FAMILY_LABELS = {
    "baseline": "baseline", "linear": "linear",
    "discriminant": "discriminant", "naive_bayes": "naive Bayes",
    "neighbours": "neighbours", "svm": "SVM", "tree": "tree",
    "bagging": "bagging", "boosting": "boosting", "neural": "neural",
    "meta": "meta-ensemble", "transformer": "transformer",
    "foundation": "foundation model",
}

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
    return SHORT_NAMES.get(name, name.replace("_", " "))

SPLIT_COLORS = {"test": NAVY, "train": SLATE}

METRIC_ORDER = ("accuracy", "weighted_precision", "weighted_recall",
                "weighted_f1", "roc_auc_weighted_ovr", "mcc", "kappa")

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
    return list(METRIC_ORDER_FULL if full else METRIC_ORDER)


def metric_headers(full: bool = False) -> list[str]:
    return [METRIC_LABELS[m] for m in metric_columns(full)]


SEQ_NAVY = LinearSegmentedColormap.from_list("mlate_navy", ["#E8EDF2", NAVY])
SEQ_TEAL = LinearSegmentedColormap.from_list("mlate_teal", ["#E9F1EF", TEAL])
DIVERGING = LinearSegmentedColormap.from_list("mlate_div", [NAVY, "#F2F0EC", RUST])

NAVY_WASH = "#EAF0F6"
TEAL_WASH = "#E9F1EF"
RUST_WASH = "#F7ECE7"

MISSING = "#D9DCE0"
GRIDLINE = "#D6D9DD"
TEXT = "#1A1A1A"
MUTED = "#6E747C"

WIDTHS = {"single": 3.46, "onehalf": 5.12, "double": 7.09}
DPI = 600
from mlate import config as cfg

OUT_DIR = cfg.RESULTS_DIR

SERIF = True
_FONT_STACK = (["Times New Roman", "Nimbus Roman", "DejaVu Serif"] if SERIF
               else ["Arial", "Helvetica", "DejaVu Sans"])


def apply() -> None:
    mpl.rcParams.update({
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

        "text.color":         TEXT,
        "axes.labelcolor":    TEXT,
        "axes.edgecolor":     MUTED,
        "xtick.color":        MUTED,
        "ytick.color":        MUTED,
        "axes.prop_cycle":    mpl.cycler(color=CATEGORICAL),

        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.linewidth":     0.6,
        "xtick.major.width":  0.6,
        "ytick.major.width":  0.6,
        "xtick.major.size":   2.5,
        "ytick.major.size":   2.5,
        "xtick.direction":    "out",
        "ytick.direction":    "out",

        "axes.grid":          True,
        "axes.axisbelow":     True,
        "grid.color":         GRIDLINE,
        "grid.linewidth":     0.5,
        "grid.alpha":         0.7,

        "legend.frameon":     False,
        "legend.handlelength": 1.2,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.2,
        "legend.borderaxespad": 0.0,

        "figure.dpi":         150,
        "savefig.dpi":        DPI,
        "figure.facecolor":   "white",
        "savefig.facecolor":  "white",
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype":       42,
        "ps.fonttype":        42,
        "svg.fonttype":       "none",

        "lines.linewidth":    1.2,
        "lines.markersize":   3.5,
        "patch.linewidth":    0.0,
    })


def figure(width: str = "single", height: float | None = None,
           nrows: int = 1, ncols: int = 1, **kw):
    w = WIDTHS.get(width, width) if isinstance(width, str) else width
    h = height if height is not None else w * 0.68 * nrows
    return plt.subplots(nrows, ncols, figsize=(w, h), **kw)


def grid_axis(ax, axis: str = "y") -> None:
    ax.grid(False)
    ax.grid(True, axis=axis)


def despine(ax, keep=("left", "bottom")) -> None:
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


def label_bars(ax, bars, values, fmt="{:.0f}", pad=0.01,
               horizontal=False, size=6.5) -> None:
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
    from matplotlib import patheffects
    return [patheffects.withStroke(linewidth=width, foreground=colour)]


def panel_tag(ax, tag: str, dx: float = -0.14, dy: float = 1.04) -> None:
    ax.text(dx, dy, tag, transform=ax.transAxes,
            size=9.5, weight="semibold", va="top", ha="left", color=TEXT)


def save(fig, name: str, step: str = "01_data",
         formats=("png", "pdf")) -> list[Path]:
    out = cfg.step_dir(step, "figures")
    written = []
    for ext in formats:
        path = out / f"{name}.{ext}"
        fig.savefig(path, format=ext)
        written.append(path)
    plt.close(fig)
    return written
