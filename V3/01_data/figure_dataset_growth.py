"""
Figure 1 - growth of the corpus from MLATE V2 to V3
===================================================

    python 01_data/figure_dataset_growth.py

Grouped bars on one shared axis, with height measured as size relative to V2.

The previous version drew four independent bar panels, each scaled so that the
V3 bar reached the top of its own panel. Under that scaling every V3 bar is full
height by construction, so the growth is encoded only in how short the V2 bar
happens to be, and three of the four ratios sit between 2.2 and 2.5. The panels
therefore look alike and the one genuinely different result - cell lines, which
nearly quadrupled - does not stand out.

Here the bar height is the fold change itself, on a single axis shared by all
four metrics. The V2 bar is 1 by definition and the V3 bar is 2.2 to 3.8, so the
difference a reader is being asked to see is the difference they actually see.
Absolute counts are not lost: both bars carry theirs as a direct label, which is
where a reader would look for an exact number anyway rather than reading it off
a gridline.

V3 counts are read from the dataset at run time rather than typed in. V2 counts
are historical and cannot be recomputed, so they are declared once below with
their source.
"""

from __future__ import annotations

import sys
from pathlib import Path
from urllib.parse import unquote

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from mlate import config as cfg
from mlate import style as ms
from mlate.dataset import load_dataset

# Reported in the MLATE V2 paper. Not recomputable from the current dataset,
# which is why they are written down here with their provenance instead of
# being derived from something that no longer exists.
V2 = {"studies": 88, "biomaterials": 60, "cell_lines": 49, "samples": 1171}

ROWS = [("samples", "Samples"),
        ("studies", "Studies"),
        ("biomaterials", "Biomaterials"),
        ("cell_lines", "Cell lines")]

# The multiplication sign U+00D7 is the typographically correct glyph and is
# what IOP journals set. A plain letter is used here at the author's request;
# lower case, because an upper-case X reads as a variable rather than as an
# operator. One constant, so the whole figure switches together.
FOLD = "x"


def v3_counts() -> dict[str, int]:
    """
    Current corpus size, computed rather than transcribed.

    The DOI count is taken after URL-decoding. Two records carry the same DOI
    twice over, once percent-encoded and once not - `10.1002%2fadfm.201907436`
    beside `10.1002/adfm.201907436` - so a naive count of distinct strings
    returns 222 for 220 studies.

    The cell-line count excludes the acellular token. The column holds 187
    distinct values, one of which marks a sample printed without cells; that is
    a category, not a cell line, and counting it would inflate the comparison
    against a V2 figure that did not include it.
    """
    df, columns = load_dataset()
    dois = df["DOI"].astype(str).str.strip().map(lambda s: unquote(s).lower())
    cell_line = cfg.CELL_COLS[0]
    cellular = df[df[cell_line] != cfg.ACELLULAR_TOKEN]
    return {"studies": int(dois.nunique()),
            "biomaterials": int(len(columns.biomaterials)),
            "cell_lines": int(cellular[cell_line].nunique()),
            "samples": int(len(df))}


def draw(v3: dict[str, int]):
    ms.apply()
    fig, ax = ms.figure(width="onehalf", height=2.6)

    x = np.arange(len(ROWS), dtype=float)
    w = 0.34
    v2_h = [1.0] * len(ROWS)
    v3_h = [v3[key] / V2[key] for key, _ in ROWS]

    b2 = ax.bar(x - w / 2, v2_h, w, color=ms.SLATE, label="MLATE V2",
                zorder=2)
    b3 = ax.bar(x + w / 2, v3_h, w, color=ms.NAVY, label="MLATE V3", zorder=2)

    for bar, (key, _) in zip(b2, ROWS):
        ax.annotate(f"{V2[key]:,}", (bar.get_x() + bar.get_width() / 2, 1.0),
                    textcoords="offset points", xytext=(0, 3),
                    ha="center", va="bottom", size=6.5, color=ms.MUTED)
    for bar, (key, _), h in zip(b3, ROWS, v3_h):
        # Count and fold change stacked above the bar. Two lines, so the offset
        # of the upper one must exceed the rendered height of the lower.
        ax.annotate(f"{v3[key]:,}", (bar.get_x() + bar.get_width() / 2, h),
                    textcoords="offset points", xytext=(0, 3),
                    ha="center", va="bottom", size=7.0, color=ms.TEXT)
        ax.annotate(f"{h:.1f}{FOLD}", (bar.get_x() + bar.get_width() / 2, h),
                    textcoords="offset points", xytext=(0, 12),
                    ha="center", va="bottom", size=7.0, color=ms.NAVY)

    # The V2 baseline drawn across the panel, so "relative to V2" is a line a
    # reader can see rather than a convention stated only in the axis label.
    ax.axhline(1.0, color=ms.GRIDLINE, lw=0.8, zorder=1)

    ax.set_xticks(x, [label for _, label in ROWS])
    ax.set_ylabel(f"size relative to MLATE V2")
    ax.set_ylim(0, max(v3_h) * 1.30)
    ax.set_yticks([0, 1, 2, 3, 4], ["0", "1{}".format(FOLD), f"2{FOLD}",
                                    f"3{FOLD}", f"4{FOLD}"])
    ms.grid_axis(ax, "y")
    ms.despine(ax)
    ax.legend(loc="upper left", frameon=False, handlelength=1.1,
              handletextpad=0.5, borderaxespad=0.2, fontsize=6.8)

    fig.tight_layout()
    return fig


def main() -> None:
    v3 = v3_counts()
    print("V2 -> V3")
    for key, label in ROWS:
        print(f"  {label:14s} {V2[key]:>6,} -> {v3[key]:>6,}   "
              f"{v3[key] / V2[key]:.2f}{FOLD}")
    fig = draw(v3)
    for path in ms.save(fig, "fig1_dataset_growth", step="01_data"):
        print(f"-> {path}")


if __name__ == "__main__":
    main()
