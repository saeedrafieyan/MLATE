from __future__ import annotations

import sys
from pathlib import Path
from urllib.parse import unquote

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from mlate import config as cfg
from mlate import style as ms
from mlate.dataset import load_dataset

V2 = {"studies": 88, "biomaterials": 60, "cell_lines": 49, "samples": 1171}

ROWS = [("samples", "Samples"),
        ("studies", "Studies"),
        ("biomaterials", "Biomaterials"),
        ("cell_lines", "Cell lines")]

FOLD = "x"


def v3_counts() -> dict[str, int]:
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
        ax.annotate(f"{v3[key]:,}", (bar.get_x() + bar.get_width() / 2, h),
                    textcoords="offset points", xytext=(0, 3),
                    ha="center", va="bottom", size=7.0, color=ms.TEXT)
        ax.annotate(f"{h:.1f}{FOLD}", (bar.get_x() + bar.get_width() / 2, h),
                    textcoords="offset points", xytext=(0, 12),
                    ha="center", va="bottom", size=7.0, color=ms.NAVY)

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
