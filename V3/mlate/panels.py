from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from mlate import style as ms

HEADLINE = "weighted_f1"


def _heat(ax, frame: pd.DataFrame, cols: list[str], annotate: bool,
          tag: str, title: str):
    data = frame[cols].to_numpy(dtype=float)
    n_rows = data.shape[0]
    vsize = 5.2 if n_rows <= 12 else (4.6 if n_rows <= 22 else 4.2)
    ax.imshow(np.clip(data, 0.0, 1.0), cmap=ms.SEQ_NAVY, vmin=0.0, vmax=1.0,
              aspect="auto", origin="lower", interpolation="nearest")

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isnan(v):
                ax.add_patch(Rectangle((j - .5, i - .5), 1, 1,
                                       facecolor=ms.MISSING, lw=0))
                if annotate:
                    ax.text(j, i, "n/a", ha="center", va="center",
                            size=vsize - 0.3, color=ms.MUTED)
                continue
            if v < 0:
                ax.add_patch(Rectangle((j - .5, i - .5), 1, 1,
                                       facecolor=ms.RUST, lw=0))
            if annotate:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        size=vsize, weight="bold",
                        color="white" if (v > .62 or v < 0) else ms.TEXT)

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([ms.METRIC_LABELS[c] for c in cols], rotation=90,
                       size=6)
    ax.set_yticks([])
    ax.set_title(title, size=6.6, color=ms.MUTED, pad=5)
    ax.text(-0.02, 1.0, tag, transform=ax.transAxes, size=8.5,
            weight="bold", ha="right", va="bottom", color=ms.TEXT)
    ax.grid(False)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)


def benchmark_panel(board: pd.DataFrame, *, families: pd.Series | None = None,
                    baselines: dict | None = None,
                    annotate_heat: bool = True,
                    sort_protocol: str = "random",
                    row_height: float = 0.155,
                    title: str = "", subtitle: str = ""):
    cols = [c for c in ms.metric_columns() if c in board.columns]
    wide = board.pivot_table(index="model", columns="protocol",
                             values=HEADLINE)
    if sort_protocol not in wide.columns:
        sort_protocol = wide.columns[0]
    order = wide.sort_values(sort_protocol, ascending=True).index.tolist()
    n = len(order)

    fam_present = []
    if families is not None:
        fam_present = [f for f in ms.FAMILY_COLORS
                       if f in set(families.reindex(order).dropna())]

    legend_rows = 1 + (len(fam_present) > 0)
    fig_h = max(2.6, n * row_height + 1.25 + 0.28 * legend_rows)
    fig = ms.plt.figure(figsize=(ms.WIDTHS["double"], fig_h))

    bottom = (0.34 + 0.20 * legend_rows) / fig_h
    top = 1 - (0.78 if title else 0.34) / fig_h
    gs = fig.add_gridspec(1, 3, width_ratios=[2.25, 1.42, 1.42],
                          wspace=0.07, left=0.235, right=0.985,
                          top=top, bottom=bottom)

    ax = fig.add_subplot(gs[0, 0])
    y = np.arange(n)
    rnd = wide.reindex(order).get("random", pd.Series(np.nan, index=order))
    doi = wide.reindex(order).get("doi", pd.Series(np.nan, index=order))
    rnd, doi = rnd.to_numpy(dtype=float), doi.to_numpy(dtype=float)

    for i in range(n):
        if np.isfinite(rnd[i]) and np.isfinite(doi[i]):
            ax.plot([doi[i], rnd[i]], [i, i], color=ms.GRIDLINE, lw=1.0,
                    zorder=1, solid_capstyle="round")
    ax.scatter(doi, y, s=21, color=ms.PROTOCOL_COLORS["doi"], zorder=3,
               linewidths=0)
    ax.scatter(rnd, y, s=21, color=ms.PROTOCOL_COLORS["random"], zorder=4,
               linewidths=0)

    for i in range(n):
        a, b = rnd[i], doi[i]
        rnd_right = (not np.isfinite(b)) or a >= b
        if np.isfinite(a):
            dx, ha = (4.5, "left") if rnd_right else (-4.5, "right")
            ax.annotate(f"{a:.3f}", (a, i), xytext=(dx, 0),
                        textcoords="offset points", va="center", ha=ha,
                        size=5.1, color=ms.PROTOCOL_COLORS["random"],
                        path_effects=ms.halo(1.8), zorder=6)
        if np.isfinite(b):
            dx, ha = (-4.5, "right") if rnd_right else (4.5, "left")
            ax.annotate(f"{b:.3f}", (b, i), xytext=(dx, 0),
                        textcoords="offset points", va="center", ha=ha,
                        size=5.1, color=ms.PROTOCOL_COLORS["doi"],
                        path_effects=ms.halo(1.8), zorder=6)

    for proto, value in (baselines or {}).items():
        ax.axvline(value, color=ms.PROTOCOL_COLORS.get(proto, ms.RUST),
                   lw=0.9, ls=(0, (3, 2)), zorder=2, alpha=0.75)

    ax.set_yticks(y)
    ax.set_yticklabels([ms.short(m) for m in order], size=6)
    ax.set_ylim(-0.7, n - 0.3)
    finite = np.concatenate([rnd[np.isfinite(rnd)], doi[np.isfinite(doi)]])
    if finite.size:
        lo, hi = float(finite.min()), float(finite.max())
        pad = max(0.05, (hi - lo) * 0.17)
        ax.set_xlim(lo - pad, hi + pad)
    ax.set_xlabel("weighted F1", size=7)
    ms.grid_axis(ax, "x")
    ms.despine(ax, keep=("bottom",))
    ax.tick_params(axis="y", length=0, pad=11 if families is not None
                   else 2)
    ax.set_title("weighted F1 by validation protocol", size=6.6,
                 color=ms.MUTED, pad=5)
    ax.text(-0.315, 1.0, "A", transform=ax.transAxes, size=8.5,
            weight="bold", ha="left", va="bottom", color=ms.TEXT)

    if families is not None:
        fam = families.reindex(order)
        rug = ax.inset_axes([-0.026, 0.0, 0.017, 1.0],
                            transform=ax.transAxes)
        for i, f in enumerate(fam):
            rug.add_patch(Rectangle(
                (0, i - 0.5), 1, 1,
                facecolor=ms.FAMILY_COLORS.get(f, ms.SLATE), lw=0))
        rug.set_xlim(0, 1)
        rug.set_ylim(-0.7, n - 0.3)
        rug.set_xticks([])
        rug.set_yticks([])
        rug.grid(False)
        for side in ("top", "right", "left", "bottom"):
            rug.spines[side].set_visible(False)

    for k, proto in enumerate(("random", "doi")):
        if proto not in set(board["protocol"]):
            continue
        axh = fig.add_subplot(gs[0, 1 + k])
        block = board[board["protocol"] == proto].set_index("model")
        _heat(axh, block.reindex(order), cols, annotate_heat, "BC"[k],
              ms.PROTOCOL_LABELS[proto])

    handles = [Line2D([], [], marker="o", ls="none", markersize=4.2,
                      markerfacecolor=ms.PROTOCOL_COLORS[p],
                      markeredgecolor="none", label=ms.PROTOCOL_LABELS[p])
               for p in ("random", "doi") if p in set(board["protocol"])]
    if baselines:
        handles.append(Line2D([], [], color=ms.MUTED, lw=0.9, ls=(0, (3, 2)),
                              label="majority-class baseline (per protocol)"))
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, (0.06 + 0.20 * (legend_rows - 1)) / fig_h),
               ncol=len(handles), frameon=False, fontsize=6.4,
               handletextpad=0.4, columnspacing=1.6)

    if fam_present:
        fh = [Line2D([], [], marker="s", ls="none", markersize=4.4,
                     markerfacecolor=ms.FAMILY_COLORS[f],
                     markeredgecolor="none",
                     label=ms.FAMILY_LABELS.get(f, f)) for f in fam_present]
        fig.legend(handles=fh, loc="lower center",
                   bbox_to_anchor=(0.5, 0.02 / fig_h),
                   ncol=min(7, len(fh)), frameon=False, fontsize=6.0,
                   handletextpad=0.35, columnspacing=1.1)

    if title:
        fig.suptitle(title, size=9.5, y=1 - 0.14 / fig_h, va="top")
    if subtitle:
        fig.text(0.5, 1 - 0.36 / fig_h, subtitle, ha="center", va="top",
                 size=6.4, color=ms.MUTED)
    return fig
