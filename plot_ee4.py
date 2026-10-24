#!/usr/bin/env python3
"""
Donnelly entanglement-entropy decomposition on a 3x2 grid (v2):
    rows = 4x4 (D = 96), 6x6 (D = 128), 8x8 (D = 128);  columns = m = 1, m = 0.2.
    (Output file name kept as EE_decomposition_2panel.pdf so the tex is unchanged.)

In each panel the three gauge-invariant contributions
    S = S_classical + S_quantum + S_edge
are drawn as DASHED lines of different width (classical widest, edge medium,
quantum thinnest); the SUM (S_don_total) is the SOLID curve with markers.
S_quantum is tiny, so it is plotted x10 (shown in its label) for visibility.
Colors are the plasma scheme / marker size shared with final_plot46.py.

Layout rules matched to the 4x1 figure:
    - single shared x-axis, g^2 label on the BOTTOM panel only,
    - no titles; the (a)/(b) letter (with lattice, chi) sits in an in-panel box,
    - COMPACT aspect: width fixed, height shrunk so panels are short (on-page
      font size is set by the width, so text stays the same and readable),
    - capped/pruned y-ticks so the short stacked panels do not crowd.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

HERE = Path(__file__).resolve().parent
# v2: revised-code data (corrected plaquette sign, E^2 per link), m = 1
DATA_DIR = HERE / "data"
OUT_DIR = HERE / "figures"

USE_TEX = True
if USE_TEX:
    try:
        from matplotlib.texmanager import TexManager
        TexManager().make_dvi(r"$x$", 10)
    except Exception:
        print("[warn] LaTeX unavailable -- falling back to mathtext")
        USE_TEX = False


# Grid of panels: rows = lattices (largest chi), columns = masses.
ROWS = [("4x4", 96, r"4\times4,\ \chi=96"),
        ("6x6", 128, r"6\times6,\ \chi=128"),
        ("8x8", 128, r"8\times8,\ \chi=128")]
MASSES = [("m1", r"$m=1$"), ("m0.2", r"$m=0.2$")]

# Plasma color scheme, shared with final_plot46.py.
_p = matplotlib.colormaps["plasma"]

# term -> (color, dash pattern, linewidth, legend label, scale).  Solid total on
# top.  S_quantum is scaled x10 (shown in its label) so it is visible.
TERMS = {
    "S_don_classical": {"color": _p(0.10), "dashes": (7, 2),     "lw": 3.6,
                        "label": r"$S_{\mathrm{classical}}$", "scale": 1.0},
    "S_don_edge":      {"color": _p(0.72), "dashes": (4, 2),     "lw": 2.6,
                        "label": r"$S_{\mathrm{edge}}$", "scale": 1.0},
    "S_don_quantum":   {"color": _p(0.30), "dashes": (1.4, 1.6), "lw": 2.0,
                        "label": r"$10\,S_{\mathrm{quantum}}$", "scale": 10.0},
}
TOTAL = {"key": "S_don_total", "color": _p(0.52), "label": r"$S_{\mathrm{total}}$"}

MS = 10.0          # marker size, shared with final_plot46.py

G_STAR = 1.0

# ----------------------------------------------------------------------
# Publication styling
# ----------------------------------------------------------------------
plt.rcParams.update({
    "text.usetex": USE_TEX,
    "font.family": "serif",
    "figure.facecolor": "white",
    "axes.facecolor": "#FBFAF4",
    "axes.edgecolor": "0.22",
    "axes.linewidth": 1.05,
    "grid.color": "0.82",
    "grid.linewidth": 0.65,
    "grid.alpha": 0.60,
    "legend.frameon": True,
    "legend.fancybox": True,
    "lines.linewidth": 2.15,
    "lines.markersize": MS,
    "savefig.dpi": 350,
})
plt.rc("font", size=21)
plt.rc("axes", titlesize=23, labelsize=24)
plt.rc("xtick", labelsize=20)
plt.rc("ytick", labelsize=20)
plt.rc("legend", fontsize=20)

GRID_KW = dict(which="major", linestyle="-", alpha=0.55)
MINOR_GRID_KW = dict(which="minor", linestyle=":", alpha=0.35)
GSTAR_COLOR = "#882255"
GSTAR_FONTSIZE = 18


def bf(s):
    return rf"\textbf{{{s}}}" if USE_TEX else rf"$\bf{{{s}}}$"


def style_axis(ax, xlabel=False):
    ax.set_xscale("log")
    if xlabel:
        ax.set_xlabel(r"$g^2$", labelpad=4)
    # Cap the y-tick count and prune the extreme ticks so short, stacked
    # panels do not crowd (top tick of one panel vs bottom tick of the next).
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
    ax.grid(**GRID_KW)
    ax.grid(**MINOR_GRID_KW)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", which="major", length=6.5, width=1.15,
                   direction="in", top=True, right=True)
    ax.tick_params(axis="both", which="minor", length=3.5, width=0.95,
                   direction="in", top=True, right=True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.05)
        spine.set_color("0.22")
    ax.margins(x=0.02)


def draw_gstar_line(ax):
    ax.axvline(G_STAR, color=GSTAR_COLOR, ls="--", lw=1.75, zorder=1)


def draw_gstar_label(ax):
    ax.text(G_STAR, 0.94, rf"$g_\star={G_STAR:.2f}$",
            transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=GSTAR_FONTSIZE,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="#FEFDF8",
                      edgecolor=GSTAR_COLOR, alpha=0.92), clip_on=False)


def panel_box(ax, letter, lattice, frac=0.30):
    """Add headroom at top, then drop the boxed '(a) lattice' label inside it."""
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + frac * (hi - lo))
    label = rf"$\mathbf{{({letter})}}\ \ {lattice}$"
    ax.text(0.025, 0.95, label, transform=ax.transAxes,
            ha="left", va="top", fontsize=20, zorder=6,
            bbox=dict(boxstyle="square,pad=0.16", facecolor="#FBFAF4",
                      edgecolor="none"))


def load_rows(path):
    with open(path) as f:
        rows = json.load(f)
    rows = [r for r in rows if np.isfinite(r.get(TOTAL["key"], np.nan))]
    rows.sort(key=lambda r: r["g2"])
    g = np.array([r["g2"] for r in rows], float)
    data = {"g2": g}
    for key in list(TERMS) + [TOTAL["key"]]:
        data[key] = np.array([r.get(key, np.nan) for r in rows], float)
    return data


# 3x2 grid (double-column figure): rows = lattices, columns = masses, rows
# share the y-axis; panel letters run down the left column, then the right.
fig, axs = plt.subplots(len(ROWS), 2, figsize=(14.0, 8.0), sharex=True,
                        sharey="row", gridspec_kw=dict(hspace=0.10, wspace=0.035))

gmin, gmax = np.inf, -np.inf
for j, (tag, mlabel) in enumerate(MASSES):
    for i, (size, chi, latt) in enumerate(ROWS):
        ax = axs[i, j]
        path = DATA_DIR / f"nonlocal_magic_{size}_D{chi}_{tag}.json"
        if path.exists():
            d = load_rows(path)
            g = d["g2"]
            gmin, gmax = min(gmin, g.min()), max(gmax, g.max())
            for key, sty in TERMS.items():
                ax.plot(g, sty.get("scale", 1.0) * d[key], ls="--",
                        dashes=sty["dashes"], lw=sty["lw"], color=sty["color"],
                        zorder=3)
            ax.plot(g, d[TOTAL["key"]], "-o", color=TOTAL["color"], lw=2.2,
                    ms=MS, mfc=TOTAL["color"], mec="white", mew=0.7, zorder=5)
        else:
            print(f"[warn] {path.name} not found -- empty panel")
        style_axis(ax, xlabel=(i == len(ROWS) - 1))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="upper"))
    axs[0, j].set_title(mlabel, fontsize=22, pad=8)

for i, (size, chi, latt) in enumerate(ROWS):
    axs[i, 0].set_ylim(bottom=0)
    axs[i, 0].set_ylabel(r"$S$", rotation=0, labelpad=20)
    lo, hi = axs[i, 0].get_ylim()
    axs[i, 0].set_ylim(lo, hi + 0.30 * (hi - lo))      # headroom, once per row
    for j in range(2):
        letter = "abcdef"[len(ROWS) * j + i]
        axs[i, j].text(0.025, 0.95, rf"$\mathbf{{({letter})}}\ \ {latt}$",
                       transform=axs[i, j].transAxes, ha="left", va="top",
                       fontsize=20, zorder=6,
                       bbox=dict(boxstyle="square,pad=0.16", facecolor="#FBFAF4",
                                 edgecolor="none"))

# explicit shared x-limits with padding so edge markers are not clipped
axs[-1, 0].set_xlim(gmin / 1.3, gmax * 1.3)

# single legend (term color/style) in panel (a)
handles = [Line2D([], [], color=TOTAL["color"], ls="-", lw=2.2, marker="o",
                  ms=MS, mec="white", mew=0.7, label=TOTAL["label"])]
handles += [Line2D([], [], color=s["color"], ls="--", dashes=s["dashes"],
                   lw=s["lw"], label=s["label"]) for s in TERMS.values()]
axs[0, 0].legend(handles=handles, loc="upper right", fontsize=14, frameon=True,
                 framealpha=0.92, facecolor="white", edgecolor="0.65",
                 handlelength=2.2, borderpad=0.3, labelspacing=0.25)

fig.align_ylabels(axs[:, 0])
fig.subplots_adjust(left=0.07, right=0.985, bottom=0.09, top=0.955)

out_pdf = OUT_DIR / "EE_decomposition_2panel.pdf"
fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.03)
plt.close(fig)
print(out_pdf)
