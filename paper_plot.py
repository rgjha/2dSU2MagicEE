from pathlib import Path
import json
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


try:
    HERE = Path(__file__).resolve().parent
except NameError:
    HERE = Path.cwd()

# v2: revised-code data (corrected plaquette sign, E^2 per link), m = 1
DATA_DIR = HERE / "data"
OUT_DIR = HERE / "figures"
MASS_TAG = "m1"


USE_TEX = True
if USE_TEX:
    try:
        from matplotlib.texmanager import TexManager
        TexManager().make_dvi(r"$x$", 10)
    except Exception:
        print("[warn] LaTeX unavailable -- falling back to mathtext")
        USE_TEX = False

def bf(s):
    """Bold text that works with or without usetex."""
    return rf"\textbf{{{s}}}" if USE_TEX else rf"$\bf{{{s}}}$"

G_STAR = 1


_p = matplotlib.colormaps["plasma"]
colors = {
    "E2":          _p(0.10),   # deep indigo
    "B2":          _p(0.45),   # magenta
    "ReBp":        _p(0.72),   # orange
    "4F":          _p(0.25),   # purple
    "logbound":    _p(0.25),   # tighter bound -log2(1-4F), same accent as 4F
    "M2_NL":       _p(0.60),   # salmon
    "M2":          _p(0.00),   # total SRE magic -- dark indigo, 4x4 only
    "eps":         _p(0.15),   # violet
    "S_don_total": _p(0.52),   # rose
}
GSTAR_COLOR = "#2A0593"


# Bond dimension is encoded by line style AND marker fill together:
#   highest chi -> solid  + filled,
#   middle chi  -> dashed + half-filled  (only used when there are 3 chi),
#   lowest chi  -> dotted + empty.
MS = 6.0          # marker size
LW = 2.0
MEW = 1.6


def chi_style(rank, n):
    """(line style, marker fill) by bond-dimension rank (0 = lowest chi)."""
    if rank == n - 1:
        return "-", "full"     # highest chi -> solid + filled
    if rank == 0:
        return ":", "none"     # lowest chi  -> dotted + empty
    return "--", "left"        # middle chi  -> dashed + half-filled


def fill_kwargs(fs, color):
    if fs == "none":
        return dict(fillstyle="none", markerfacecolor="none",
                    markeredgecolor=color, markeredgewidth=MEW)
    if fs == "full":
        return dict(fillstyle="full", markerfacecolor=color,
                    markeredgecolor=color, markeredgewidth=MEW)
    return dict(fillstyle="left", markerfacecolor=color,
                markerfacecoloralt="none", markeredgecolor=color,
                markeredgewidth=MEW)

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
    "lines.linewidth": LW,
    "lines.markersize": MS,
    "savefig.dpi": 350,
})
plt.rc("font", size=21)
plt.rc("axes", titlesize=23, labelsize=24)
plt.rc("xtick", labelsize=16)
plt.rc("ytick", labelsize=16)
plt.rc("legend", fontsize=16)


# ----------------------------------------------------------------------
# Tighter lower bound: -log2(1-4F).  Derived from the stored 4F column,
# so no new data is needed.  Finite because 4F < 1 for every row.
# ----------------------------------------------------------------------
def add_logbound(data):
    """Lower bound of the bracket, max{-log2(1-4F), M2(psi~) - 4}; the plotted
    M2_NL column is M2(psi~).  'antiflat_only' records whether the anti-flatness
    term attains the max at every point (it does whenever M2(psi~) < 4)."""
    for x in data:
        f4 = x["4F"]
        af = -math.log2(1.0 - f4) if f4 < 1.0 else float("nan")
        x["logbound"] = max(af, x["M2_NL"] - 4.0)
        x["antiflat_only"] = af >= x["M2_NL"] - 4.0


def style_axis(ax, xlabel=False):
    ax.set_xscale("log")
    if xlabel:
        ax.set_xlabel(r"$g^2$", labelpad=4)
    # Cap the y-tick count and prune the extreme ticks so short, stacked
    # panels do not crowd (top tick of one panel vs bottom tick of the next).
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
    ax.grid(which="major", linestyle="-", alpha=0.55)
    ax.grid(which="minor", linestyle=":", alpha=0.35)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", which="major", length=6.5, width=1.15,
                   direction="in", top=True, right=True)
    ax.tick_params(axis="both", which="minor", length=3.5, width=0.95,
                   direction="in", top=True, right=True)
    for sp in ax.spines.values():
        sp.set_linewidth(1.05); sp.set_color("0.22")
    ax.margins(x=0.20)


def draw_gstar_line(ax):
    ax.axvline(G_STAR, color=GSTAR_COLOR, ls="--", lw=1.75, zorder=1)


def draw_gstar_label(ax):
    ax.text(G_STAR, 0.94, rf"$g_\star^2 \approx {G_STAR:.0f}$",
            transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=17,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="#FEFDF8",
                      edgecolor=GSTAR_COLOR, alpha=0.92),
            clip_on=True)


def panel_letter(ax, s, frac=0.10):
    """Add headroom at the top, then drop the bold panel letter inside it."""
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + frac * (hi - lo))
    ax.text(0.022, 0.95, bf(s), transform=ax.transAxes,
            ha="left", va="top", fontsize=20, zorder=6,
            bbox=dict(boxstyle="square,pad=0.14", facecolor="#FBFAF4",
                      edgecolor="none"))


def series(ax, g, data, key, marker, rank, n, label=None, scale=1.0):
    c = colors[key]
    ls, fs = chi_style(rank, n)
    ax.plot(g, [scale * r[key] for r in data],
            color=c, marker=marker, ls=ls, lw=LW, markersize=MS,
            label=(label if label is not None else "_nolegend_"),
            **fill_kwargs(fs, c))


#def series_err(ax, g, data, key, err_key, marker, rank, n, color, label=None):
def series_err(ax, g, data, key, err_key, marker, rank, n, color, label=None, scale=1.0):
    """Like series() but with (Monte-Carlo) error bars -- used for M2."""
    ls, fs = chi_style(rank, n)
    #ax.errorbar(g, [r[key] for r in data], yerr=[r[err_key] for r in data],
    ax.errorbar(g, [scale * r[key] for r in data], yerr=[scale * r[err_key] for r in data],
                color=color, marker=marker, ls=ls, lw=LW, markersize=MS,
                capsize=2.8, elinewidth=1.0,
                label=(label if label is not None else "_nolegend_"),
                **fill_kwargs(fs, color))


MASSES = [("m1", r"$m=1$"), ("m0.2", r"$m=0.2$")]


def load_set(size, chi, tag):
    path = DATA_DIR / f"nonlocal_magic_{size}_D{chi}_{tag}.json"
    if not path.exists():
        print(f"[warn] {path.name} not found -- panel drawn without it")
        return None
    with open(path) as f:
        d = sorted(json.load(f), key=lambda x: x["g2"])
    add_logbound(d)
    return d


def make_figure(size, chis, outstem):
    """4x2 grid: rows (a) gauge observables, (b) magic, (c) energy, (d) entropy;
    left column m = 1, right column m = 0.2 (panels e-h).  Rows share the y-axis
    so the two masses can be compared directly; all panels share the x-axis."""
    n = len(chis)
    rank = {chi: i for i, chi in enumerate(chis)}   # 0 = lowest chi
    cols = [[(chi, load_set(size, chi, tag)) for chi in chis] for tag, _ in MASSES]
    cols = [[(chi, d) for chi, d in c if d is not None] for c in cols]
    alldata = [d for c in cols for _, d in c]
    antiflat_only = all(x["antiflat_only"] for d in alldata for x in d)

    # double-column figure: twice the single-column width, same panel height,
    # so the on-page font size matches the earlier one-column figures.
    fig, axs = plt.subplots(4, 2, figsize=(14.0, 10.4), sharex=True,
                            sharey="row",
                            gridspec_kw=dict(hspace=0.06, wspace=0.035))

    if antiflat_only:
        lb_lab = r"$-\log_2(1-4F)$"
    else:
        lb_lab = r"$\max\{-\log_2(1-4F),\,M_2(\tilde\psi)-4\}$"
    nl_lab = r"$M_2(|\tilde\psi\rangle)$"   # CB-representative SRE: upper bound on M2^NL

    for j, datasets in enumerate(cols):
        top = (j == 0)          # legends only in the left column
        # (a)/(e) gauge observables
        ax = axs[0, j]
        for key, mk, lab in [("E2", "o", r"$\langle E^2\rangle$"),
                             ("ReBp", "^", r"$\mathrm{Re}\langle B_p\rangle$")]:
            for chi, d in datasets:
                series(ax, [r["g2"] for r in d], d, key, mk, rank[chi], n,
                       label=(lab if (top and chi == chis[-1]) else None))
        # (b)/(f) magic: lower bound, M2^NL, and M2/5 (4x4 only)
        ax = axs[1, j]
        for key, mk, lab in [("logbound", "o", lb_lab), ("M2_NL", "D", nl_lab)]:
            for chi, d in datasets:
                series(ax, [r["g2"] for r in d], d, key, mk, rank[chi], n,
                       label=(lab if (top and chi == chis[-1]) else None))
        if size == "4x4":
            for chi, d in datasets:
                series_err(ax, [r["g2"] for r in d], d, "M2", "M2_err", "^",
                           rank[chi], n, colors["M2"], scale=1/5,
                           label=(r"$M_2/5$" if (top and chi == chis[-1]) else None))
        # (c)/(g) energy density, (d)/(h) entanglement entropy
        for row, key in [(2, "eps"), (3, "S_don_total")]:
            for chi, d in datasets:
                series(axs[row, j], [r["g2"] for r in d], d, key, "o", rank[chi], n)
        for row in range(4):
            style_axis(axs[row, j], xlabel=(row == 3))
            draw_gstar_line(axs[row, j])
        draw_gstar_label(axs[0, j])
        axs[0, j].set_title(MASSES[j][1], fontsize=22, pad=8)

    # shared-row y-limits: headroom once per row, then letters in both columns
    axs[1, 0].set_ylim(bottom=0)
    for row, frac in enumerate([0.30, 0.10, 0.10, 0.30]):
        lo, hi = axs[row, 0].get_ylim()
        axs[row, 0].set_ylim(lo, hi + frac * (hi - lo))
        for j in range(2):
            axs[row, j].text(0.022, 0.95, bf("(" + "abcdefgh"[4 * j + row] + ")"),
                             transform=axs[row, j].transAxes, ha="left", va="top",
                             fontsize=20, zorder=6,
                             bbox=dict(boxstyle="square,pad=0.14",
                                       facecolor="#FBFAF4", edgecolor="none"))
    axs[2, 0].set_ylabel(r"$\varepsilon$", rotation=0, labelpad=15)
    axs[3, 0].set_ylabel(r"$S$", rotation=0, labelpad=20)

    legkw = dict(fontsize=15, framealpha=0.92, facecolor="white",
                 edgecolor="0.65", handlelength=2.2, labelspacing=0.3,
                 borderpad=0.3)
    axs[0, 0].legend(loc="center right", **legkw)
    axs[1, 0].legend(loc="upper right", **legkw)

    gall = [r["g2"] for d in alldata for r in d]
    axs[-1, 0].set_xlim(min(gall) / 1.3, max(gall) * 1.3)

    fig.subplots_adjust(left=0.075, right=0.985, top=0.965, bottom=0.075)
    fig.savefig(OUT_DIR / f"{outstem}.pdf")
    plt.close(fig)
    print(f"wrote {OUT_DIR / outstem}.pdf   ({size}, chi={chis}, "
          f"lower bound = -log2(1-4F) everywhere: {antiflat_only})")


if __name__ == "__main__":
    make_figure("4x4", [32, 64, 96],  "4x4_res")
    make_figure("6x6", [64, 128],     "6x6_res")
    make_figure("8x8", [64, 128],     "8x8_res")
