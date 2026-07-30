#!/usr/bin/env python3
"""
Generate every figure in the manuscript from the artefacts on disk.

    venv/bin/python paper/tex/make_figures.py

No number in this file is typed by hand. Every value plotted is read from a JSON
artefact under pipeline_out/ or paper/, and each function prints the artefact it
read so a reviewer can trace any mark on any panel back to a file.

Output: paper/tex/figures/fig1_collapse.pdf ... fig5_case_study.pdf
Vector PDF, fonts embedded as Type-42 (TrueType) subsets.

Palette: Okabe-Ito colour-blind-safe. No red/green pair is used to carry meaning.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# --------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------- #

HERE = Path(__file__).resolve().parent            # paper/tex
REPO = HERE.parent.parent                          # repo root
OUT = HERE / "figures"
OUT.mkdir(parents=True, exist_ok=True)

_SOURCES: list[str] = []


def load(relpath: str, panel: str):
    """Read a JSON artefact and record it as the provenance of `panel`."""
    p = REPO / relpath
    if not p.exists():
        sys.exit(f"MISSING ARTEFACT: {p}\n  needed for: {panel}")
    with p.open() as fh:
        obj = json.load(fh)
    line = f"  {panel:<34s} <- {relpath}"
    print(line)
    _SOURCES.append(line)
    return obj


# --------------------------------------------------------------------------- #
# style
# --------------------------------------------------------------------------- #

# Okabe-Ito
BLACK = "#000000"
ORANGE = "#E69F00"
SKY = "#56B4E9"
GREEN = "#009E73"
YELLOW = "#F0E442"
BLUE = "#0072B2"
VERM = "#D55E00"
PURPLE = "#CC79A7"
GREY = "#8C8C8C"
LIGHTGREY = "#D9D9D9"

# The two workhorse contrasts. Blue/orange and blue/purple are both safe under
# deuteranopia, protanopia and tritanopia; no red/green pair carries meaning.
C_SLICE = BLUE
C_PATIENT = ORANGE
C_NULL = GREY

plt.rcParams.update(
    {
        "pdf.fonttype": 42,          # embed TrueType subsets, not Type-3
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 7.5,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.1,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)

COL_W = 3.35    # single column, inches
FULL_W = 7.0    # double column, inches


def save(fig, name: str):
    path = OUT / name
    fig.savefig(path, format="pdf")
    plt.close(fig)
    print(f"  wrote {path.relative_to(REPO)}")


def panel_tag(ax, letter, x=-0.13, y=1.06):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9.5,
            fontweight="bold", va="bottom", ha="left")


# --------------------------------------------------------------------------- #
# Figure 1 -- the collapse
# --------------------------------------------------------------------------- #

LABEL_ORDER = ["any", "epidural", "intraparenchymal", "intraventricular",
               "subarachnoid", "subdural"]
LABEL_PRETTY = {
    "any": "any",
    "epidural": "epidural",
    "intraparenchymal": "intraparenchymal",
    "intraventricular": "intraventricular",
    "subarachnoid": "subarachnoid",
    "subdural": "subdural",
}


def fig1():
    print("\nFigure 1 -- RSNA ICH positional baseline, slice unit vs patient unit")
    d = load("pipeline_out/trivial_baselines/rsna_ich_unit_collapse.json",
             "Fig 1A + 1B, all marks")

    labs = [d["labels"][k] for k in LABEL_ORDER]
    n = len(labs)
    x = np.arange(n, dtype=float)

    slice_auc = np.array([L["slice_auc"] for L in labs])
    slice_lo = np.array([L["slice_ci_clustered"][0] for L in labs])
    slice_hi = np.array([L["slice_ci_clustered"][1] for L in labs])
    pat_auc = np.array([L["patient_auc_mean_agg"] for L in labs])
    pat_lo = np.array([L["patient_ci_clustered"][0] for L in labs])
    pat_hi = np.array([L["patient_ci_clustered"][1] for L in labs])
    pat_null = np.array([L["within_series_permutation_null"]["patient_mean"]
                         for L in labs])
    gap = np.array([L["collapse_slice_minus_patient"] for L in labs])

    naive_w = np.array([L["slice_ci_naive_WRONG"][1] - L["slice_ci_naive_WRONG"][0]
                        for L in labs])
    clust_w = np.array([L["slice_ci_clustered"][1] - L["slice_ci_clustered"][0]
                        for L in labs])
    ratio = clust_w / naive_w

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(FULL_W, 3.05),
        gridspec_kw={"width_ratios": [1.95, 1.0], "wspace": 0.42},
    )

    # ---- Panel A: the collapse ------------------------------------------- #
    dx = 0.17
    axA.axhline(0.5, color=BLACK, lw=0.7, ls=(0, (4, 3)), zorder=1)
    axA.text(-0.54, 0.5035, "chance", fontsize=6.4, color=BLACK,
             ha="left", va="bottom")

    for i in range(n):
        axA.plot([x[i] - dx, x[i] + dx], [slice_auc[i], pat_auc[i]],
                 color=LIGHTGREY, lw=3.0, solid_capstyle="round", zorder=2)

    axA.errorbar(x - dx, slice_auc, yerr=[slice_auc - slice_lo, slice_hi - slice_auc],
                 fmt="o", ms=5.0, color=C_SLICE, mfc=C_SLICE, mec=C_SLICE,
                 ecolor=C_SLICE, elinewidth=1.4, capsize=2.4, capthick=1.0, zorder=4)
    axA.errorbar(x + dx, pat_auc, yerr=[pat_auc - pat_lo, pat_hi - pat_auc],
                 fmt="s", ms=5.0, color=C_PATIENT, mfc=C_PATIENT, mec=C_PATIENT,
                 ecolor=C_PATIENT, elinewidth=1.4, capsize=2.4, capthick=1.0, zorder=4)

    # patient-level permutation null: one tick per label, at the patient column
    for i in range(n):
        axA.plot([x[i] + dx - 0.115, x[i] + dx + 0.115],
                 [pat_null[i], pat_null[i]],
                 color=C_NULL, lw=1.6, ls="-", zorder=3)

    # annotate the collapse gap
    for i in range(n):
        mid = 0.5 * (slice_auc[i] + pat_auc[i])
        axA.annotate(f"−{gap[i]:.3f}", (x[i], mid), fontsize=6.2,
                     color="#404040", ha="center", va="center",
                     bbox=dict(boxstyle="round,pad=0.10", fc="white",
                               ec="none", alpha=0.9), zorder=5)

    axA.annotate(
        f"patient-level permutation null\n({pat_null[0]:.3f} for 'any')",
        xy=(x[-1] + dx + 0.10, pat_null[-1]),
        xytext=(x[-1] - 0.75, 0.408),
        fontsize=6.3, color=C_NULL, ha="center", va="bottom",
        arrowprops=dict(arrowstyle="-", color=C_NULL, lw=0.7,
                        shrinkA=2, shrinkB=1),
    )

    axA.set_xticks(x)
    axA.set_xticklabels([LABEL_PRETTY[k] for k in LABEL_ORDER],
                        rotation=32, ha="right")
    axA.set_ylabel("AUROC of the zero-image positional baseline")
    axA.set_ylim(0.39, 0.90)
    axA.set_xlim(-0.6, n - 0.35)
    axA.set_title(
        f"{d['n_patients']:,} patients · {d['n_series']:,} series · "
        f"{d['n_slices']:,} slices",
        pad=6,
    )
    handles = [
        Line2D([], [], marker="o", ls="none", ms=5, color=C_SLICE,
               label="slice unit"),
        Line2D([], [], marker="s", ls="none", ms=5, color=C_PATIENT,
               label="patient unit (mean aggregation)"),
        Line2D([], [], color=C_NULL, lw=1.6, label="patient permutation null"),
    ]
    axA.legend(handles=handles, loc="upper left", frameon=False,
               handletextpad=0.6, borderaxespad=0.2)
    panel_tag(axA, "A", x=-0.115)

    # ---- Panel B: what ignoring clustering costs ------------------------- #
    axB.axvline(1.0, color=BLACK, lw=0.7, ls=(0, (4, 3)))
    y = np.arange(n)[::-1]
    axB.barh(y, ratio, height=0.62, color=SKY, edgecolor=BLUE, lw=0.7)
    for yy, r in zip(y, ratio):
        axB.text(r + 0.03, yy, f"{r:.2f}×", va="center", ha="left",
                 fontsize=6.6, color=BLUE)
    axB.set_yticks(y)
    axB.set_yticklabels([LABEL_PRETTY[k] for k in LABEL_ORDER])
    axB.set_xlabel("clustered ÷ naive CI width", fontsize=7.2)
    axB.set_xlim(0, max(ratio) * 1.32)
    axB.set_ylim(-0.75, n - 0.35)
    axB.set_title("what a slice-resampled\ninterval leaves out", pad=6, fontsize=7.8)
    panel_tag(axB, "B", x=-0.46)

    save(fig, "fig1_collapse.pdf")

    print(f"    slice 'any'   {slice_auc[0]:.4f} [{slice_lo[0]:.4f}, {slice_hi[0]:.4f}]")
    print(f"    patient 'any' {pat_auc[0]:.4f} [{pat_lo[0]:.4f}, {pat_hi[0]:.4f}]")
    print(f"    collapse gaps {gap.min():.3f} to {gap.max():.3f}")
    print(f"    CI width ratio {ratio.min():.2f}x to {ratio.max():.2f}x")


# --------------------------------------------------------------------------- #
# Figure 2 -- trivial fraction across benchmarks
# --------------------------------------------------------------------------- #

def fig2():
    print("\nFigure 2 -- trivial fraction across benchmarks")
    d = load("paper/trivial_fraction_distribution.json", "Fig 2, every row")
    picai = load("pipeline_out/trivial_baselines/picai_case_level.json",
                 "Fig 2, PI-CAI positional-only")

    rows = [r for r in d["rows"] if r["primary_comparator"]]
    rows.sort(key=lambda r: r["trivial_fraction"])

    summ = {s["set"]: s for s in d["summaries"]}
    pr = summ["peer-reviewed comparators, strongest per benchmark-arm"]

    # the PI-CAI positional baseline, at exactly chance -> fraction 0
    pos = picai["evaluations"]["official_split"]["baselines"]["positional_20bin"]
    picai_row = next(r for r in rows if r["benchmark"] == "PI-CAI")
    picai_pos_auc = pos["patient_auc"]
    picai_pos_tf = ((picai_pos_auc - picai_row["chance"])
                    / (picai_row["published"] - picai_row["chance"]))

    n = len(rows)
    fig, ax = plt.subplots(figsize=(FULL_W, 0.315 * n + 1.35))

    y = np.arange(n)[::-1].astype(float)

    # peer-reviewed IQR band and median, drawn behind everything
    ax.axvspan(pr["q25"], pr["q75"], color=LIGHTGREY, alpha=0.55, lw=0, zorder=0)
    ax.axvline(pr["median"], color=GREY, lw=1.0, ls="-", zorder=1)
    ax.axvline(0.0, color=BLACK, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.axvline(1.0, color=BLACK, lw=0.8, ls=(0, (1, 2)), zorder=1)

    labels = []
    for yy, r in zip(y, rows):
        preprint = r["peer_reviewed"].startswith("preprint")
        col = PURPLE if preprint else BLUE
        marker = "D" if preprint else "o"
        mfc = "white" if preprint else col
        tf = r["trivial_fraction"]
        ci = r["trivial_fraction_ci"]
        if ci is not None:
            ax.errorbar([tf], [yy], xerr=[[tf - ci[0]], [ci[1] - tf]],
                        fmt="none", ecolor=col, elinewidth=1.5,
                        capsize=2.6, capthick=1.0, zorder=3)
        ax.plot([tf], [yy], marker=marker, ms=6.0, color=col, mfc=mfc,
                mec=col, mew=1.4, ls="none", zorder=4)
        if ci is None:
            ax.annotate("no interval available for this metric",
                        (tf + 0.035, yy), fontsize=6.0, color=col,
                        va="center", ha="left")
        arm = r["arm"] if r["arm"] not in ("case level",) else "case level"
        labels.append(f"{r['benchmark']} — {arm}")

    # PI-CAI positional-only, same visual weight, open marker
    ypic = y[[r["benchmark"] for r in rows].index("PI-CAI")]
    ax.plot([picai_pos_tf], [ypic], marker="o", ms=6.0, color=BLACK,
            mfc="white", mec=BLACK, mew=1.4, ls="none", zorder=4)
    ax.plot([picai_pos_tf, picai_row["trivial_fraction"]], [ypic, ypic],
            color=BLACK, lw=0.6, ls=(0, (2, 2)), zorder=2)
    ax.annotate(
        f"positional baseline alone: AUROC exactly {picai_pos_auc:.3f}",
        xy=(picai_pos_tf, ypic), xytext=(picai_pos_tf + 0.05, ypic - 0.55),
        fontsize=6.2, color=BLACK, ha="left", va="center",
        arrowprops=dict(arrowstyle="-", color=BLACK, lw=0.6,
                        shrinkA=0, shrinkB=2),
    )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_ylim(-1.15, n - 0.35)
    ax.set_xlabel("trivial fraction   (best zero-image baseline − chance) "
                  "÷ (published − chance)")
    ax.set_xlim(-0.13, 1.30)

    ax.text(1.0, -1.10, "baseline equals\nthe published system", fontsize=6.2,
            color=BLACK, ha="center", va="bottom")
    ax.text(0.0, -1.10, "baseline at\nchance", fontsize=6.2,
            color=BLACK, ha="center", va="bottom")

    handles = [
        Line2D([], [], marker="o", ls="none", ms=6, color=BLUE,
               label="peer-reviewed comparator"),
        Line2D([], [], marker="D", ls="none", ms=6, color=PURPLE, mfc="white",
               mew=1.4, label="preprint comparator (Rempe et al.)"),
        Line2D([], [], marker="o", ls="none", ms=6, color=BLACK, mfc="white",
               mew=1.4, label="positional baseline alone"),
        mpatches.Patch(facecolor=LIGHTGREY, edgecolor="none",
                       label=f"peer-reviewed IQR (n = {pr['n']})"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False,
              handletextpad=0.6, borderaxespad=0.4)
    ax.set_title("strongest published comparator per benchmark-arm — "
                 f"peer-reviewed median {pr['median']:.3f}, "
                 f"IQR [{pr['q25']:.3f}, {pr['q75']:.3f}]", pad=6)

    save(fig, "fig2_trivial_fraction.pdf")

    luna = next(r for r in rows if r["benchmark"] == "LUNA16")
    print(f"    LUNA16 trivial fraction {luna['trivial_fraction']:.4f} "
          f"(baseline {luna['baseline']} vs random-score reference {luna['chance']})")
    print(f"    PI-CAI positional baseline AUROC {picai_pos_auc:.4f} "
          f"-> trivial fraction {picai_pos_tf:.4f}")
    print(f"    peer-reviewed strongest-per-arm: n={pr['n']} median "
          f"{pr['median']:.4f} IQR [{pr['q25']:.4f}, {pr['q75']:.4f}]")


# --------------------------------------------------------------------------- #
# Figure 3 -- PRISMA flow
# --------------------------------------------------------------------------- #

def _box(ax, x, y, w, h, text, fc="white", ec=BLACK, fontsize=6.8, lw=0.8,
         color=BLACK, style="round,pad=0.008"):
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle=style, linewidth=lw,
        edgecolor=ec, facecolor=fc, mutation_aspect=1.0, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=color, zorder=3, linespacing=1.35)


def _arrow(ax, x0, y0, x1, y1, color=BLACK, lw=0.9):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                shrinkA=0, shrinkB=0, mutation_scale=8),
                zorder=4)


def fig3():
    print("\nFigure 3 -- PRISMA-style flow for the prevalence screen")
    pf = load("paper/screen/analysis/pooled_final.json", "Fig 3, every count")

    f = pf["flow_pre_registered"]
    ident = pf["deduplication"]
    prim = pf["primary_pre_registered"]
    blocks = pf["flow_by_block"]
    ext = pf["extension_rule"]

    frame_meta = load("paper/screen/frame_meta.json", "Fig 3, frame size + SHA")
    sample = load("paper/screen_sample.json", "Fig 3, permutation seed")
    frame_n = frame_meta["esearch_count"]
    frame_sha = frame_meta["frame_sha256"]
    seed = sample["sampling"]["seed"]

    exc1 = f["excluded_by_code_at_stage1"]
    excF = f["excluded_by_code_at_fulltext"]

    def codes(dd, per_line=3):
        items = [f"{k} {v}" for k, v in
                 sorted(dd.items(), key=lambda kv: -kv[1])]
        return "\n".join("   ".join(items[i:i + per_line])
                         for i in range(0, len(items), per_line))

    fig = plt.figure(figsize=(FULL_W, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.55, 1.0], wspace=0.34)
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    LX, LW = 0, 53          # main column
    RX, RW = 55, 45         # exclusion column
    H = 11.0
    H_LAST = H + 3.5

    rows_y = [85, 67, 49, 31, 10]
    heights = [H, H, H, H, H_LAST]

    _box(ax, LX, rows_y[0], LW, H,
         f"Frozen PubMed frame\n{frame_n:,} records   ·   seed {seed}\n"
         f"SHA-256 {frame_sha[:12]}…",
         fc="#EAF2F8", ec=BLUE)

    _box(ax, LX, rows_y[1], LW, H,
         f"Records screened at title / abstract\n"
         f"n = {f['records_screened']}\n"
         f"blocks {', '.join(ext['pre_registered_blocks'])}",
         fc="white")

    _box(ax, LX, rows_y[2], LW, H,
         f"Reports sought for retrieval\nn = {f['reports_sought_for_retrieval']}",
         fc="white")

    _box(ax, LX, rows_y[3], LW, H,
         f"Reports assessed for eligibility\nn = {f['reports_assessed_for_eligibility']}",
         fc="white")

    _box(ax, LX, rows_y[4], LW, H_LAST,
         f"Included and reachable\nn = {f['included_and_reachable']}\n"
         f"eligible-looking denominator\n"
         f"{f['included_and_reachable']} + {f['unreachable_eligibility_unresolved']}"
         f" = {prim['n_eligible']}",
         fc="#EAF2F8", ec=BLUE, lw=1.3)

    for i in range(len(rows_y) - 1):
        _arrow(ax, LX + LW / 2, rows_y[i],
               LX + LW / 2, rows_y[i + 1] + heights[i + 1])

    # exclusion boxes
    _box(ax, RX, rows_y[1] + 0.5, RW, H,
         f"Excluded at title / abstract\nn = "
         f"{f['excluded_at_stage1_title_abstract']}\n" + codes(exc1),
         fontsize=5.2, fc="#F5F5F5", ec=GREY, color="#333333")
    _arrow(ax, LX + LW, rows_y[1] + H / 2, RX, rows_y[1] + H / 2, color=GREY)

    _box(ax, RX, rows_y[2] + 0.5, RW, H,
         f"NOT RETRIEVED\nn = {f['unreachable_eligibility_unresolved']}\n"
         "carried into both bounding\nanalyses, never excluded",
         fontsize=6.0, fc="#FDF1E0", ec=ORANGE, lw=1.3)
    _arrow(ax, LX + LW, rows_y[2] + H / 2, RX, rows_y[2] + H / 2, color=ORANGE)

    _box(ax, RX, rows_y[3] + 0.5, RW, H,
         f"Excluded at full text\nn = {f['excluded_at_fulltext']}\n" + codes(excF),
         fontsize=5.2, fc="#F5F5F5", ec=GREY, color="#333333")
    _arrow(ax, LX + LW, rows_y[3] + H / 2, RX, rows_y[3] + H / 2, color=GREY)

    s6 = prim["S6_unreachable"]
    p1 = prim["P1_complete_case"]
    lo = prim["P1_bound_lower_unreachable_all_negative"]
    hi = prim["P1_bound_upper_unreachable_all_positive"]
    ax.text(
        LX, 5.5,
        f"S6 unreachable = {s6['k']}/{s6['n']} = {s6['pct']:.1f}% "
        f"[{s6['ci95'][0]:.1f}%, {s6['ci95'][1]:.1f}%] — above the\n"
        f"pre-registered 15% threshold, so the protocol makes the\n"
        f"BOUNDING INTERVAL the headline: "
        f"P1 ∈ [{lo['pct']:.1f}%, {hi['pct']:.1f}%].\n"
        f"Complete case {p1['k']}/{p1['n']} = {p1['pct']:.1f}% "
        f"[{p1['ci95'][0]:.1f}%, {p1['ci95'][1]:.1f}%] is reported, "
        f"but is not the headline.",
        fontsize=6.4, va="top", ha="left", color=BLACK, linespacing=1.55,
    )

    # ---- right: unreachable rate by block -------------------------------- #
    axB = fig.add_subplot(gs[0, 1])
    order = ext["pre_registered_blocks"] + ext["post_hoc_blocks"]
    names, pct, lo_, hi_, posthoc = [], [], [], [], []
    for b in order:
        s = blocks[b]["S6_unreachable"]
        names.append(f"{b}\n(n={s['n']})")
        pct.append(s["pct"])
        lo_.append(s["pct"] - s["ci95"][0])
        hi_.append(s["ci95"][1] - s["pct"])
        posthoc.append(b in ext["post_hoc_blocks"])

    yy = np.arange(len(names))[::-1].astype(float)
    for i, (y_, p_, l_, h_, ph) in enumerate(zip(yy, pct, lo_, hi_, posthoc)):
        col = LIGHTGREY if ph else ORANGE
        ec = GREY if ph else ORANGE
        axB.errorbar([p_], [y_], xerr=[[l_], [h_]], fmt="none", ecolor=ec,
                     elinewidth=1.3, capsize=2.4, capthick=0.9)
        axB.plot([p_], [y_], "o", ms=5.2, color=col, mec=ec, mew=1.2)
    axB.axvline(15.0, color=BLACK, lw=0.8, ls=(0, (4, 3)))
    axB.axvline(s6["pct"], color=GREY, lw=0.9)
    axB.set_yticks(yy)
    axB.set_yticklabels(names, fontsize=6.4)
    axB.set_xlabel("unreachable (% of eligible)", fontsize=7.0)
    axB.set_xlim(0, 66)
    axB.set_title("the rate does not fall as\nthe sample grows", pad=6, fontsize=7.6)
    axB.text(14.0, yy[-1] - 1.55, "pre-registered\n15% threshold", fontsize=6.0,
             ha="right", va="bottom")
    axB.text(s6["pct"] + 1.2, yy[-1] - 1.55, f"pooled\n{s6['pct']:.1f}%",
             fontsize=6.0, color=GREY, ha="left", va="bottom")
    axB.set_ylim(yy[-1] - 1.7, yy[0] + 0.9)
    axB.legend(handles=[
        Line2D([], [], marker="o", ls="none", ms=5.2, color=ORANGE,
               label="pre-registered block"),
        Line2D([], [], marker="o", ls="none", ms=5.2, color=LIGHTGREY, mec=GREY,
               label="post-hoc block (R4)"),
    ], loc="upper right", frameon=False, fontsize=6.2, borderaxespad=0.2)

    save(fig, "fig3_prisma_flow.pdf")

    print(f"    stopping rule: {ext['verdict'][:110]}…")
    print(f"    headline: {prim['headline_per_protocol_section_7']}")


# --------------------------------------------------------------------------- #
# Figure 4 -- rank inversion: a null
# --------------------------------------------------------------------------- #

COHORT_ORDER = ["brain", "prostate_t2", "prostate_dwi", "breast", "knee"]
COHORT_PRETTY = {
    "brain": "brain",
    "prostate_t2": "prostate T2",
    "prostate_dwi": "prostate DWI",
    "breast": "breast",
    "knee": "knee",
}


def fig4():
    print("\nFigure 4 -- rank inversion between units against the within-unit noise floor")
    d = load("pipeline_out/rankinversion.json", "Fig 4A-4D, every mark")
    by = {c["cohort"]: c for c in d["cohorts"]}

    fig = plt.figure(figsize=(FULL_W, 5.3))
    gs = fig.add_gridspec(2, 2, hspace=0.62, wspace=0.42,
                          height_ratios=[1.0, 1.0])
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])

    # ---- A: between-unit disagreement vs the within-unit noise floor ----- #
    have = [c for c in COHORT_ORDER
            if by[c]["stability"].get("d_within_slice") is not None]
    y = np.arange(len(have))[::-1].astype(float)
    off = 0.17
    for yy, name in zip(y, have):
        s = by[name]["stability"]
        for key, dy, col, mk in (("d_within_slice", +off, C_SLICE, "o"),
                                 ("d_within_patient", -off, C_PATIENT, "s")):
            w = s[key]
            axA.plot([w["lo"], w["hi"]], [yy + dy, yy + dy], color=col,
                     lw=4.0, alpha=0.30, solid_capstyle="butt", zorder=2)
            axA.plot([w["median"]], [yy + dy], mk, ms=4.2, color=col, zorder=3)
        axA.plot([s["d_between_obs"]], [yy], marker="*", ms=11,
                 color=BLACK, mec=BLACK, ls="none", zorder=5)

    axA.set_yticks(y)
    axA.set_yticklabels(
        [f"{COHORT_PRETTY[c]}\nP = {by[c]['stability']['p_exceed']:.3f}"
         for c in have])
    axA.set_xlabel("rank distance  d = 1 − Kendall τ")
    axA.set_xlim(0, 2.6)
    axA.set_ylim(y[-1] - 0.9, y[0] + 1.05)
    axA.set_title("between-unit disagreement against\nthe within-unit noise floor",
                  pad=6, fontsize=7.8)
    axA.legend(handles=[
        Line2D([], [], color=C_SLICE, lw=4, alpha=0.30,
               label="within-unit floor, slice (95%)"),
        Line2D([], [], color=C_PATIENT, lw=4, alpha=0.30,
               label="within-unit floor, patient (95%)"),
        Line2D([], [], marker="*", ls="none", ms=9, color=BLACK,
               label="observed between-unit d"),
    ], loc="upper right", frameon=False, fontsize=6.0, borderaxespad=0.15,
        labelspacing=0.35)
    axA.text(0.02, -0.30, "P = P(within-unit d ≥ observed between-unit d)",
             transform=axA.transAxes, fontsize=6.0, color="#333333",
             ha="left", va="top")
    panel_tag(axA, "A", x=-0.30)

    # ---- B: pairs examined / sign-flipping / survivors ------------------- #
    ex, cand, surv = [], [], []
    for c in COHORT_ORDER:
        inv = by[c]["inversions"]
        ex.append(inv["n_pairs_examined"])
        cand.append(len(inv["candidates"]))
        surv.append(len(inv["survivors"]))
    yb = np.arange(len(COHORT_ORDER))[::-1].astype(float)
    h = 0.24
    axB.barh(yb + h, ex, height=h, color=LIGHTGREY, edgecolor=GREY, lw=0.6,
             label="pairs examined")
    axB.barh(yb, cand, height=h, color=SKY, edgecolor=BLUE, lw=0.6,
             label="sign-flipping candidates")
    axB.barh(yb - h, np.maximum(surv, 0.0), height=h, color=VERM,
             edgecolor=VERM, lw=0.6, label="survive the noise floor")
    for yy, s in zip(yb, surv):
        axB.text(3, yy - h, f"{s}", va="center", ha="left", fontsize=6.6,
                 color=VERM, fontweight="bold")
    for yy, e, c_ in zip(yb, ex, cand):
        axB.text(e + 4, yy + h, f"{e}", va="center", ha="left", fontsize=6.2,
                 color="#555555")
        if c_:
            axB.text(c_ + 4, yy, f"{c_}", va="center", ha="left", fontsize=6.2,
                     color=BLUE)
    axB.set_yticks(yb)
    axB.set_yticklabels([COHORT_PRETTY[c] for c in COHORT_ORDER])
    axB.set_xlabel("method pairs")
    axB.set_xlim(0, max(ex) * 1.22)
    axB.set_title(f"{sum(ex)} pairs examined, {sum(cand)} flip sign,\n"
                  f"{sum(surv)} survive", pad=6, fontsize=7.8)
    axB.set_ylim(yb[-1] - 0.65, yb[0] + 0.85)
    axB.legend(loc="lower right", frameon=False, fontsize=6.0, borderaxespad=0.2)
    panel_tag(axB, "B", x=-0.30)

    # ---- C: the cautionary panel -- prostate T2 split-half tau ----------- #
    pt = by["prostate_t2"]["stability"]
    sh = pt["split_half"]
    agree = by["prostate_t2"]["agreement"]
    entries = [
        ("between units\n(observed τ)", agree["tau_obs"], None, None, BLACK, "*"),
        ("within slice unit\nsplit-half τ", sh["tau_slice_median"],
         sh["tau_slice_lo"], sh["tau_slice_hi"], C_SLICE, "o"),
        ("within patient unit\nsplit-half τ", sh["tau_agg_median"],
         sh["tau_agg_lo"], sh["tau_agg_hi"], C_PATIENT, "s"),
    ]
    yc = np.arange(len(entries))[::-1].astype(float)
    axC.axvline(0.0, color=BLACK, lw=0.8, ls=(0, (4, 3)))
    axC.axvline(pt["stable_tau_threshold"], color=GREEN, lw=1.0, ls=(0, (1, 2)))
    for yy, (lab, v, lo_, hi_, col, mk) in zip(yc, entries):
        if lo_ is not None:
            axC.errorbar([v], [yy], xerr=[[v - lo_], [hi_ - v]], fmt="none",
                         ecolor=col, elinewidth=1.5, capsize=2.6, capthick=1.0)
        axC.plot([v], [yy], mk, ms=8 if mk == "*" else 5.5, color=col,
                 mec=col, ls="none")
        axC.text(v, yy + 0.26, f"{v:+.2f}", fontsize=6.4, color=col,
                 ha="center", va="bottom")
    axC.set_yticks(yc)
    axC.set_yticklabels([e[0] for e in entries])
    axC.set_xlim(-1.02, 1.02)
    axC.set_ylim(yc[-1] - 0.85, yc[0] + 0.85)
    axC.set_xlabel("Kendall τ")
    axC.set_title("prostate T2: the near-miss —  "
                  f"{len(by['prostate_t2']['inversions']['candidates'])}"
                  f"/{by['prostate_t2']['inversions']['n_pairs_examined']}"
                  f" pairs flip,\nverdict {pt['verdict']}", pad=6, fontsize=7.6)
    axC.text(pt["stable_tau_threshold"] + 0.03, yc[-1] - 0.80,
             "stability floor", fontsize=6.0, color=GREEN, ha="left", va="bottom")
    panel_tag(axC, "C", x=-0.30)

    # ---- D: what does survive -- the brain interaction ------------------- #
    it = by["brain"]["interaction"]
    rows = it["rows"]
    yd = np.arange(len(rows) + 1)[::-1].astype(float)
    axD.axvline(0.0, color=BLACK, lw=0.8, ls=(0, (4, 3)))
    for yy, r in zip(yd[:-1], rows):
        col = BLUE if r["supported"] else GREY
        axD.errorbar([r["interaction"]], [yy],
                     xerr=[[r["interaction"] - r["ci_lo"]],
                           [r["ci_hi"] - r["interaction"]]],
                     fmt="none", ecolor=col, elinewidth=1.4, capsize=2.4,
                     capthick=0.9)
        axD.plot([r["interaction"]], [yy], "o", ms=4.6, color=col, ls="none")
        if r["sign_flip"]:
            axD.text(r["ci_hi"] + 0.004, yy, "sign flip", fontsize=5.9,
                     color=VERM, va="center", ha="left")
    m = it["mean"]
    axD.errorbar([m["interaction"]], [yd[-1]],
                 xerr=[[m["interaction"] - m["ci_lo"]],
                       [m["ci_hi"] - m["interaction"]]],
                 fmt="none", ecolor=BLACK, elinewidth=2.0, capsize=3.0,
                 capthick=1.2)
    axD.plot([m["interaction"]], [yd[-1]], "D", ms=6.0, color=BLACK, ls="none")
    axD.set_yticks(yd)
    axD.set_yticklabels([r["model"] for r in rows]
                        + [f"mean over {it['n_models']}"])
    for tick, r in zip(axD.get_yticklabels(), rows + [None]):
        if r is None:
            tick.set_fontweight("bold")
    axD.set_xlabel("interaction I  =  d(patient unit) − d(slice unit),\n"
                   "with d = AUROC(magnitude) − AUROC(phase)")
    axD.set_ylim(yd[-1] - 0.75, yd[0] + 0.75)
    axD.set_title(f"what does survive: aggregation shifts the comparison\n"
                  f"by {m['interaction']:+.3f} "
                  f"[{m['ci_lo']:+.3f}, {m['ci_hi']:+.3f}], p = {m['p']:.3f}  "
                  f"(brain, {by['brain']['n_subjects']} subjects;\n"
                  f"{m['n_positive']}/{it['n_models']} positive, "
                  f"{it['n_supported']} Holm-supported)", pad=6, fontsize=7.4)
    panel_tag(axD, "D", x=-0.26)

    save(fig, "fig4_rank_inversion.pdf")

    print(f"    total pairs examined {sum(ex)}, sign-flipping {sum(cand)}, "
          f"survivors {sum(surv)}")
    print(f"    prostate_t2 split-half tau patient {sh['tau_agg_median']:.3f} "
          f"[{sh['tau_agg_lo']:.3f}, {sh['tau_agg_hi']:.3f}]")
    print(f"    brain interaction mean {m['interaction']:+.4f} "
          f"[{m['ci_lo']:+.4f}, {m['ci_hi']:+.4f}] p={m['p']}")
    dz = [r for r in rows if r["model"] == "densenet121/imagenet"][0]
    print(f"    densenet121 slice-level d = {dz['d_slice']:+.4f} "
          f"(the individual sign flip is NOT supported at the slice unit)")


# --------------------------------------------------------------------------- #
# Figure 5 -- the case study
# --------------------------------------------------------------------------- #

CASE_COHORTS = ["prostate_t2", "breast", "prostate_dwi"]
CASE_PRETTY = {"prostate_t2": "prostate T2", "breast": "breast",
               "prostate_dwi": "prostate DWI"}
CONDITIONS = ["magnitude", "phase", "both"]
COND_COLOR = {"magnitude": BLUE, "phase": ORANGE, "both": GREEN}
COND_MARK = {"magnitude": "o", "phase": "s", "both": "^"}

_C4_RE = re.compile(
    r"background only \(anatomy removed\)\s+([\d.]+)\s+"
    r"\[([\d.]+),\s*([\d.]+)\]\s+vs headline\s+([\d.]+)\s+"
    r"\[([\d.]+),\s*([\d.]+)\]"
)


def fig5():
    print("\nFigure 5 -- the multi-organ k-space phase case study")
    st = load("pipeline_out/results/statistics.json",
              "Fig 5A, patient-level AUROC + CI")
    vd = load("pipeline_out/report/verdict.json",
              "Fig 5B, background-only control + verdicts")

    runs = {(r["cohort"], r["condition"], r["seed"]): r for r in st["runs"]}
    agg = {(a["cohort"], a["condition"]): a for a in st["across_seeds"]}

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(FULL_W, 3.20),
        gridspec_kw={"width_ratios": [1.55, 1.0], "wspace": 0.32})

    # ---- A: patient-level AUROC per cohort per condition ----------------- #
    axA.axhline(0.5, color=BLACK, lw=0.8, ls=(0, (4, 3)))
    xticks, xlabels, condticks, condlabels = [], [], [], []
    xpos = 0.0
    for ci, coh in enumerate(CASE_COHORTS):
        base = xpos
        for k, cond in enumerate(CONDITIONS):
            xc = base + k * 0.80
            col = COND_COLOR[cond]
            a = agg[(coh, cond)]
            seeds = a["seeds"]
            # per-seed clustered CI, thin
            for j, sd in enumerate(seeds):
                r = runs[(coh, cond, sd)]["patient_level_mean"]
                jitter = (j - (len(seeds) - 1) / 2) * 0.19
                axA.errorbar([xc + jitter], [r["auc"]],
                             yerr=[[r["auc"] - r["ci_lo"]],
                                   [r["ci_hi"] - r["auc"]]],
                             fmt="none", ecolor=col, elinewidth=0.9,
                             alpha=0.45, capsize=1.8, capthick=0.7, zorder=2)
                axA.plot([xc + jitter], [r["auc"]], marker=COND_MARK[cond],
                         ms=3.0, color=col, mfc="white", mec=col, mew=0.8,
                         ls="none", alpha=0.85, zorder=3)
            # seed mean, bold
            axA.plot([xc], [a["patient_mean_auc"]["mean"]],
                     marker=COND_MARK[cond], ms=7.0, color=col, mec=col,
                     ls="none", zorder=4)
            condticks.append(xc)
            condlabels.append({"magnitude": "mag", "phase": "phase",
                               "both": "both"}[cond])
        xticks.append(base + 0.80)
        xlabels.append(CASE_PRETTY[coh])
        xpos = base + 3.05

    axA.set_xticks(condticks)
    axA.set_xticklabels(condlabels, fontsize=6.2)
    for xt, xl in zip(xticks, xlabels):
        axA.text(xt, -0.085, xl, transform=axA.get_xaxis_transform(),
                 ha="center", va="top", fontsize=7.5)
    axA.set_ylabel("patient-level AUROC (mean aggregation)")
    axA.set_ylim(0.15, 1.00)
    axA.set_xlim(-0.6, xpos - 0.75)
    verdicts = {c: vd["cohorts"][c]["verdict"] for c in CASE_COHORTS}
    assert set(verdicts.values()) == {"NOT SUPPORTED"}, verdicts
    axA.set_title("every cohort: NOT SUPPORTED", pad=6)
    axA.text(-0.55, 0.505, "chance", fontsize=6.3, va="bottom", ha="left")
    axA.legend(handles=[
        Line2D([], [], marker=COND_MARK[c], ls="none", ms=6,
               color=COND_COLOR[c], label=c) for c in CONDITIONS
    ] + [Line2D([], [], marker="o", ls="none", ms=3.0, color=GREY, mfc="white",
                label="individual seed")],
        loc="upper left", frameon=False, ncol=2, handletextpad=0.5,
        columnspacing=1.0, borderaxespad=0.15)
    panel_tag(axA, "A", x=-0.155)

    # ---- B: training on air alone ---------------------------------------- #
    ctrl = {}
    for coh in CASE_COHORTS:
        c4 = next(c for c in vd["cohorts"][coh]["criteria"] if c["code"] == "C4")
        m = _C4_RE.search(c4["evidence"])
        if m is None:
            sys.exit(f"could not parse the C4 evidence string for {coh}")
        bg, bglo, bghi, hd, hdlo, hdhi = (float(g) for g in m.groups())
        # cross-check the parsed headline against the machine-readable field
        assert abs(hd - round(c4["headline_point"], 3)) < 1e-9, (
            coh, hd, c4["headline_point"])
        ctrl[coh] = dict(bg=bg, bglo=bglo, bghi=bghi, hd=hd, hdlo=hdlo,
                         hdhi=hdhi, key=c4["headline_key"], gap=hd - bg)

    yb = np.arange(len(CASE_COHORTS))[::-1].astype(float)
    dy = 0.16
    axB.axvline(0.5, color=BLACK, lw=0.8, ls=(0, (4, 3)))
    for yy, coh in zip(yb, CASE_COHORTS):
        c = ctrl[coh]
        axB.plot([c["bg"], c["hd"]], [yy - dy, yy + dy], color=LIGHTGREY,
                 lw=2.6, solid_capstyle="round", zorder=1)
        axB.errorbar([c["hd"]], [yy + dy],
                     xerr=[[c["hd"] - c["hdlo"]], [c["hdhi"] - c["hd"]]],
                     fmt="none", ecolor=BLUE, elinewidth=1.4, capsize=2.4,
                     capthick=0.9, zorder=3)
        axB.plot([c["hd"]], [yy + dy], "o", ms=5.6, color=BLUE, zorder=4)
        axB.errorbar([c["bg"]], [yy - dy],
                     xerr=[[c["bg"] - c["bglo"]], [c["bghi"] - c["bg"]]],
                     fmt="none", ecolor=PURPLE, elinewidth=1.4, capsize=2.4,
                     capthick=0.9, zorder=3)
        axB.plot([c["bg"]], [yy - dy], "D", ms=5.0, color=PURPLE, mfc="white",
                 mec=PURPLE, mew=1.4, zorder=4)
        axB.text(0.5 * (c["bg"] + c["hd"]), yy + 0.30,
                 f"shift {c['gap']:+.3f}", fontsize=6.3, ha="center",
                 va="bottom", color="#333333")

    gaps = np.array([ctrl[c]["gap"] for c in CASE_COHORTS])
    axB.set_yticks(yb)
    axB.set_yticklabels([CASE_PRETTY[c] for c in CASE_COHORTS])
    axB.set_xlabel("slice-level AUROC\n(the unit the control is defined at)")
    axB.set_xlim(0.40, 0.80)
    axB.set_ylim(yb[-1] - 1.15, yb[0] + 0.85)
    axB.set_title("delete the anatomy, keep the air:\n"
                  f"mean shift {gaps.mean():+.3f} AUROC", pad=6, fontsize=7.8)
    axB.legend(handles=[
        Line2D([], [], marker="o", ls="none", ms=5.6, color=BLUE,
               label="headline (phase, whole image)"),
        Line2D([], [], marker="D", ls="none", ms=5.0, color=PURPLE, mfc="white",
               mew=1.4, label="background only (anatomy zeroed)"),
    ], loc="lower left", frameon=False, fontsize=6.2, borderaxespad=0.15)
    panel_tag(axB, "B", x=-0.34)

    save(fig, "fig5_case_study.pdf")

    for coh in CASE_COHORTS:
        a = {c: agg[(coh, c)]["patient_mean_auc"]["mean"] for c in CONDITIONS}
        print(f"    {coh:13s} patient-level  "
              + "  ".join(f"{c} {a[c]:.3f}" for c in CONDITIONS)
              + f"   [verdict {verdicts[coh]}]")
    for coh in CASE_COHORTS:
        c = ctrl[coh]
        print(f"    {coh:13s} air-only {c['bg']:.3f} [{c['bglo']:.3f}, "
              f"{c['bghi']:.3f}] vs headline {c['hd']:.3f} "
              f"[{c['hdlo']:.3f}, {c['hdhi']:.3f}]  gap {c['gap']:+.3f}")
    print(f"    mean signed gap across the three cohorts: {gaps.mean():+.4f}")
    print(f"    largest single gap: {gaps.max():+.4f}")


# --------------------------------------------------------------------------- #

def main():
    print("=" * 78)
    print("make_figures.py -- every panel traced to the artefact it was read from")
    print(f"repo: {REPO}")
    print(f"out:  {OUT.relative_to(REPO)}")
    print("=" * 78)
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    print("\n" + "=" * 78)
    print("SOURCE LEDGER")
    for line in _SOURCES:
        print(line)
    print("=" * 78)


if __name__ == "__main__":
    main()
