#!/usr/bin/env python3
"""
Generate every figure in the manuscript from the artefacts on disk.

    venv/bin/python paper/tex/make_figures.py

No number in this file is typed by hand. Every value plotted is read from a JSON
artefact under pipeline_out/ or paper/, and each function prints the artefact it
read so a reviewer can trace any mark on any panel back to a file.

Output
------
    fig1_collapse.pdf            main text
    fig2_trivial_fraction.pdf    main text
    fig3_prisma_flow.pdf         main text
    fig4_rank_inversion.pdf      main text
    fig5_case_study.pdf          main text
    fig6_qualitative_phase.pdf   main text -- the network's literal input
    figS1_acquisition_fingerprint.pdf   supplement
    figS2_recon_fidelity.pdf            supplement
    figS3_qualitative_cohorts.pdf       supplement

Vector PDF, fonts embedded as Type-42 (TrueType) subsets.

Palette: Okabe-Ito colour-blind-safe. No red/green pair is used to carry meaning.

PHASE IS CIRCULAR. Raw phase in radians is drawn with a CYCLIC colormap
(`twilight`) on exactly [-pi, +pi], so -pi and +pi receive the same colour and
the wrap is visible as a wrap rather than as an edge. sin(phase) and cos(phase)
are NOT circular -- they are ordinary quantities on [-1, +1] -- so they get a
diverging Okabe-Ito blue/orange map instead, where -1 and +1 are as far apart as
the colours can make them. Min-max normalising phase is one of the three defects
recorded in legacy/README.md as invalidating the original study; nothing here
normalises phase at all.

IMAGE DATA. The MRI panels are raster by nature and are embedded as PDF image
XObjects at their native 224x224 resolution (`interpolation="none"`, so nothing
resamples them on the way out). Every axis, tick, label, contour, colourbar and
annotation around them is vector. `verify_pdfs()` checks this after the build.

DATA USE. The MRI panels are derived from the NYU fastMRI database, whose data
sharing agreement permits use "in academic publications and presentations" but
forbids redistribution. paper/tex/figures/*.pdf is a TRACKED directory in a
public git repository, so the imaging PDFs written here are excluded by name in
paper/tex/.gitignore. Do not remove those rules to "fix" a missing figure; ship
the file to the journal out of band instead.
"""

from __future__ import annotations

import json
import re
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

# --------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------- #

HERE = Path(__file__).resolve().parent            # paper/tex
REPO = HERE.parent.parent                          # repo root
OUT = HERE / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# The stage-2 image cache. Written by s02_*.py, never committed (it is fastMRI
# k-space, derived). The imaging figures read it directly, in stored order, with
# no transpose and no re-reconstruction.
CACHE = REPO / "pipeline_out" / "cache"

_SOURCES: list[str] = []


def _ledger(panel: str, relpath: str) -> None:
    line = f"  {panel:<34s} <- {relpath}"
    print(line)
    _SOURCES.append(line)


def load(relpath: str, panel: str):
    """Read a JSON artefact and record it as the provenance of `panel`."""
    p = REPO / relpath
    if not p.exists():
        sys.exit(f"MISSING ARTEFACT: {p}\n  needed for: {panel}")
    with p.open() as fh:
        obj = json.load(fh)
    _ledger(panel, relpath)
    return obj


def load_glob(pattern: str, panel: str) -> list[tuple[str, dict]]:
    """
    Read every JSON matching `pattern` (relative to the repo root), sorted.

    Used where the set of artefacts is itself a finding -- the confound controls
    that stage 5 happened to write for a cohort -- so that the figure shows what
    ran rather than a list chosen here.
    """
    hits = sorted(REPO.glob(pattern))
    if not hits:
        sys.exit(f"MISSING ARTEFACTS: no match for {pattern}\n  needed for: {panel}")
    out = []
    for p in hits:
        with p.open() as fh:
            out.append((str(p.relative_to(REPO)), json.load(fh)))
    _ledger(panel, f"{pattern}  ({len(out)} file(s))")
    return out


def load_index(cohort: str, panel: str):
    """The stage-2 slice index for a cohort: one row per cached slice."""
    import pandas as pd

    rel = f"pipeline_out/cache/{cohort}_index.csv"
    p = REPO / rel
    if not p.exists():
        sys.exit(f"MISSING ARTEFACT: {p}\n  needed for: {panel}")
    _ledger(panel, rel)
    return pd.read_csv(p, low_memory=False)


def load_csv(relpath: str, panel: str):
    import pandas as pd

    p = REPO / relpath
    if not p.exists():
        sys.exit(f"MISSING ARTEFACT: {p}\n  needed for: {panel}")
    _ledger(panel, relpath)
    return pd.read_csv(p, low_memory=False)


def require_cache(cohorts: list[str]) -> None:
    """
    Fail loudly rather than fall back to the report PNGs.

    Rebuilding an imaging panel from `pipeline_out/report/figures/*.png` would
    make the figure unverifiable -- the PNG is a rendering, not the array -- and
    would silently pick up whichever of the three report* directories happened
    to be on disk. If the cache is gone, re-run stage 2.
    """
    missing = [c for c in cohorts
               if not (CACHE / f"{c}.h5").exists()
               or not (CACHE / f"{c}_index.csv").exists()]
    if missing:
        sys.exit(
            "MISSING IMAGE CACHE for: " + ", ".join(missing) + "\n"
            f"  looked in {CACHE}\n"
            "  Re-run stage 2 (pipeline/s02_*.py). This script will NOT rebuild an\n"
            "  imaging panel from pipeline_out/report/figures/*.png: a PNG is a\n"
            "  rendering of the array, not the array, and cannot be verified.")


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

# --------------------------------------------------------------------------- #
# colormaps for the imaging panels
# --------------------------------------------------------------------------- #
#
# MAGNITUDE -> greyscale. It is an intensity image and nothing else.
#
# RAW PHASE -> `twilight`, a CYCLIC map, on exactly [-pi, +pi]. Phase is an angle:
# -pi and +pi are the same physical state, and a linear map would draw the wrap as
# a hard edge that a reader would take for structure. twilight closes the loop, so
# a wrap looks like a wrap. The map is also perceptually uniform in lightness and
# stays readable in greyscale print.
#
# sin(phase), cos(phase) -> NOT cyclic. These are the two channels the network is
# actually fed, and each is an ordinary quantity on [-1, +1]; a cyclic map here
# would give -1 and +1 the same colour, which would be a second, opposite way of
# lying about the same data. They get a diverging Okabe-Ito blue/orange map with
# white at 0, so zero-crossings read as zero-crossings.
CMAP_MAG = "gray"
CMAP_PHASE = "twilight"                       # cyclic, for radians on [-pi, pi]
CMAP_SINCOS = LinearSegmentedColormap.from_list(
    "okabeito_div", [BLUE, "#F2F2F2", ORANGE])  # diverging, for [-1, +1]

# Body-mask contour. Drawn as a dark halo under a light core so it survives on
# top of greyscale, on top of twilight's near-black band and on top of the
# diverging map's white centre. It carries no quantitative meaning.
MASK_HALO = "#101010"
MASK_CORE = SKY


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
# Figure 6 -- the network's literal input
# --------------------------------------------------------------------------- #
#
# The manuscript describes a "phase channel" and never shows one. This figure is
# the only place a reader can see what the estimator was handed. It is read out
# of pipeline_out/cache/<cohort>.h5 in stored order -- no transpose, no
# re-reconstruction, no renormalisation beyond the exact channel construction in
# s03_train.CacheSliceDataset:
#
#     magnitude channel : zscore(mag)
#     phase channels    : sin(phase), cos(phase)
#
# and the fourth column shows the raw radians those two are computed from, which
# the network never sees, so that the reader can check the wrap for themselves.
#
# The slice is chosen by a rule, not by eye: the MEDIAN ROW of the official test
# split at that label, in cache order. That is the same rule
# pipeline/s06_report.py:fig_qualitative uses, it involves no ranking on any
# outcome, and it is reproducible from the index CSV with one line of pandas.
# Picking the most convincing-looking slice would make this figure an argument
# instead of an example.

# Row-label wording. Which cohorts are DIAGNOSTIC and which are CONFOUND cohorts
# is read from verdict.json (`cohorts` vs `confound_cohorts`) -- never assumed
# here -- because printing "tumour present" over a coil-count label would be a
# false statement typeset into a figure.
#
# The same rule has a second edge, and it is the one that bites: a DIAGNOSTIC
# cohort whose label was recorded per PATIENT and broadcast to every slice
# cannot carry the words "on this slice" either. The breast release is exactly
# that case -- s02_breast.load_labels() is keyed by patient id, and no breast
# patient in the index carries more than one distinct slice label -- and its
# negative class also absorbs the release's benign code, so "no tumour
# annotated" is false there twice over. Whether a cohort's label is
# slice-specific is therefore MEASURED from the index rather than assumed, and
# a cohort that fails the test gets the neutral "label = 1 (patient level)"
# wording, the same shape the confound cohorts already use.
DIAG_POS = "tumour annotated\non this slice"
DIAG_NEG = "no tumour\nannotated"
DIAG_POS_PT = "label = 1\n(patient level)"
DIAG_NEG_PT = "label = 0\n(patient level)"


def _label_is_slice_specific(index) -> bool:
    """
    True iff some patient in the index carries more than one distinct label.

    That is the only evidence the index itself offers that the annotation was
    made at the slice and not broadcast down from a coarser unit. If no patient
    varies, the figure must not claim the annotation is on the slice, even if it
    happens to be: the index cannot show it, so the panel may not say it.
    """
    if not {"patient_id", "label"}.issubset(index.columns):
        return False
    return int(index.groupby("patient_id")["label"].nunique().max()) > 1


def _pick_median_test_row(index, label: int):
    """Median row of the official test split at `label`, in cache order."""
    pool = index[index["official_split"].astype(str) == "test"]
    if pool.empty:
        pool = index
    sub = pool[pool["label"] == label].sort_values("idx")
    if sub.empty:
        return None
    return sub.iloc[len(sub) // 2].to_dict()


def _class_words(cohort: str, index, vd: dict) -> tuple[str, str]:
    """
    (name of label 1, name of label 0), taken from the artefacts.

    For a CONFOUND cohort the class name is whatever stage 2 wrote in the index's
    own `label_name` column -- ">=16", "CORPD_FBK" -- never a rewording of it,
    and never anything containing the word "tumour". What that label MEANS is
    quoted separately from verdict.json by `_label_meaning`, so the two cannot
    drift apart on the page.

    For a DIAGNOSTIC cohort the words "on this slice" are used only where the
    index can show the label really is per-slice; see `_label_is_slice_specific`.
    """
    if cohort in vd.get("confound_cohorts", {}):
        pos, neg = "label = 1", "label = 0"
        if "label_name" in index.columns:
            got = {int(k): sorted({str(x) for x in v.dropna().unique()})
                   for k, v in index.groupby("label")["label_name"]}
            if got.get(1):
                pos = f"label = {'/'.join(got[1])}"
            if got.get(0):
                neg = f"label = {'/'.join(got[0])}"
        return pos, neg
    if cohort in vd.get("cohorts", {}):
        if _label_is_slice_specific(index):
            return DIAG_POS, DIAG_NEG
        print(f"    {cohort}: no patient in the index carries more than one "
              f"distinct slice label, so the label is not shown as slice-level")
        return DIAG_POS_PT, DIAG_NEG_PT
    sys.exit(f"{cohort} is in neither verdict.json cohorts nor confound_cohorts")


def _label_meaning(cohort: str, vd: dict) -> str:
    """What the cohort's label actually is, in verdict.json's own words."""
    if cohort in vd.get("confound_cohorts", {}):
        return vd["confound_cohorts"][cohort]["label"]
    return "clinically annotated tumour present on this slice"


def _read_slice(fh, k: int):
    mag = np.asarray(fh["mag"][k], dtype=np.float32)
    phase = np.asarray(fh["phase"][k], dtype=np.float32)
    mask = np.asarray(fh["mask"][k], dtype=bool)
    return mag, phase, mask


def _show(ax, img, cmap, vmin, vmax, mask=None):
    """
    One MRI panel.

    `interpolation="none"` is load-bearing: with the PDF backend it embeds the
    224x224 array as an image XObject at native resolution instead of resampling
    it to the figure dpi. The contour and the frame stay vector paths.
    """
    im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="none",
                   aspect="equal", origin="upper")
    if mask is not None and mask.any():
        ax.contour(mask.astype(float), levels=[0.5], colors=[MASK_HALO],
                   linewidths=1.5, zorder=5)
        ax.contour(mask.astype(float), levels=[0.5], colors=[MASK_CORE],
                   linewidths=0.65, zorder=6)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.5)
        s.set_color("#B0B0B0")
    return im


def _prov(meta: dict) -> str:
    return (f"cache row {int(meta['idx'])} · {meta['file']} · slice "
            f"{meta['slice']} · patient {meta['patient_id']}")


QUAL_MAIN = "prostate_t2"


def fig6():
    print("\nFigure 6 -- the four channels, read straight out of the stage-2 cache")
    import h5py

    require_cache([QUAL_MAIN])
    vd = load("pipeline_out/report/verdict.json",
              "Fig 6, class wording + verdict")
    index = load_index(QUAL_MAIN, "Fig 6, slice choice + provenance")
    pos_w, neg_w = _class_words(QUAL_MAIN, index, vd)

    picks = []
    for lab, word in ((1, pos_w), (0, neg_w)):
        meta = _pick_median_test_row(index, lab)
        if meta is None:
            sys.exit(f"no test-split row with label={lab} in {QUAL_MAIN}")
        picks.append((lab, word, meta))

    cols = [
        ("magnitude channel", "zscore(mag)", CMAP_MAG, None),
        ("phase channel 1", "sin(phase)", CMAP_SINCOS, (-1.0, 1.0)),
        ("phase channel 2", "cos(phase)", CMAP_SINCOS, (-1.0, 1.0)),
        ("raw phase", "radians, never rescaled", CMAP_PHASE, (-np.pi, np.pi)),
    ]

    fig = plt.figure(figsize=(FULL_W, 4.55))
    gs = fig.add_gridspec(
        2, 4, left=0.135, right=0.995, top=0.835, bottom=0.150,
        wspace=0.075, hspace=0.115)

    ims: list = [None] * 4
    bottom: list = [None] * 4
    h5 = CACHE / f"{QUAL_MAIN}.h5"
    _ledger("Fig 6, every image pixel", str(h5.relative_to(REPO)))
    with h5py.File(h5, "r") as fh:
        for r, (lab, word, meta) in enumerate(picks):
            k = int(meta["idx"])
            mag, phase, mask = _read_slice(fh, k)
            mag_z = (mag - mag.mean()) / (mag.std() + 1e-8)
            arrays = [mag_z, np.sin(phase), np.cos(phase), phase]
            for c, ((_, _, cmap, lims), img) in enumerate(zip(cols, arrays)):
                ax = fig.add_subplot(gs[r, c])
                if lims is None:
                    v = float(np.percentile(np.abs(img), 99)) or 1.0
                    lims = (-v, v)
                im = _show(ax, img, cmap, lims[0], lims[1], mask=mask)
                ims[c] = im
                if r == len(picks) - 1:
                    bottom[c] = ax
                if r == 0:
                    ax.set_title(f"{cols[c][0]}\n{cols[c][1]}", fontsize=7.0,
                                 pad=3.5, linespacing=1.25)
                if c == 0:
                    ax.set_ylabel(word, fontsize=7.0, labelpad=4,
                                  linespacing=1.3, fontweight="bold")
                    panel_tag(ax, "AB"[r], x=-0.40, y=0.94)
                    ax.text(0.0, -0.045, _prov(meta), transform=ax.transAxes,
                            fontsize=5.4, color="#555555", va="top", ha="left")
            print(f"    row {'AB'[r]}  label={lab}  {_prov(meta)}  "
                  f"mag range [{mag.min():.4g}, {mag.max():.4g}]  "
                  f"phase range [{phase.min():+.4f}, {phase.max():+.4f}] rad  "
                  f"mask covers {mask.mean():.4f} of the frame")

    # One shared colourbar per column, under the bottom row, inset so that
    # neighbouring end-labels ("1.0" and "-1.0") cannot run into each other.
    for c in range(4):
        box = bottom[c].get_position()
        inset = box.width * 0.11
        cax = fig.add_axes([box.x0 + inset, 0.088,
                            box.width - 2 * inset, 0.019])
        cb = fig.colorbar(ims[c], cax=cax, orientation="horizontal")
        cb.outline.set_linewidth(0.5)
        cb.ax.tick_params(labelsize=5.6, width=0.5, length=2.0, pad=1.2)
        if c == 3:
            cb.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
            cb.set_ticklabels(["−π", "−π/2", "0", "π/2", "π"])
        elif c in (1, 2):
            # the endpoints matter: sin and cos really do reach ±1, and a
            # pruned locator would let a reader think the range was clipped
            cb.set_ticks([-1.0, 0.0, 1.0])
        else:
            cb.set_ticks(MaxNLocator(nbins=3, prune="both"))
            # The magnitude ARRAY is the z-score as fed; the magnitude DISPLAY
            # is windowed at the 99th centile of |z|, exactly as Figure S3's
            # shared colourbar already says. Saying it in one figure and not the
            # other is how two panels of the same data come to look different.
            cb.set_label("±99th centile of |z|", fontsize=5.2, labelpad=1.4)

    fig.text(0.010, 0.988,
             f"{COHORT_PRETTY[QUAL_MAIN]} — the estimator's input, read from "
             f"pipeline_out/cache/{QUAL_MAIN}.h5 in stored order",
             fontsize=8.0, ha="left", va="top", fontweight="bold")
    fig.text(0.010, 0.947,
             "no transpose · no re-reconstruction · phase never min–max scaled · "
             "outline = stage-2 body mask, whose complement is the air-only control\n"
             "columns 2 and 3 are what the network is fed; column 4 is the raw "
             "angle they are computed from, on a cyclic scale so the wrap reads "
             "as a wrap",
             fontsize=6.2, ha="left", va="top", color="#444444", linespacing=1.5)
    save(fig, "fig6_qualitative_phase.pdf")


# --------------------------------------------------------------------------- #
# Figure S1 -- what the phase channel does predict
# --------------------------------------------------------------------------- #
#
# fig6_confound_predictability.png in pipeline_out/report showed only the brain
# and knee cohorts, whose label IS the acquisition property -- so the panel was a
# tautology, which is why it had to carry a "READ THIS FIGURE BACKWARDS" title.
# The damning measurements are on the three CLINICAL cohorts, where a diagnostic
# claim was actually made, and they were buried inside fig4_controls_*.png.
#
# Every stage-5 confound control that exists for the clinical cohorts is plotted,
# not a chosen subset: the near-chance ones (breast/folder, prostate DWI/
# institution) sit in the same panel at the same weight as the near-ceiling ones.

_C6_CEIL_RE = re.compile(r"predictability from the same input\s*<\s*([\d.]+)\s*AUC")
_CTRL_GLOB = ("pipeline_out/controls/results/{c}/"
              "{c}__confound_predictability__*__*__seed*.json")


def figS1():
    print("\nFigure S1 -- the input channel predicts the acquisition, "
          "on the cohorts where a diagnostic claim was made")
    vd = load("pipeline_out/report/verdict.json",
              "Fig S1, C6 ceiling + headlines + confound cohorts")
    rb = load("pipeline_out/robustness/s09_robustness.json",
              "Fig S1B, within-site stratified coil result")

    # the C6 ceiling, parsed out of the criterion's own rule text
    ceil = None
    for coh in vd["cohorts"].values():
        c6 = next(c for c in coh["criteria"] if c["code"] == "C6")
        m = _C6_CEIL_RE.search(c6["rule"])
        if m:
            ceil = float(m.group(1))
            break
    if ceil is None:
        sys.exit("could not parse the C6 ceiling out of verdict.json")

    # ---- panel A rows: every confound control stage 5 wrote, per cohort ---- #
    rows = []
    for coh in CASE_COHORTS:
        got: dict[str, dict] = {}
        for rel, d in load_glob(_CTRL_GLOB.format(c=coh),
                                f"Fig S1A, {coh} confound controls"):
            cd = d["control_detail"]
            ci = cd["test_auc_ci95"]
            got.setdefault(cd["target"], {})[d["condition"]] = dict(
                auc=ci["auc"], lo=ci["lo"], hi=ci["hi"],
                n=ci["n"], k=ci["n_clusters"], src=rel)
        for target in sorted(got):
            rows.append(dict(cohort=coh, target=target, by=got[target]))

    hl = {c: vd["cohorts"][c]["headlines"][f"{c}/phase@slice"] for c in CASE_COHORTS}
    cc = vd["confound_cohorts"]
    cvs = rb["coil_vs_site"]
    # s09 writes one within-stratum record PER SEED. Taking [0] and printing the
    # number without the seed silently reports the better of two runs (site:
    # 0.979 at seed 42, 0.974 at seed 123), so the seed is carried onto the
    # figure and every other seed is printed to the console ledger below.
    ws_all = cvs["verdict"]["within_stratum"]["site"]
    ws = ws_all[0]
    het_floor = cvs["verdict"]["floor_per_class"]

    # One axis, not two. The clinical rows and the hardware-label rows are the
    # same measurement on the same scale, and putting them on separate axes
    # invites the reader to compare two different x ranges by eye.
    # 5.05 rather than 4.60: the within-site block is one row per stratum, and
    # the below-floor stratum is a row like any other.
    fig, ax = plt.subplots(figsize=(FULL_W, 5.25))
    fig.subplots_adjust(left=0.305, right=0.955, top=0.845, bottom=0.155)

    ax.axvspan(ceil, 1.02, color=LIGHTGREY, alpha=0.55, lw=0, zorder=0)
    ax.axvline(ceil, color=BLACK, lw=0.8, ls=(0, (1, 2)), zorder=1)
    ax.axvline(0.5, color=BLACK, lw=0.8, ls=(0, (4, 3)), zorder=1)

    dy = 0.20
    yy = 0.0
    yticks: list[float] = []
    ylabs: list[str] = []

    def point(y, cond, e, col=None, mfc=None, annotate=True):
        col = col or COND_COLOR[cond]
        ax.errorbar([e["auc"]], [y],
                    xerr=[[max(0.0, e["auc"] - e["lo"])],
                          [max(0.0, e["hi"] - e["auc"])]],
                    fmt="none", ecolor=col, elinewidth=1.3, capsize=2.2,
                    capthick=0.9, zorder=3)
        ax.plot([e["auc"]], [y], marker=COND_MARK.get(cond, "D"), ms=5.0,
                color=col, mfc=mfc or col, mec=col, mew=1.3, ls="none", zorder=4)
        if annotate:
            # keep the number inside the axis: flip it to the left of the
            # interval once the interval runs into the right-hand edge
            if e["hi"] > 0.90:
                ax.text(e["lo"] - 0.012, y, f"{e['auc']:.3f}", fontsize=5.9,
                        color=col, va="center", ha="right", zorder=5)
            else:
                ax.text(e["hi"] + 0.012, y, f"{e['auc']:.3f}", fontsize=5.9,
                        color=col, va="center", ha="left", zorder=5)

    # ---- block 1: the cohorts a diagnostic claim was made on -------------- #
    top_of_block1 = yy
    for row in rows:
        ph, mg = row["by"].get("phase"), row["by"].get("magnitude")
        if ph is not None:
            point(yy + dy, "phase", ph)
        if mg is not None:
            point(yy - dy, "magnitude", mg, annotate=False)
        h = hl[row["cohort"]]
        ax.plot([h["point"]], [yy], marker="|", ms=10, mew=1.5, color=BLACK,
                ls="none", zorder=4)
        yticks.append(yy)
        ylabs.append(f"{CASE_PRETTY[row['cohort']]}  ·  {ph['k']} test subjects\n"
                     f"label = {row['target']}")
        yy -= 1.0
    sep = yy + 0.5
    yy -= 0.55

    # ---- block 2: cohorts whose label contains no pathology at all -------- #
    top_of_block2 = yy
    for name in sorted(cc):
        blk = cc[name]
        for cond in ("phase", "magnitude"):
            a = blk["auc"][cond]
            e = dict(auc=a["point"], lo=a["lo"], hi=a["hi"])
            point(yy + (dy if cond == "phase" else -dy), cond, e,
                  annotate=(cond == "phase"))
        yticks.append(yy)
        short = blk["label"].split("(")[0].strip().rstrip(",")
        ylabs.append(f"{COHORT_PRETTY.get(name, name)}  ·  "
                     f"{blk['n_test_subjects']} test subjects\nlabel = {short}")
        yy -= 1.0

    # The brain measurement again, stratified within site: the answer to
    # "you have only shown that phase encodes the SITE".
    #
    # It is drawn one stratum per row, and NOT as a single "stratified within
    # site" number, because s09's own heterogeneity note says so in as many
    # words: "the estimate is carried by ['NYU']; the ['TH'] stratum/a fall
    # below the power floor and sit as low as 0.578, so the within-stratum claim
    # rests on the large stratum and must be written that way". A figure that
    # advertises showing every near-chance control on the rows above cannot then
    # drop a 0.54 stratum here. The below-floor stratum has no interval in the
    # artefact, so none is drawn for it.
    het = cvs["verdict"]["within_stratum_heterogeneity"]["site"]
    seed = str(ws["seed"])
    strata = [r for r in het["per_stratum"] if str(r["seed"]) == seed]
    if not strata:
        sys.exit(f"Fig S1B: no per-stratum site rows for seed {seed}")
    strata.sort(key=lambda r: (not r["counted"], r["stratum"]))
    n_all = sum(r["n_pos"] + r["n_neg"] for r in strata)
    for r in strata:
        n_r = r["n_pos"] + r["n_neg"]
        if r["counted"]:
            point(yy, "stratified",
                  dict(auc=ws["stratified_auc"], lo=ws["ci_lo"],
                       hi=ws["ci_hi"]), col=PURPLE, mfc="white")
            tag = ""
        else:
            ax.plot([r["auc"]], [yy], marker="D", ms=5.0, color=PURPLE,
                    mfc="white", mec=PURPLE, mew=1.3, ls="none", alpha=0.5,
                    zorder=4)
            ax.text(r["auc"] + 0.012, yy, f"{r['auc']:.3f}", fontsize=5.9,
                    color=PURPLE, alpha=0.75, va="center", ha="left", zorder=5)
            tag = " — NOT COUNTED"
        yticks.append(yy)
        note = ("" if r["counted"]
                else f", below the {het_floor}-per-class floor")
        ylabs.append(f"brain, phase, WITHIN SITE {r['stratum']}{tag}\n"
                     f"{n_r} of {n_all} subjects{note}; seed {seed}\n"
                     f"unstratified {ws['unstratified_auc']:.3f}")
        yy -= 1.45
    bottom = yy + 1.45

    ax.axhline(sep, color=GREY, lw=0.6, ls="-", zorder=1)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabs, fontsize=6.3, linespacing=1.35)
    # extra room under the last row so the key sits in white space rather than
    # on top of the hardware-label rows
    ax.set_ylim(bottom - 2.75, top_of_block1 + 0.85)
    ax.set_xlim(0.0, 1.02)
    ax.set_xlabel("AUROC at predicting the ACQUISITION PROPERTY  "
                  "(slice level, 95% subject-clustered)", fontsize=7.2)

    ax.text(-0.295, top_of_block1 + 0.52,
            "A DIAGNOSTIC CLAIM WAS MADE ON THESE COHORTS", fontsize=6.4,
            fontweight="bold", color="#333333", ha="left", va="center",
            transform=ax.get_yaxis_transform(which="grid"))
    ax.text(-0.295, top_of_block2 + 0.60,
            "THESE COHORTS CARRY NO PATHOLOGY LABEL AT ALL", fontsize=6.4,
            fontweight="bold", color="#333333", ha="left", va="center",
            transform=ax.get_yaxis_transform(which="grid"))
    ax.text(ceil + 0.008, bottom - 2.68,
            f"pre-registered C6 ceiling {ceil:.2f}", fontsize=6.0,
            color="#333333", ha="left", va="bottom")
    ax.text(0.5 - 0.008, bottom - 2.68, "chance", fontsize=6.0,
            color="#333333", ha="right", va="bottom")

    ax.legend(handles=[
        Line2D([], [], marker=COND_MARK["phase"], ls="none", ms=5,
               color=COND_COLOR["phase"], label="phase channel"),
        Line2D([], [], marker=COND_MARK["magnitude"], ls="none", ms=5,
               color=COND_COLOR["magnitude"], label="magnitude channel"),
        Line2D([], [], marker="|", ls="none", ms=10, mew=1.5, color=BLACK,
               label="the same cohort's phase headline for the TUMOUR label"),
        Line2D([], [], marker="D", ls="none", ms=5, color=PURPLE, mfc="white",
               mew=1.3, label="phase, within one site stratum"),
        Line2D([], [], marker="D", ls="none", ms=5, color=PURPLE, mfc="white",
               mew=1.3, alpha=0.5,
               label="the same, in a stratum below that floor "
                     "(s09 writes no interval for it)"),
    ], loc="lower left", frameon=False, fontsize=6.0, borderaxespad=0.0,
        labelspacing=0.40, handletextpad=0.6,
        bbox_to_anchor=(0.004, 0.028))

    fig.text(0.005, 0.988,
             "What the input channel predicts: acquisition identity.",
             fontsize=8.0, ha="left", va="top", fontweight="bold")
    fig.text(0.005, 0.951,
             "A HIGH value here is evidence AGAINST the phase hypothesis, not "
             "for it — the label being predicted is the scanner, not the disease.\n"
             "Every confound control stage 5 wrote for the three clinical cohorts "
             "is shown, including the ones near chance;\nthe within-site block "
             "shows every site stratum, including the one below the power floor.",
             fontsize=6.3, ha="left", va="top", color="#444444", linespacing=1.5)
    save(fig, "figS1_acquisition_fingerprint.pdf")

    for row in rows:
        ph, mg = row["by"].get("phase"), row["by"].get("magnitude")
        print(f"    {row['cohort']:13s} {row['target']:18s} "
              f"phase {ph['auc']:.4f} [{ph['lo']:.4f}, {ph['hi']:.4f}]   "
              f"magnitude {mg['auc']:.4f} [{mg['lo']:.4f}, {mg['hi']:.4f}]   "
              f"({ph['k']} clusters, {ph['n']} slices)")
    for name in sorted(cc):
        blk = cc[name]
        a = blk["auc"]["phase"]
        print(f"    {name:13s} phase->{blk['label_target_from_cache']:34s} "
              f"{a['point']:.4f} [{a['lo']:.4f}, {a['hi']:.4f}] "
              f"on {blk['n_test_subjects']} test subjects")
    for w in ws_all:
        print(f"    brain phase, within-site, seed {w['seed']}: "
              f"{w['stratified_auc']:.4f} [{w['ci_lo']:.4f}, {w['ci_hi']:.4f}] "
              f"vs unstratified {w['unstratified_auc']:.4f}"
              f"{'   <- the one drawn' if w is ws else ''}")
    for r in het["per_stratum"]:
        print(f"      site stratum {r['stratum']:4s} seed {r['seed']:>3s}  "
              f"AUROC {r['auc']:.4f}  n={r['n_pos']}+{r['n_neg']}  "
              f"{'counted' if r['counted'] else 'BELOW FLOOR, not counted'}")
    print(f"      s09's own note: {het['note']}")
    print(f"    coil/site separability verdict: {cvs['verdict']['claim']}")
    print(f"    C6 ceiling parsed from the criterion rule: {ceil}")


# --------------------------------------------------------------------------- #
# Figure S2 -- reconstruction fidelity
# --------------------------------------------------------------------------- #
#
# A pure rebuttal figure. It answers exactly one objection -- "your null is a
# broken reconstruction" -- by correlating our cached magnitude against the
# vendor reference shipped in the same HDF5, and it carries no argument of its
# own, which is why it belongs in the supplement.
#
# The honest reading needs the null. r is high partly because any two slices of
# the same body correlate: `r_null_shift` correlates our slice against the vendor
# reference at a DIFFERENT slice of the same volume, so it is the floor that
# shared anatomy alone buys. r_margin = r - r_null_shift is the slice-specific
# part. Both are plotted; quoting r alone would overstate the result.

RECON_ORDER = ["brain", "knee", "prostate_t2", "breast", "prostate_dwi"]


def figS2():
    print("\nFigure S2 -- reconstruction fidelity against the vendor reference "
          "in the same HDF5")
    s = load("pipeline_out/recon_fidelity/recon_fidelity_summary.json",
             "Fig S2, per-cohort summary + verdicts")
    cohorts = [c for c in RECON_ORDER if c in s["cohorts"]]
    if len(cohorts) != len(s["cohorts"]):
        sys.exit(f"recon summary has cohorts this figure does not order: "
                 f"{sorted(set(s['cohorts']) - set(cohorts))}")

    per = {}
    for c in cohorts:
        df = load_csv(f"pipeline_out/recon_fidelity/{c}.csv",
                      f"Fig S2A, {c} per-slice r")
        per[c] = df

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(FULL_W, 3.30),
        gridspec_kw={"width_ratios": [1.95, 1.0], "wspace": 0.40,
                     "left": 0.205, "right": 0.955, "top": 0.815,
                     "bottom": 0.215})

    # ---- A: observed r against the anatomy-support null ------------------- #
    y = np.arange(len(cohorts))[::-1].astype(float)
    dy = 0.20
    n_plotted: dict[str, int] = {}
    for yy, c in zip(y, cohorts):
        df = per[c]
        obs = df["r"].to_numpy(dtype=float)
        nul = df["r_null_shift"].to_numpy(dtype=float)
        obs = obs[np.isfinite(obs)]
        nul = nul[np.isfinite(nul)]
        # The row label must count the slices this row actually DRAWS, not the
        # slices the cohort cached. Breast separates the two: 2,240 slices are
        # cached, 16 have no computable correlation, and 2,224 are plotted --
        # which is the number supplement Table S16 already prints.
        n_plotted[c] = int(obs.size)
        n_summary = int(s["cohorts"][c]["per_slice"]["n"])
        if n_plotted[c] != n_summary:
            sys.exit(f"Fig S2A: {c} plots {n_plotted[c]} finite correlations but "
                     f"the summary's per_slice n is {n_summary}; the figure and "
                     f"the supplement table would disagree")
        for vals, off, col, mk in ((obs, +dy, BLUE, "o"), (nul, -dy, GREY, "s")):
            q = np.percentile(vals, [5, 25, 50, 75, 95])
            axA.plot([q[0], q[4]], [yy + off] * 2, color=col, lw=0.9,
                     solid_capstyle="butt", zorder=2)
            axA.plot([q[1], q[3]], [yy + off] * 2, color=col, lw=4.0,
                     alpha=0.35, solid_capstyle="butt", zorder=2)
            axA.plot([q[2]], [yy + off], marker=mk, ms=4.2, color=col, mec=col,
                     ls="none", zorder=4)
            axA.plot([vals.min()], [yy + off], marker="|", ms=5, mew=0.9,
                     color=col, ls="none", zorder=3)

    axA.set_yticks(y)
    axA.set_yticklabels(
        [f"{COHORT_PRETTY[c]}   {n_plotted[c]:,} slices\n"
         f"ref = {s['cohorts'][c]['reference']}"
         + ("" if s["cohorts"][c]["reference_is_ground_truth"] else " †")
         for c in cohorts],
        fontsize=6.3, linespacing=1.35)
    axA.set_xlim(0.0, 1.03)
    axA.set_ylim(y[-1] - 1.25, y[0] + 0.70)
    axA.set_xlabel("Pearson r vs the vendor reference in the same HDF5\n"
                   "bar = IQR · whisker = 5th–95th centile · tick = minimum",
                   fontsize=7.0)
    # The thresholds the summary counts slices against. Rotated, because at this
    # width 0.90 / 0.95 / 0.99 are 0.02 in apart and would overprint.
    for t in s["thresholds"]:
        axA.axvline(t, color=BLACK, lw=0.6, ls=(0, (1, 3)), zorder=1)
        axA.text(t, y[0] + 0.74, f"{t:g}", fontsize=5.4, color="#555555",
                 ha="center", va="bottom", rotation=90)
    axA.set_title("our reconstruction vs the vendor's, against the floor\n"
                  "that shared anatomy alone buys", pad=10, fontsize=7.6)
    axA.legend(handles=[
        Line2D([], [], marker="o", ls="none", ms=4.2, color=BLUE,
               label="observed — our slice vs the vendor's SAME slice"),
        Line2D([], [], marker="s", ls="none", ms=4.2, color=GREY,
               label=f"anatomy-support null — vs the vendor's slice "
                     f"± {s['null_shift']}"),
    ], loc="lower left", frameon=False, fontsize=5.9, borderaxespad=0.15,
        labelspacing=0.35, bbox_to_anchor=(0.0, -0.02))
    panel_tag(axA, "A", x=-0.335)

    # ---- B: the slice-specific part -------------------------------------- #
    axB.axvline(0.0, color=BLACK, lw=0.8, ls=(0, (4, 3)))
    for yy, c in zip(y, cohorts):
        m = s["cohorts"][c]["anatomy_support_null"]["r_margin"]
        axB.errorbar([m["median"]], [yy],
                     xerr=[[max(0.0, m["median"] - m["p05"])],
                           [max(0.0, m["std"])]],
                     fmt="none", ecolor=ORANGE, elinewidth=1.4, capsize=2.4,
                     capthick=0.9, zorder=3)
        axB.plot([m["median"]], [yy], "o", ms=4.6, color=ORANGE, ls="none",
                 zorder=4)
        axB.text(m["median"], yy + 0.24, f"{m['median']:.3f}", fontsize=6.0,
                 color=ORANGE, ha="center", va="bottom")
    axB.set_yticks(y)
    axB.set_yticklabels([COHORT_PRETTY[c] for c in cohorts], fontsize=6.4)
    axB.set_ylim(y[-1] - 1.25, y[0] + 0.70)
    axB.set_xlabel("r − r(null)\nmedian, 5th centile to +1 SD", fontsize=7.0)
    axB.set_title("the slice-specific part\nof the agreement", pad=10,
                  fontsize=7.6)
    panel_tag(axB, "B", x=-0.42)

    dagger = [c for c in cohorts
              if not s["cohorts"][c]["reference_is_ground_truth"]]
    fig.text(0.005, 0.055,
             "† " + "; ".join(f"{COHORT_PRETTY[c]}: "
                              f"{s['cohorts'][c]['caveat'].split('.')[0]}"
                              for c in dagger) + ".",
             fontsize=5.7, color="#444444", ha="left", va="top")
    save(fig, "figS2_recon_fidelity.pdf")

    if s["discrepancies_vs_documented_claims"]:
        print("    DISCREPANCIES vs documented claims: "
              f"{s['discrepancies_vs_documented_claims']}")
    else:
        print("    discrepancies vs documented claims: none recorded")
    for c in cohorts:
        blk = s["cohorts"][c]
        ps, mg = blk["per_slice"], blk["anatomy_support_null"]["r_margin"]
        v = blk["verdict"]
        print(f"    {c:13s} ref {blk['reference']:19s} per-slice r mean "
              f"{ps['mean']:.5f} min {ps['min']:.5f}  margin median "
              f"{mg['median']:.4f}  verdict {v['status']} "
              f"(strength {v['comparison_strength']})")
        if blk.get("caveat"):
            print(f"                  caveat: {blk['caveat'][:96]}…")


# --------------------------------------------------------------------------- #
# Figure S3 -- the same four channels, every other cohort
# --------------------------------------------------------------------------- #
#
# Same selection rule as Figure 6, applied to the four cohorts the main text does
# not show. Two of these rows are CONFOUND cohorts, whose label is coil count or
# pulse sequence; the row labels say so, taken from verdict.json, because a reader
# who skims this page must not come away thinking brain and knee carry a tumour
# annotation.

QUAL_SUPP = ["prostate_dwi", "breast", "brain", "knee"]


def figS3():
    print("\nFigure S3 -- magnitude and raw phase for the four remaining cohorts")
    import h5py

    require_cache(QUAL_SUPP)
    vd = load("pipeline_out/report/verdict.json",
              "Fig S3, class wording per cohort")

    fig = plt.figure(figsize=(FULL_W, 6.90))
    gs = fig.add_gridspec(
        len(QUAL_SUPP), 4, left=0.150, right=0.965, top=0.862, bottom=0.090,
        wspace=0.07, hspace=0.32)

    im_mag = im_ph = None
    heads: list = [None] * 4
    for r, cohort in enumerate(QUAL_SUPP):
        index = load_index(cohort, f"Fig S3 row {r + 1}, {cohort} slice choice")
        pos_w, neg_w = _class_words(cohort, index, vd)
        h5 = CACHE / f"{cohort}.h5"
        _ledger(f"Fig S3 row {r + 1}, {cohort} pixels",
                str(h5.relative_to(REPO)))
        with h5py.File(h5, "r") as fh:
            for j, (lab, word) in enumerate(((1, pos_w), (0, neg_w))):
                meta = _pick_median_test_row(index, lab)
                if meta is None:
                    sys.exit(f"no test-split row with label={lab} in {cohort}")
                mag, phase, mask = _read_slice(fh, int(meta["idx"]))
                mag_z = (mag - mag.mean()) / (mag.std() + 1e-8)
                v = float(np.percentile(np.abs(mag_z), 99)) or 1.0
                a = fig.add_subplot(gs[r, 2 * j])
                im_mag = _show(a, mag_z, CMAP_MAG, -v, v, mask=mask)
                b = fig.add_subplot(gs[r, 2 * j + 1])
                im_ph = _show(b, phase, CMAP_PHASE, -np.pi, np.pi, mask=mask)
                if r == 0:
                    heads[2 * j], heads[2 * j + 1] = a, b
                a.text(0.0, 1.015, word.replace("\n", " "),
                       transform=a.transAxes, fontsize=6.0, ha="left",
                       va="bottom", fontweight="bold", color="#222222")
                a.text(0.0, -0.035, _prov(meta), transform=a.transAxes,
                       fontsize=5.1, color="#555555", va="top", ha="left")
                if j == 0:
                    a.set_ylabel(COHORT_PRETTY.get(cohort, cohort),
                                 fontsize=8.0, fontweight="bold", labelpad=6)
                    panel_tag(a, "ABCD"[r], x=-0.36, y=0.90)
                print(f"    {cohort:13s} label={lab}  {_prov(meta)}  "
                      f"phase range [{phase.min():+.4f}, {phase.max():+.4f}] rad")

    # Column headings above the class labels, so the two cannot overprint on
    # the first row the way they do when the heading is an axes title.
    for c, (ax_, txt) in enumerate(zip(heads, ["magnitude\nzscore(mag)",
                                              "raw phase\nradians",
                                              "magnitude\nzscore(mag)",
                                              "raw phase\nradians"])):
        box = ax_.get_position()
        fig.text(box.x0 + box.width / 2, box.y1 + 0.032, txt, fontsize=6.6,
                 ha="center", va="bottom", linespacing=1.2)

    for im, x0, lab in ((im_mag, 0.150, "zscore(magnitude), ±99th centile"),
                        (im_ph, 0.560, "raw phase (radians), cyclic scale")):
        cax = fig.add_axes([x0, 0.038, 0.255, 0.010])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.outline.set_linewidth(0.5)
        cb.ax.tick_params(labelsize=5.4, width=0.5, length=2.0, pad=1.2)
        cb.set_label(lab, fontsize=5.8, labelpad=1.5)
        if im is im_ph:
            cb.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
            cb.set_ticklabels(["−π", "−π/2", "0", "π/2", "π"])

    fig.text(0.005, 0.994,
             "The remaining four cohorts. Columns 1–2 are the positive class, "
             "columns 3–4 the negative class.",
             fontsize=7.6, ha="left", va="top", fontweight="bold")
    meanings = "\n".join(
        textwrap.fill(f"{COHORT_PRETTY.get(c, c)} — {_label_meaning(c, vd)}",
                      132, subsequent_indent="    ")
        for c in QUAL_SUPP if c in vd.get("confound_cohorts", {}))
    fig.text(0.005, 0.972,
             "Outline = stage-2 body mask. Class names are stage 2's own "
             "`label_name` values; what each label means, in verdict.json's "
             "words:\n" + meanings,
             fontsize=5.6, ha="left", va="top", color="#444444", linespacing=1.55)
    save(fig, "figS3_qualitative_cohorts.pdf")


# --------------------------------------------------------------------------- #
# verification
# --------------------------------------------------------------------------- #

def verify_pdfs() -> int:
    """
    Check the two properties the journal and the reader both depend on.

      1. No Type-3 font anywhere. Type 3 embeds glyphs as uninterpreted PDF
         drawing operators: the text stops being text, so it cannot be searched,
         copied out of the proof, or re-hinted by the typesetter.
      2. Text is present as text on every page. An imaging figure that had been
         flattened to a bitmap would still open and still look right, and the
         only cheap way to notice is that its axis labels have no font resources.

    Image XObjects are EXPECTED on the imaging figures and are reported, not
    faulted: an MRI slice is raster data and drawing it as vector would be a
    lie about its resolution as well as a 50 MB file.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        print("  pypdf not installed -- SKIPPING font/vector verification")
        return 0

    bad = 0
    print(f"  {'file':<36s} {'fonts':<7s} {'type3':<6s} {'images':<7s} subsets")
    for path in sorted(OUT.glob("*.pdf")):
        reader = PdfReader(str(path))
        fonts: set[str] = set()
        subtypes: set[str] = set()
        n_img = 0
        for page in reader.pages:
            res = page.get("/Resources")
            if res is None:
                continue
            res = res.get_object()
            for name, ref in (res.get("/Font") or {}).items():
                f = ref.get_object()
                fonts.add(str(f.get("/BaseFont", name)).lstrip("/"))
                subtypes.add(str(f.get("/Subtype")).lstrip("/"))
            for _, ref in (res.get("/XObject") or {}).items():
                x = ref.get_object()
                if str(x.get("/Subtype")) == "/Image":
                    n_img += 1
        t3 = "Type3" in subtypes
        # a subset-embedded font is tagged ABCDEF+Name
        subset = all("+" in f for f in fonts) if fonts else False
        ok = fonts and not t3 and subset
        print(f"  {path.name:<36s} {len(fonts):<7d} "
              f"{'YES' if t3 else 'no':<6s} {n_img:<7d} "
              f"{'all subsetted' if subset else 'NOT ALL SUBSET'}"
              f"{'' if ok else '   <-- CHECK'}")
        if not ok:
            bad += 1
        if not fonts:
            print(f"      {path.name} has NO font resources: its text may have "
                  f"been flattened into the raster")
    return bad


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
    fig6()
    figS1()
    figS2()
    figS3()
    print("\n" + "=" * 78)
    print("SOURCE LEDGER")
    for line in _SOURCES:
        print(line)
    print("=" * 78)
    print("VECTOR / FONT VERIFICATION")
    bad = verify_pdfs()
    print("=" * 78)
    if bad:
        sys.exit(f"{bad} figure(s) failed the font/vector check")


if __name__ == "__main__":
    main()
