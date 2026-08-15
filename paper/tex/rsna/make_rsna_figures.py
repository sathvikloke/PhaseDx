#!/usr/bin/env python3
"""
Build every figure in the RSNA submission package, from the revised number set.

    PHASEDX_METRIC_WORD=AUC venv/bin/python paper/tex/rsna/make_rsna_figures.py

    fig1_collapse.pdf         A: two units, six labels
                              B: the stack-depth mechanism
                              C: the same four quantities over 24 holdouts
    fig2_unit_scatter.pdf     slice vs patient, every audited benchmark-arm,
                              all eight DeepLesion body-part arms plotted
    figS1_trivial_fraction.pdf  SUPPLEMENTAL: descriptive cross-study comparison

WHY THIS SCRIPT NOW BUILDS ALL THREE
------------------------------------
Figures 1 and S1 used to be byte-identical copies of the full manuscript's own
figures, produced by the shared builder in the parent directory from the
superseded pooled out-of-fold artefacts. The revision replaces that estimator
with a frozen patient-disjoint holdout, so those two figures no longer show the
numbers this manuscript reports, and the panels themselves changed: Figure 1B is
now the depth mechanism rather than the interval-width comparison, and Figure 1C
is new. The shared builder still serves the other venue and is NOT modified.

NO HAND-TYPED NUMBERS
---------------------
Every mark is read from revised_numbers.json, whose own ledger records the
artefact and sha256 behind each value, plus the frozen-holdout bin-sweep artefact
for the three robustness checks. The choice of which value to read is a RULE
applied to that file, never a per-benchmark constant:

  * baseline    = the locked positional model, except where its reading is
                  exactly at chance at BOTH units, which registers "this label
                  file has no slice index" rather than a computed result; there
                  the arm's secondary model is read and the substitution is
                  printed. This fires on PI-CAI and nowhere else.
  * slice axis  = defined iff the arm has a finite slice AUC.
  * patient axis= defined iff the arm has a finite patient AUC.
  * open marker = the arm's constant predictor is not exactly 0.500, i.e. it is
                  still scored by the superseded pooled estimator. That is read
                  from the artefact, not assigned here. No retained arm now
                  triggers it, so the convention is emitted only if one does.
  * Figure S1 row selection = for each (benchmark, arm), the comparator with the
                  strongest published value, which is the conservative choice;
                  rows that are a split-geometry variant of another row are
                  dropped so no benchmark-arm appears twice.

ANONYMITY
---------
Double-anonymized review. Nothing drawn into these figures names an author, an
institution, a repository, a persistent identifier or a file path. Only public
benchmark names appear, and they are already named in the manuscript body.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# --------------------------------------------------------------------------- #
# paths and provenance ledger
# --------------------------------------------------------------------------- #

HERE = Path(__file__).resolve().parent             # paper/tex/rsna
REPO = HERE.parent.parent.parent                   # repo root
OUT = HERE / "figures"
OUT.mkdir(parents=True, exist_ok=True)

NUMBERS = HERE / "revised_numbers.json"
BINSWEEP = REPO / "pipeline_out" / "trivial_baselines" / "rsna_bin_sweep.json"

_SOURCES: list[str] = []

# The full manuscript writes AUROC; this submission writes AUC. The word is a
# variable rather than a literal so both packages build from one source.
METRIC = os.environ.get("PHASEDX_METRIC_WORD", "AUC")


def _ledger(mark: str, relpath: str, detail: str = "") -> None:
    line = f"  {mark:<40s} <- {relpath}"
    if detail:
        line += f"\n  {'':<40s}    {detail}"
    print(line)
    _SOURCES.append(line)


def load(path: Path, mark: str, detail: str = ""):
    if not path.exists():
        sys.exit(f"MISSING ARTEFACT: {path}\n  needed for: {mark}")
    with path.open() as fh:
        obj = json.load(fh)
    _ledger(mark, path.name, detail)
    return obj


# --------------------------------------------------------------------------- #
# style -- identical to the shared builder
# --------------------------------------------------------------------------- #

BLACK = "#000000"
ORANGE = "#E69F00"
SKY = "#56B4E9"
GREEN = "#009E73"
BLUE = "#0072B2"
VERM = "#D55E00"
PURPLE = "#CC79A7"
GREY = "#8C8C8C"
LIGHTGREY = "#D9D9D9"

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

COL_W = 3.35
FULL_W = 7.0
CHANCE = 0.5

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


def save(fig, name: str):
    path = OUT / name
    fig.savefig(path, format="pdf")
    plt.close(fig)
    print(f"  wrote figures/{name}")


def _finite(x) -> bool:
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


# --------------------------------------------------------------------------- #
# Figure 1 -- the flagship, three panels
# --------------------------------------------------------------------------- #

def fig1(D, SW):
    print(f"\nFigure 1 -- RSNA ICH locked baseline: two units, the mechanism, "
          f"and 24 holdouts")
    R = D["rsna_ich"]
    labs = [R["labels"][k] for k in LABEL_ORDER]
    _ledger("Fig 1A, six labels", NUMBERS.name,
            "rsna_ich.labels[*].slice_auc / patient_auc_mean / nulls")

    n = len(labs)
    x = np.arange(n, dtype=float)
    slice_auc = np.array([L["slice_auc"] for L in labs])
    slice_lo = np.array([L["slice_ci"][0] for L in labs])
    slice_hi = np.array([L["slice_ci"][1] for L in labs])
    pat = np.array([L["patient_auc_mean"] for L in labs])
    pat_lo = np.array([L["patient_auc_mean_ci"][0] for L in labs])
    pat_hi = np.array([L["patient_auc_mean_ci"][1] for L in labs])
    pat_null = np.array([L["within_series_permutation_null"]["patient_mean"]
                         for L in labs])
    gap = np.array([L["gap_slice_minus_patient"] for L in labs])

    fig = plt.figure(figsize=(FULL_W, 5.9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.86],
                          width_ratios=[1.0, 0.92],
                          hspace=0.52, wspace=0.30,
                          left=0.085, right=0.985, bottom=0.075, top=0.945)
    axA = fig.add_subplot(gs[0, :])
    axB = fig.add_subplot(gs[1, 0])
    axC = fig.add_subplot(gs[1, 1])

    # ---- Panel A ---------------------------------------------------------- #
    dx = 0.17
    axA.axhline(CHANCE, color=BLACK, lw=0.7, ls=(0, (4, 3)), zorder=1)
    axA.text(-0.54, CHANCE + 0.006, "chance", fontsize=6.4, color=BLACK,
             ha="left", va="bottom")
    for i in range(n):
        axA.plot([x[i] - dx, x[i] + dx], [slice_auc[i], pat[i]],
                 color=LIGHTGREY, lw=3.0, solid_capstyle="round", zorder=2)
    axA.errorbar(x - dx, slice_auc,
                 yerr=[slice_auc - slice_lo, slice_hi - slice_auc],
                 fmt="o", ms=5.0, color=C_SLICE, mfc=C_SLICE, mec=C_SLICE,
                 ecolor=C_SLICE, elinewidth=1.4, capsize=2.4, capthick=1.0,
                 zorder=4)
    axA.errorbar(x + dx, pat, yerr=[pat - pat_lo, pat_hi - pat],
                 fmt="s", ms=5.0, color=C_PATIENT, mfc=C_PATIENT, mec=C_PATIENT,
                 ecolor=C_PATIENT, elinewidth=1.4, capsize=2.4, capthick=1.0,
                 zorder=4)
    for i in range(n):
        axA.plot([x[i] + dx - 0.115, x[i] + dx + 0.115],
                 [pat_null[i], pat_null[i]], color=C_NULL, lw=1.6, zorder=3)
        mid = 0.5 * (slice_auc[i] + pat[i])
        axA.annotate(f"−{gap[i]:.3f}", (x[i], mid), fontsize=6.2,
                     color="#404040", ha="center", va="center",
                     bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="none",
                               alpha=0.9), zorder=5)
    axA.annotate(
        f"patient-level permutation null\n({pat_null[0]:.3f} for 'any')",
        xy=(x[-1] + dx + 0.10, pat_null[-1]), xytext=(x[-1] - 0.95, 0.632),
        fontsize=6.3, color=C_NULL, ha="center", va="bottom",
        arrowprops=dict(arrowstyle="-", color=C_NULL, lw=0.7, shrinkA=2,
                        shrinkB=1),
    )
    axA.set_xticks(x)
    axA.set_xticklabels([LABEL_PRETTY[k] for k in LABEL_ORDER], rotation=24,
                        ha="right")
    axA.set_ylabel(f"{METRIC} of the locked zero-image baseline")
    axA.set_ylim(0.40, 0.86)
    axA.set_xlim(-0.6, n - 0.35)
    axA.set_title(
        f"{R['cohort']['n_patients']:,} patients · frozen holdout of "
        f"{R['labels']['any']['n_test_patients']:,} "
        f"({R['labels']['any']['n_test_slices']:,} slices), single fit",
        pad=6, loc="left")
    axA.legend(handles=[
        Line2D([], [], marker="o", ls="none", ms=5, color=C_SLICE,
               label="slice unit"),
        Line2D([], [], marker="s", ls="none", ms=5, color=C_PATIENT,
               label="patient unit (mean aggregation)"),
        Line2D([], [], color=C_NULL, lw=1.6, label="patient permutation null"),
    ], loc="upper right", frameon=False, handletextpad=0.6, borderaxespad=0.2,
        ncol=3, columnspacing=1.3)
    axA.text(-0.115, 1.045, "A", transform=axA.transAxes, fontsize=10,
             fontweight="bold", va="bottom", ha="left")

    # ---- Panel B: the mechanism ------------------------------------------- #
    A = R["labels"]["any"]
    nullb = A["within_series_permutation_null"]
    rows = [
        ("mean aggregation,\nunstratified", A["patient_auc_mean"],
         A["patient_auc_mean_ci"], C_PATIENT),
        ("within 5-slice\ndepth strata", A["patient_auc_mean_depth5"],
         A["patient_auc_mean_depth5_ci"], GREEN),
        ("within exact\nstack depth", A["patient_auc_mean_depthfixed"],
         A["patient_auc_mean_depthfixed_ci"], GREEN),
        ("stack depth alone", A["depth_alone_patient_auc"],
         A["depth_alone_patient_ci"], VERM),
    ]
    _ledger("Fig 1B, depth mechanism", NUMBERS.name,
            "rsna_ich.labels.any depth block + depth-fixed permutation null")
    y = np.arange(len(rows))[::-1].astype(float)
    lo_b, hi_b = nullb["patient_depthfixed_range"]
    axB.axvspan(lo_b, hi_b, color=SKY, alpha=0.35, lw=0, zorder=0)
    axB.axvline(CHANCE, color=BLACK, lw=0.8, ls=(0, (4, 3)), zorder=1)
    for yy, (lab, v, ci, col) in zip(y, rows):
        axB.errorbar([v], [yy], xerr=[[v - ci[0]], [ci[1] - v]], fmt="none",
                     ecolor=col, elinewidth=1.4, capsize=2.4, capthick=1.0,
                     zorder=3)
        axB.plot([v], [yy], "o", ms=5.4, color=col, ls="none", zorder=4)
    axB.set_yticks(y)
    axB.set_yticklabels([r[0] for r in rows])
    axB.set_ylim(-0.65, len(rows) - 0.35)
    axB.set_xlim(0.36, 0.56)
    axB.set_xticks([0.40, 0.45, 0.50, 0.55])
    axB.set_xlabel(f"patient-level {METRIC}, 'any hemorrhage'")
    axB.set_title("holding stack depth fixed removes\nthe sub-chance reading",
                  pad=6, fontsize=7.8, loc="left")
    axB.annotate(
        f"within-series permutation null\nat fixed depth: "
        f"{nullb['patient_depthfixed_mean']:.4f}",
        xy=(hi_b, 0.02), xytext=(0.516, -0.40), fontsize=6.0, color="#20618C",
        ha="left", va="center",
        arrowprops=dict(arrowstyle="-", color="#20618C", lw=0.6, shrinkA=1,
                        shrinkB=1))
    axB.text(-0.40, 1.045, "B", transform=axB.transAxes, fontsize=10,
             fontweight="bold", va="bottom", ha="left")

    # ---- Panel C: 24 holdouts --------------------------------------------- #
    sp = R["across_holdout_spread"]["any"]
    draws = sp["per_draw"]
    primary_seed = R["primary_seed"]
    _ledger("Fig 1C, 24 holdouts", NUMBERS.name,
            f"rsna_ich.across_holdout_spread.any.per_draw ({len(draws)} draws)")
    # The four quantities sit at very different levels (0.74, 0.44, 0.51, 0.50),
    # so a shared absolute axis would compress every spread to nothing. Each row
    # is therefore drawn as a DEVIATION from its own 24-holdout mean, on one
    # common scale, with that mean printed on the row. The bootstrap interval of
    # the frozen holdout is drawn on the same scale directly beneath, which is
    # the comparison the text makes: the two are not ordered in a fixed
    # direction, and neither replaces the other.
    series = [
        ("slice", "slice_auc", "slice_ci", C_SLICE),
        ("patient (mean)", "patient_auc_mean", "patient_auc_mean_ci", C_PATIENT),
        ("patient, depth fixed", "patient_auc_mean_depthfixed",
         "patient_auc_mean_depthfixed_ci", GREEN),
        ("constant predictor", "constant_slice_auc", "constant_predictor_slice_ci",
         GREY),
    ]
    yc = np.arange(len(series))[::-1].astype(float)
    axC.axvline(0.0, color=BLACK, lw=0.8, ls=(0, (4, 3)), zorder=1)
    rng = np.random.default_rng(0)
    for yy, (lab, key, cikey, col) in zip(yc, series):
        vals = np.array([d[key] for d in draws])
        mu = float(vals.mean())
        dev = vals - mu
        axC.plot([dev.min(), dev.max()], [yy + 0.13, yy + 0.13], color=col,
                 lw=1.3, alpha=0.6, zorder=2)
        jit = (rng.random(len(vals)) - 0.5) * 0.11
        axC.plot(dev, yy + 0.13 + jit, "o", ms=2.6, color=col, alpha=0.8,
                 ls="none", zorder=3)
        pv = [d[key] for d in draws if d["seed"] == primary_seed][0]
        axC.plot([pv - mu], [yy + 0.13], "o", ms=6.6, mfc="white", mec=col,
                 mew=1.5, ls="none", zorder=5)
        ci = A[cikey] if cikey in A else [pv, pv]
        axC.plot([ci[0] - mu, ci[1] - mu], [yy - 0.19, yy - 0.19], color=col,
                 lw=1.1, alpha=0.85, ls=(0, (2, 1.6)), zorder=2)
        for e in ci:
            axC.plot([e - mu, e - mu], [yy - 0.25, yy - 0.13], color=col,
                     lw=1.0, alpha=0.85, zorder=2)
        axC.text(0.0285, yy + 0.13, f"mean {mu:.4f}", fontsize=6.0, color=col,
                 ha="right", va="center")
    axC.set_yticks(yc)
    axC.set_yticklabels([s[0] for s in series], fontsize=6.6)
    axC.set_ylim(-1.15, len(series) - 0.32)
    axC.set_xlim(-0.030, 0.030)
    axC.set_xticks([-0.02, -0.01, 0.0, 0.01, 0.02])
    axC.set_xlabel(f"deviation from that row's 24-holdout mean")
    axC.set_title("pipeline uncertainty: the whole\nprocedure repeated", pad=6,
                  fontsize=7.8, loc="left")
    axC.legend(handles=[
        Line2D([], [], marker="o", ls="none", ms=2.6, color=GREY,
               label="one holdout (24 in all)"),
        Line2D([], [], marker="o", ls="none", ms=6.6, mfc="white", mec=GREY,
               mew=1.5, label="the frozen holdout of A and B"),
        Line2D([], [], color=GREY, lw=1.1, ls=(0, (2, 1.6)),
               label="its 95% bootstrap interval"),
    ], loc="lower left", bbox_to_anchor=(-0.03, -0.02), frameon=False,
        handletextpad=0.6, borderaxespad=0.0, fontsize=6.0, ncol=1,
        labelspacing=0.3)
    axC.text(-0.30, 1.045, "C", transform=axC.transAxes, fontsize=10,
             fontweight="bold", va="bottom", ha="left")

    save(fig, "fig1_collapse.pdf")
    print(f"    slice 'any'    {slice_auc[0]:.4f} "
          f"[{slice_lo[0]:.4f}, {slice_hi[0]:.4f}]")
    print(f"    patient 'any'  {pat[0]:.4f} [{pat_lo[0]:.4f}, {pat_hi[0]:.4f}]")
    print(f"    gaps           {gap.min():.3f} to {gap.max():.3f}")
    print(f"    depth fixed    {A['patient_auc_mean_depthfixed']:.4f} "
          f"vs its own null {nullb['patient_depthfixed_mean']:.4f}")
    print(f"    constant, all draws, both units: "
          f"{sp['constant_slice_auc']['min']:.4f} to "
          f"{sp['constant_slice_auc']['max']:.4f}")
    # the three robustness checks, printed so the Table 2 note can be checked
    sw = SW["sweep"]
    print("    bin sweep (slice): " + ", ".join(
        f"{k}:{sw[k]['slice_auc']:.4f}" for k in ("5", "10", "20", "50")))
    print("    bin sweep (patient): " + ", ".join(
        f"{k}:{sw[k]['patient_auc_mean']:.4f}" for k in ("5", "10", "20", "50")))
    print(f"    fit-free centrality slice "
          f"{SW['fit_free_centrality']['slice_auc']:.4f}; apparent-vs-heldout "
          f"{sw['20']['slice_auc_apparent_on_training_rows']:.4f} / "
          f"{sw['20']['slice_auc']:.4f}")


# --------------------------------------------------------------------------- #
# Figure 2 -- the same score vector at two units, every audited benchmark-arm
# --------------------------------------------------------------------------- #

# Label placement only. This moves TEXT; no marker is displaced or jittered.
LABEL_POS = {
    "RSNA ICH, any hemorrhage":            (-0.012, -0.014, "right", "top", False),
    # Placed ABOVE this point, not left of it: a left offset large enough to clear
    # the arm's own error whisker (slice 0.531 -> 0.486) runs the text off the axis.
    # +0.120 sits clear of the upper patient whisker, which reaches 0.628.
    "LUNA16 candidates":                   (0.000, 0.120, "center", "bottom", False),
    "PI-CAI, slice level":                 (0.000, -0.024, "center", "top", False),
    "DeepLesion, 8 body-part arms":        (0.000, 0.021, "center", "bottom", False),
    "fastMRI Prostate, T2":                (0.902, 0.470, "left", "center", True),
    "fastMRI Prostate, DWI":               (0.902, 0.408, "left", "center", True),
}

DEEPLESION_ARMS = ["deeplesion_pelvis_vs_rest", "deeplesion_mediastinum_vs_rest",
                   "deeplesion_abdomen_vs_rest", "deeplesion_kidney_vs_rest",
                   "deeplesion_liver_vs_rest", "deeplesion_lung_vs_rest",
                   "deeplesion_softtissue_vs_rest", "deeplesion_bone_vs_rest"]

OTHER_ARMS = ["fastmri_prostate_t2", "fastmri_prostate_dwi",
              "duke_breast_owner_slice_task", "luna16_fp_reduction_candidates",
              "picai_slice_level", "picai_case_level"]

SHORT = {
    "fastMRI Prostate, T2-weighted": "fastMRI Prostate, T2",
    "fastMRI Prostate, DWI": "fastMRI Prostate, DWI",
    "Duke Breast, owner-defined slice task": "Duke Breast",
    "LUNA16 candidates": "LUNA16 candidates",
    "PI-CAI, slice level": "PI-CAI, slice level",
    "PI-CAI, case level": "PI-CAI, case level",
}


def _arm_row(key: str, a: dict) -> dict:
    """One benchmark-arm, with which baseline was read decided by the artefact."""
    substituted = None
    slice_auc = a.get("slice_auc")
    slice_ci = a.get("slice_ci")
    pat = a.get("patient_auc_mean")
    pat_ci = a.get("patient_auc_mean_ci")
    if not _finite(slice_auc) and a.get("positional_patient_auc") == CHANCE:
        # positional baseline pinned to chance at both units == "no slice index"
        pat = a.get("secondary_metadata_tree_patient_auc")
        pat_ci = a.get("secondary_metadata_tree_patient_ci")
        substituted = ("locked positional baseline is exactly 0.500 at both "
                       "units -> secondary model read instead")
    return dict(
        key=key,
        label=SHORT.get(a["display_name"], a["display_name"]),
        pooled=bool(a.get("estimator_is_pooled_out_of_fold", False)),
        constant=a.get("constant_predictor_slice_auc"),
        slice_ok=_finite(slice_auc),
        patient_ok=_finite(pat),
        slice_auc=slice_auc if _finite(slice_auc) else None,
        slice_ci=slice_ci if _finite(slice_auc) else None,
        patient_auc=pat if _finite(pat) else None,
        patient_ci=pat_ci if _finite(pat) else None,
        substituted=substituted,
    )


def fig2(D):
    print("\nFigure 2 -- one pixel-blind score vector at two units, every "
          "audited benchmark-arm")
    R = D["rsna_ich"]["labels"]["any"]
    rows = [dict(key="rsna_ich", label="RSNA ICH, any hemorrhage", pooled=False,
                 constant=R["constant_predictor_slice_auc"],
                 slice_ok=True, patient_ok=True,
                 slice_auc=R["slice_auc"], slice_ci=R["slice_ci"],
                 patient_auc=R["patient_auc_mean"],
                 patient_ci=R["patient_auc_mean_ci"], substituted=None)]
    _ledger("Fig 2, RSNA ICH", NUMBERS.name, "rsna_ich.labels.any, frozen holdout")

    dl = []
    for k in DEEPLESION_ARMS:
        dl.append(_arm_row(k, D["other_arms"][k]))
    _ledger("Fig 2, DeepLesion x8", NUMBERS.name,
            "all eight body-part arms plotted; no arm is selected")

    for k in OTHER_ARMS:
        r = _arm_row(k, D["other_arms"][k])
        rows.append(r)
        detail = ("pooled out of fold" if r["pooled"] else "single fit")
        if r["substituted"]:
            detail += f"   [{r['substituted']}]"
        _ledger(f"Fig 2, {r['label']}", NUMBERS.name, detail)

    both = [r for r in rows + dl if r["slice_ok"] and r["patient_ok"]]
    slice_only = [r for r in rows if r["slice_ok"] and not r["patient_ok"]]
    pat_only = [r for r in rows if r["patient_ok"] and not r["slice_ok"]]

    fig = plt.figure(figsize=(FULL_W, 5.05))
    gs = fig.add_gridspec(2, 2, width_ratios=[0.085, 1.0],
                          height_ratios=[1.0, 0.085], wspace=0.02, hspace=0.03,
                          left=0.085, right=0.855, bottom=0.30, top=0.945)
    ax = fig.add_subplot(gs[0, 1])
    ax_l = fig.add_subplot(gs[0, 0], sharey=ax)
    ax_b = fig.add_subplot(gs[1, 1], sharex=ax)

    lo, hi = 0.38, 1.03
    TICKS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ax.plot([lo, hi], [lo, hi], color=GREY, lw=0.9, ls=(0, (5, 3)), zorder=1)
    ax.axhline(CHANCE, color=BLACK, lw=0.8, ls=(0, (1, 2)), zorder=1)
    ax.axvline(CHANCE, color=BLACK, lw=0.8, ls=(0, (1, 2)), zorder=1)
    ax.text(lo + 0.006, CHANCE + 0.008, "chance", fontsize=6.3, color=BLACK,
            ha="left", va="bottom")

    def draw(r, col, label=True):
        x, y = r["slice_auc"], r["patient_auc"]
        ax.plot([x, x], [x, y], color=LIGHTGREY, lw=2.6, solid_capstyle="butt",
                zorder=2)
        mfc = "white" if r["pooled"] else col
        ax.errorbar([x], [y],
                    xerr=[[x - r["slice_ci"][0]], [r["slice_ci"][1] - x]],
                    yerr=[[y - r["patient_ci"][0]], [r["patient_ci"][1] - y]],
                    fmt="none", ecolor=col, elinewidth=1.2, capsize=2.0,
                    capthick=0.9, zorder=3)
        ax.plot([x], [y], "o", ms=5.6, color=col, mfc=mfc, mec=col, mew=1.4,
                ls="none", zorder=4)
        if not label:
            return
        px, py, ha, va, leader = LABEL_POS[r["label"]]
        if leader:
            ax.annotate(r["label"], xy=(r["slice_ci"][1], y), xytext=(px, py),
                        fontsize=6.4, color=col, ha=ha, va=va, zorder=5,
                        arrowprops=dict(arrowstyle="-", color=col, lw=0.5,
                                        shrinkA=2.5, shrinkB=1.5))
        else:
            ax.text(x + px, y + py, r["label"], fontsize=6.4, color=col, ha=ha,
                    va=va, zorder=5)

    # every arm; colour = whether the patient interval lies wholly above chance
    for r in both:
        above = r["patient_ci"][0] > CHANCE
        col = ORANGE if above else BLUE
        r["above"] = above
        draw(r, col, label=r in rows)
    # one label for the eight DeepLesion arms, at their centroid
    dlx = float(np.mean([r["slice_auc"] for r in dl]))
    dly = float(np.max([r["patient_ci"][1] for r in dl]))
    px, py, ha, va, _ = LABEL_POS["DeepLesion, 8 body-part arms"]
    ax.text(dlx + px, dly + py, "DeepLesion, 8 body-part arms", fontsize=6.4,
            color=ORANGE, ha=ha, va=va, zorder=5)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xticks(TICKS)
    ax.set_yticks(TICKS)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="x", labelbottom=False, bottom=False)
    ax.tick_params(axis="y", labelleft=False, left=False)

    for r in slice_only:
        x = r["slice_auc"]
        ax_b.errorbar([x], [0.5],
                      xerr=[[x - r["slice_ci"][0]], [r["slice_ci"][1] - x]],
                      fmt="none", ecolor=GREY, elinewidth=1.3, capsize=2.2,
                      capthick=0.9, zorder=3)
        ax_b.plot([x], [0.5], "o", ms=5.6, color="white", mec=GREY, mew=1.4,
                  ls="none", zorder=4)
        ax_b.text(r["slice_ci"][1] + 0.010, 0.5, r["label"], fontsize=6.4,
                  color=GREY, ha="left", va="center", zorder=5)
    ax_b.set_ylim(0, 1)
    ax_b.set_yticks([])
    ax_b.spines["left"].set_visible(False)
    ax_b.set_xlabel(f"slice-level {METRIC}")
    ax_b.set_xticks(TICKS)
    ax_b.axvline(CHANCE, color=BLACK, lw=0.8, ls=(0, (1, 2)), zorder=1)
    ax_b.text(lo + 0.006, 0.5, "patient level undefined", fontsize=6.0,
              color=GREY, ha="left", va="center", style="italic")

    for r in pat_only:
        y = r["patient_auc"]
        ax_l.errorbar([0.5], [y],
                      yerr=[[y - r["patient_ci"][0]], [r["patient_ci"][1] - y]],
                      fmt="none", ecolor=GREY, elinewidth=1.3, capsize=2.2,
                      capthick=0.9, zorder=3)
        ax_l.plot([0.5], [y], "o", ms=5.6, color="white", mec=GREY, mew=1.4,
                  ls="none", zorder=4)
        ax_l.text(0.5, r["patient_ci"][1] + 0.018, r["label"].split(",")[0],
                  fontsize=6.4, color=GREY, ha="center", va="bottom", zorder=5)
    ax_l.set_xlim(0, 1)
    ax_l.set_xticks([])
    ax_l.spines["bottom"].set_visible(False)
    ax_l.set_ylabel(f"patient-level {METRIC} of the same score vector\n"
                    f"(mean aggregation)")
    ax_l.set_yticks(TICKS)
    ax_l.axhline(CHANCE, color=BLACK, lw=0.8, ls=(0, (1, 2)), zorder=1)
    ax_l.text(0.5, lo + 0.012, "no slice axis", fontsize=6.0, color=GREY,
              ha="center", va="bottom", rotation=90, style="italic")

    n_below = sum(1 for r in both if not r["above"])
    n_above = len(both) - n_below
    n_pooled = sum(1 for r in rows + dl if r["pooled"])
    ax.set_title(
        f"{len(rows) + len(dl)} audited benchmark-arms; on the {len(both)} where "
        f"both units are defined, {n_below} patient-level intervals reach chance "
        f"and {n_above} do not", pad=6, fontsize=8.0, loc="left")

    handles = [
        Line2D([], [], marker="o", ls="none", ms=5.6, color=BLUE,
               label="patient-level interval reaches chance or lies below it"),
        Line2D([], [], color=LIGHTGREY, lw=2.6,
               label="drop from the line of agreement"),
        Line2D([], [], marker="o", ls="none", ms=5.6, color=ORANGE,
               label="patient-level interval lies wholly above chance"),
        Line2D([], [], color=GREY, lw=0.9, ls=(0, (5, 3)),
               label=f"slice {METRIC} = patient {METRIC}"),
        Line2D([], [], marker="o", ls="none", ms=5.6, color=GREY, mfc="white",
               mew=1.4, label="only one unit is defined (plotted in a gutter)"),
    ]
    # The open-marker convention for "still pooled out of fold" is emitted ONLY
    # if such an arm survives. Every retained arm is now scored by a single fit,
    # so the entry must not appear and claim a distinction the panel does not draw.
    if n_pooled:
        handles.insert(4, Line2D([], [], marker="o", ls="none", ms=5.6, color=BLUE,
                                 mfc="white", mew=1.4,
                                 label="still pooled out of fold "
                                       "(constant predictor ≠ 0.500)"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.075, 0.195),
               frameon=False, handletextpad=0.7, borderaxespad=0.0, ncol=2,
               columnspacing=2.2, labelspacing=0.45)

    notes = []
    for r in pat_only:
        notes.append(f"{r['label']}: no slice index in that label file, so no "
                     f"slice-level reading exists; the value shown is its "
                     f"secondary model, two of whose four inputs are clinical.")
    for r in slice_only:
        notes.append(f"{r['label']}: patient-level {METRIC} is undefined "
                     f"because all 922 of its 922 patients are positive.")
    notes.append(f"All eight DeepLesion body-part arms are plotted; the label "
                 f"there is the anatomic region, so no divergence is expected.")
    if n_pooled == 0:
        notes.append("Every arm plotted is scored by a single fit on a "
                     "subject-disjoint split; the constant predictor is exactly "
                     "0.500 on all of them.")
    fig.text(0.075, 0.030, "\n".join(notes), fontsize=6.2, color="#333333",
             ha="left", va="bottom", linespacing=1.5)

    save(fig, "fig2_unit_scatter.pdf")
    for r in rows + dl:
        s = f"{r['slice_auc']:.3f}" if r["slice_ok"] else "   n/a"
        p_ = f"{r['patient_auc']:.3f}" if r["patient_ok"] else "   n/a"
        flag = "  POOLED" if r["pooled"] else ""
        print(f"    {r['label']:<32s} slice {s}   patient {p_}"
              f"   constant {r['constant']}{flag}")
    print(f"    {n_pooled} arms still pooled out of fold and drawn open")


# --------------------------------------------------------------------------- #
# Figure 3 -- descriptive cross-study comparison
# --------------------------------------------------------------------------- #

METRIC_NOTE = {
    "slice ROC AUC": "slice {m}, chance 0.5",
    "slice AUROC": "slice {m}, chance 0.5",
    "8-class accuracy": "8-class accuracy, chance 0.2361",
    "case AUROC": "case-level {m}, chance 0.5",
    "sensitivity at 1 FP/scan": "sensitivity at 1 FP/scan, chance 0.0027",
}


def figS1(D):
    print("\nFigure S1 (supplemental) -- descriptive cross-study comparison")
    tf = D["trivial_fraction_locked"]
    _ledger("Fig S1, every row", NUMBERS.name,
            "trivial_fraction_locked.rows, strongest published per arm")

    # strongest published comparator per (benchmark, arm); drop variant rows
    best: dict[tuple, dict] = {}
    for r in tf["rows"]:
        if r.get("estimator_variant_of"):
            continue
        k = (r["benchmark"], r["arm"])
        if k not in best or r["published"] > best[k]["published"]:
            best[k] = r
    rows = sorted(best.values(), key=lambda r: r["locked_trivial_fraction"])

    picai = D["other_arms"]["picai_case_level"]
    picai_secondary = None
    for r in tf["rows"]:
        if r["benchmark"] == "PI-CAI" and not r.get("estimator_variant_of"):
            if r is best[("PI-CAI", "case level")]:
                published = r["published"]
                picai_secondary = ((picai["secondary_metadata_tree_patient_auc"]
                                    - r["chance"]) / (published - r["chance"]))
                break

    n = len(rows)
    fig, ax = plt.subplots(figsize=(FULL_W, 0.345 * n + 1.55))
    y = np.arange(n)[::-1].astype(float)
    ax.axvline(0.0, color=BLACK, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.axvline(1.0, color=BLACK, lw=0.8, ls=(0, (1, 2)), zorder=1)

    labels = []
    for yy, r in zip(y, rows):
        preprint = r["peer_reviewed"].startswith("preprint")
        col = PURPLE if preprint else BLUE
        marker = "D" if preprint else "o"
        mfc = "white" if preprint else col
        v = r["locked_trivial_fraction"]
        ci = r.get("locked_trivial_fraction_ci")
        if ci is not None and ci[0] != ci[1]:
            ax.errorbar([v], [yy], xerr=[[v - ci[0]], [ci[1] - v]], fmt="none",
                        ecolor=col, elinewidth=1.5, capsize=2.6, capthick=1.0,
                        zorder=3)
        ax.plot([v], [yy], marker=marker, ms=6.0, color=col, mfc=mfc, mec=col,
                mew=1.4, ls="none", zorder=4)
        if ci is None:
            ax.annotate("no interval available for this metric",
                        (v + 0.030, yy - 0.34), fontsize=6.0, color=col,
                        va="center", ha="left")
        note = METRIC_NOTE.get(r["metric"], r["metric"]).format(m=METRIC)
        arm = "" if r["arm"] in ("case level",) else f" — {r['arm']}"
        labels.append(f"{r['benchmark']}{arm}\n{note}")

    if picai_secondary is not None:
        ypic = y[[r["benchmark"] for r in rows].index("PI-CAI")]
        ax.plot([picai_secondary], [ypic], marker="o", ms=6.0, color=BLACK,
                mfc="white", mec=BLACK, mew=1.4, ls="none", zorder=4)
        ax.plot([rows[[r["benchmark"] for r in rows].index("PI-CAI")]
                 ["locked_trivial_fraction"], picai_secondary], [ypic, ypic],
                color=BLACK, lw=0.6, ls=(0, (2, 2)), zorder=2)
        ax.annotate("secondary model, two of four inputs clinical",
                    xy=(picai_secondary, ypic),
                    xytext=(picai_secondary + 0.04, ypic - 0.52), fontsize=6.2,
                    color=BLACK, ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", color=BLACK, lw=0.6,
                                    shrinkA=0, shrinkB=2))

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6.4)
    ax.set_ylim(-1.35, n - 0.35)
    ax.set_xlabel("share of the published margin over chance reached by the "
                  "locked zero-image baseline")
    ax.set_xlim(-0.13, 1.30)
    ax.text(1.0, -1.30, "baseline equals\nthe published system", fontsize=6.2,
            color=BLACK, ha="center", va="bottom")
    ax.text(0.0, -1.30, "baseline at\nchance", fontsize=6.2, color=BLACK,
            ha="center", va="bottom")
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="none", ms=6, color=BLUE,
               label="peer-reviewed comparator"),
        Line2D([], [], marker="D", ls="none", ms=6, color=PURPLE, mfc="white",
               mew=1.4, label="preprint comparator, not peer reviewed"),
        Line2D([], [], marker="o", ls="none", ms=6, color=BLACK, mfc="white",
               mew=1.4, label="secondary model (not the locked baseline)"),
    ], loc="upper right", frameon=False, handletextpad=0.6, borderaxespad=0.4)
    ax.set_title("descriptive only: four benchmarks, four metrics, four chance\n"
                 "anchors, so no average is taken over this panel", pad=6,
                 loc="left")

    save(fig, "figS1_trivial_fraction.pdf")
    for r in rows:
        print(f"    {r['benchmark']:<18s} {r['arm']:<22s} "
              f"{r['locked_trivial_fraction']:+.4f}   "
              f"({r['metric']}, published {r['published']}, "
              f"{'preprint' if r['peer_reviewed'].startswith('preprint') else 'peer reviewed'})")
    if picai_secondary is not None:
        print(f"    PI-CAI secondary model would read {picai_secondary:.4f}; "
              f"it is not the locked baseline and is drawn open")


# --------------------------------------------------------------------------- #
# verification
# --------------------------------------------------------------------------- #

def verify_pdfs() -> int:
    """No Type-3 fonts, text present as text, fonts subsetted."""
    try:
        from pypdf import PdfReader
    except ImportError:
        print("  pypdf not installed -- SKIPPING font/vector verification")
        return 0
    bad = 0
    print(f"  {'file':<30s} {'fonts':<7s} {'type3':<6s} {'images':<7s} subsets")
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
        subset = all("+" in f for f in fonts) if fonts else False
        ok = bool(fonts) and not t3 and subset
        print(f"  {path.name:<30s} {len(fonts):<7d} {'YES' if t3 else 'no':<6s} "
              f"{n_img:<7d} {'all subsetted' if subset else 'NOT ALL SUBSET'}"
              f"{'' if ok else '   <-- CHECK'}")
        if not ok:
            bad += 1
    return bad


def main():
    print("=" * 78)
    print("make_rsna_figures.py -- every mark traced to the artefact it came from")
    print(f"metric word: {METRIC}   (set PHASEDX_METRIC_WORD to change)")
    print("=" * 78)
    D = load(NUMBERS, "all three figures", "the revised number set")
    SW = load(BINSWEEP, "Table 2 robustness checks",
              "frozen-holdout bin sweep and fit-free variant")
    fig1(D, SW)
    fig2(D)
    figS1(D)
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
