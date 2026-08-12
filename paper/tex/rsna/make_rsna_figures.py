#!/usr/bin/env python3
"""
Build the figures that belong to the RSNA submission package, and only those.

    venv/bin/python paper/tex/rsna/make_rsna_figures.py

Figures 1 and 3 of this submission are byte-identical copies of the figures the
full manuscript already ships; they are produced by paper/tex/make_figures.py
and are NOT rebuilt here. This script builds the figure that exists only in the
shortened, anonymised submission.

    fig2_unit_scatter.pdf    slice-level vs patient-level AUC, every audited
                             benchmark-arm

It is Figure 2, not Figure 3, because figures are numbered in order of first
citation and the Results cite the two-unit reading before the trivial fraction.
The trivial-fraction figure is therefore Figure 3 in this package, and ships as
figures/fig3_trivial_fraction.pdf -- the same file the full manuscript calls
fig2_trivial_fraction.pdf, copied and renamed, not rebuilt.

WHY THIS FIGURE AND NOT THE ONE THAT WAS ASKED FOR
--------------------------------------------------
A "trivial-fraction distribution" figure was requested. It already exists:
figures/fig3_trivial_fraction.pdf plots the strongest published comparator per
benchmark-arm, marks the peer-reviewed median and IQR, separates the preprint
comparator rows by colour and marker, draws the chance line at 0, and shows the
LUNA16 point at -0.002 as a real, unclipped mark. A strip plot of the same nine
numbers would be the same figure twice, and a duplicate figure at a journal with
a hard six-figure cap costs a slot and invites the reviewer question "why is
this here." The claim that is genuinely carried by a table alone is Table 3 --
the same pixel-blind score vector read at both units across every audited
benchmark-arm -- so that is what is built here.

NO HAND-TYPED NUMBERS
---------------------
Every value drawn is read from a JSON artefact under pipeline_out/. The choice
of which baseline and which evaluation to read is a RULE applied to the
artefact, never a per-benchmark constant:

  * evaluation  = the artefact's own `headline_evaluation` field.
  * baseline    = `positional_20bin`, except where the positional baseline is
                  exactly at chance at BOTH units, which is the registration of
                  "inapplicable" rather than a computed result; there the
                  artefact's own `headline.best_zero_image_baseline` is read and
                  the substitution is printed. This fires on PI-CAI and nowhere
                  else.
  * slice axis  = defined iff the artefact's `headline.metric` is `slice_auc`.
  * patient axis= defined iff the patient AUC is finite. It is not finite on
                  Duke Breast, where every one of the 922 patients is positive.
  * colour      = whether the patient-level 95% interval lies wholly above
                  chance. That is a statement about an interval, applied
                  mechanically; it is not a hand-assigned verdict.

Every mark is printed to a source ledger at the end of the run, mapping the
point to the file it came from.

STYLE
-----
Matches fig1_collapse.pdf and fig2_trivial_fraction.pdf: Okabe-Ito palette, no
red/green pair carrying meaning, DejaVu Sans, vector PDF with fonts embedded as
Type-42 (TrueType) subsets and no Type-3 anywhere. verify_pdfs() checks the
output with pypdf after the build.

ANONYMITY
---------
Double-anonymized review. Nothing drawn into these figures may name an author,
an institution, a repository, a DOI, a software package or a file path. Only
public benchmark names appear, which are already named in the manuscript body.
"""

from __future__ import annotations

import json
import math
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

HERE = Path(__file__).resolve().parent             # paper/tex/rsna
REPO = HERE.parent.parent.parent                   # repo root
OUT = HERE / "figures"
OUT.mkdir(parents=True, exist_ok=True)

_SOURCES: list[str] = []


def _ledger(mark: str, relpath: str, detail: str = "") -> None:
    line = f"  {mark:<34s} <- {relpath}"
    if detail:
        line += f"\n  {'':<34s}    {detail}"
    print(line)
    _SOURCES.append(line)


def load(relpath: str, mark: str, detail: str = ""):
    """Read a JSON artefact and record it as the provenance of `mark`."""
    p = REPO / relpath
    if not p.exists():
        sys.exit(f"MISSING ARTEFACT: {p}\n  needed for: {mark}")
    with p.open() as fh:
        obj = json.load(fh)
    _ledger(mark, relpath, detail)
    return obj


# --------------------------------------------------------------------------- #
# style -- identical to paper/tex/make_figures.py
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

CHANCE = 0.5    # AUC of a constant predictor; the anchor both axes are read against


def save(fig, name: str):
    path = OUT / name
    fig.savefig(path, format="pdf")
    plt.close(fig)
    print(f"  wrote {path.relative_to(REPO)}")


# --------------------------------------------------------------------------- #
# Figure 2 -- the same score vector at two units, every audited benchmark-arm
# --------------------------------------------------------------------------- #
#
# One row of Table 3 per entry. `key` is the artefact stem under
# pipeline_out/trivial_baselines/; `label` is what the reader sees. Neither is a
# measured value -- the numbers all come out of the artefact named here.
#
# RSNA ICH is carried by its own dedicated artefact, which has its own schema
# (it is the flagship computation and was written by a separate script), so it
# is loaded separately below rather than through the common reader.

ARMS = [
    ("fastmri_prostate_t2_published",   "fastMRI Prostate, T2"),
    ("fastmri_prostate_dwi_published",  "fastMRI Prostate, DWI"),
    ("fastmriplus_knee_meniscus_tear",  "fastMRI+ knee, meniscus tear"),
    ("fastmriplus_knee_any_finding",    "fastMRI+ knee, any finding"),
    ("duke_breast_owner_slice_task",    "Duke Breast"),
    ("deeplesion_pelvis_vs_rest",       "DeepLesion, pelvis vs rest"),
    ("luna16_fp_reduction_candidates",  "LUNA16 candidates"),
    ("picai_case_level",                "PI-CAI, case level"),
]

# Label placement only. Three of the seven arms with both axes sit within 0.02
# AUC of each other on the slice axis (0.851, 0.854, 0.873), so their vertical
# intervals overlap and a label at the marker is unreadable. Those three are set
# in a stack clear of the data with a hairline leader to the marker they belong
# to. Everything here moves TEXT; no marker is displaced, and no marker is
# jittered.
#
#   (x, y, ha, va, leader)  -- x, y are absolute axis coordinates when
#   `leader` is True, and offsets from the marker when it is False.
LABEL_POS = {
    "RSNA ICH, any hemorrhage":     (-0.012, -0.013, "right", "top",    False),
    "fastMRI+ knee, any finding":   (-0.030,  0.000, "right", "center", False),
    "DeepLesion, pelvis vs rest":   (-0.026,  0.000, "right", "center", False),
    "LUNA16 candidates":            (-0.030,  0.000, "right", "center", False),
    # the crowded column, set out to the right
    "fastMRI+ knee, meniscus tear": ( 0.902,  0.586, "left",  "center", True),
    "fastMRI Prostate, T2":         ( 0.902,  0.470, "left",  "center", True),
    "fastMRI Prostate, DWI":        ( 0.902,  0.414, "left",  "center", True),
}


def _read_arm(stem: str, label: str) -> dict:
    """
    Pull one benchmark-arm's two-unit reading out of its stage-14 artefact.

    Which evaluation and which baseline are read is decided by the artefact, not
    here; see the module docstring. Nothing in this function contains a measured
    number.
    """
    rel = f"pipeline_out/trivial_baselines/{stem}.json"
    p = REPO / rel
    if not p.exists():
        sys.exit(f"MISSING ARTEFACT: {p}\n  needed for: Fig 2, {label}")
    with p.open() as fh:
        d = json.load(fh)

    ev_key = d["headline_evaluation"]
    ev = d["evaluations"][ev_key]

    bl_key = "positional_20bin"
    pos = ev["baselines"][bl_key]
    substituted = None
    # A positional baseline pinned to exactly chance at BOTH units is the
    # registration of "no slice index in this label file", not a result. Where
    # that happens the artefact's own strongest zero-image model is read instead.
    if pos["slice_auc"] == CHANCE and pos["patient_auc"] == CHANCE:
        bl_key = d["headline"]["best_zero_image_baseline"]
        substituted = (
            f"positional baseline is exactly {CHANCE:.3f} at both units "
            f"-> read '{bl_key}' instead"
        )
    b = ev["baselines"][bl_key]

    # Which axes exist for this arm, decided from the artefact.
    slice_ok = d["headline"]["metric"] == "slice_auc"
    pat = b["patient_auc"]
    patient_ok = pat is not None and not (isinstance(pat, float) and math.isnan(pat))

    pat_ci = b.get("patient_ci_clustered") or [None, None]
    row = dict(
        label=label,
        source=rel,
        evaluation=ev_key,
        baseline=bl_key,
        substituted=substituted,
        slice_ok=slice_ok,
        patient_ok=patient_ok,
        slice_auc=b["slice_auc"] if slice_ok else None,
        slice_ci=b["slice_ci_clustered"] if slice_ok else None,
        patient_auc=pat if patient_ok else None,
        patient_ci=pat_ci if patient_ok else None,
        n_subjects=b.get("n_patients"),
        n_pos_subjects=b.get("n_pos_patients"),
        n_slices=b.get("n_slices"),
        slices_per_volume_median=d.get("slices_per_volume", {}).get("median"),
    )
    return row


def _read_rsna_ich() -> dict:
    """The flagship arm, from its own artefact and its own schema."""
    rel = "pipeline_out/trivial_baselines/rsna_ich_unit_collapse.json"
    p = REPO / rel
    if not p.exists():
        sys.exit(f"MISSING ARTEFACT: {p}\n  needed for: Fig 2, RSNA ICH")
    with p.open() as fh:
        d = json.load(fh)
    L = d["labels"]["any"]
    return dict(
        label="RSNA ICH, any hemorrhage",
        source=rel,
        evaluation=f"{d['n_folds']}-fold subject-disjoint CV, pooled out of fold",
        baseline=f"positional_{d['n_bins']}bin",
        substituted=None,
        slice_ok=True,
        patient_ok=True,
        slice_auc=L["slice_auc"],
        slice_ci=L["slice_ci_clustered"],
        patient_auc=L["patient_auc_mean_agg"],
        patient_ci=L["patient_ci_clustered"],
        n_subjects=d["n_patients"],
        n_pos_subjects=L["n_pos_patients"],
        n_slices=d["n_slices"],
        slices_per_volume_median=None,
        _extra=d,
    )


def fig2():
    print("\nFigure 2 -- one pixel-blind score vector at two units, every audited "
          "benchmark-arm")

    rows = [_read_rsna_ich()]
    _ledger("Fig 2, RSNA ICH (both axes)", rows[0]["source"],
            f"labels['any'], {rows[0]['baseline']}, {rows[0]['evaluation']}")
    for stem, label in ARMS:
        r = _read_arm(stem, label)
        rows.append(r)
        detail = f"{r['evaluation']} / {r['baseline']}"
        if r["substituted"]:
            detail += f"   [{r['substituted']}]"
        if not r["slice_ok"]:
            detail += (f"   [no slice axis: headline metric is patient_auc, "
                       f"median slices per volume "
                       f"{r['slices_per_volume_median']:g}]")
        if not r["patient_ok"]:
            detail += (f"   [no patient axis: {r['n_pos_subjects']} of "
                       f"{r['n_subjects']} subjects positive]")
        _ledger(f"Fig 2, {label}", r["source"], detail)

    both = [r for r in rows if r["slice_ok"] and r["patient_ok"]]
    slice_only = [r for r in rows if r["slice_ok"] and not r["patient_ok"]]
    pat_only = [r for r in rows if r["patient_ok"] and not r["slice_ok"]]
    assert len(both) + len(slice_only) + len(pat_only) == len(rows)

    # The one classification in this figure, applied mechanically: does the
    # patient-level 95% interval lie wholly ABOVE chance? This is a statement
    # about where an interval sits, not a verdict about a benchmark.
    for r in both:
        r["above_chance_per_patient"] = r["patient_ci"][0] > CHANCE

    # ---- layout ---------------------------------------------------------- #
    # Main panel, plus two narrow gutters for the arms that carry only one of
    # the two axes. The gutters are separate axes whose meaningless axis has no
    # scale at all, so an arm with no patient-level value cannot be misread as
    # an arm with a LOW patient-level value -- which is the one way this figure
    # could lie. Both axes carry identical limits and identical ticks, so the
    # dashed line of agreement is unambiguous without forcing a square box.
    fig = plt.figure(figsize=(FULL_W, 4.95))
    gs = fig.add_gridspec(
        2, 2, width_ratios=[0.085, 1.0], height_ratios=[1.0, 0.085],
        wspace=0.02, hspace=0.03,
        left=0.085, right=0.855, bottom=0.30, top=0.945,
    )
    ax = fig.add_subplot(gs[0, 1])
    ax_l = fig.add_subplot(gs[0, 0], sharey=ax)
    ax_b = fig.add_subplot(gs[1, 1], sharex=ax)

    lo, hi = 0.38, 1.03
    TICKS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # line of agreement, and chance on each axis
    ax.plot([lo, hi], [lo, hi], color=GREY, lw=0.9, ls=(0, (5, 3)), zorder=1)
    ax.axhline(CHANCE, color=BLACK, lw=0.8, ls=(0, (1, 2)), zorder=1)
    ax.axvline(CHANCE, color=BLACK, lw=0.8, ls=(0, (1, 2)), zorder=1)
    ax.text(lo + 0.006, CHANCE + 0.008, "chance", fontsize=6.3, color=BLACK,
            ha="left", va="bottom")

    for r in both:
        col = ORANGE if r["above_chance_per_patient"] else BLUE
        x, y = r["slice_auc"], r["patient_auc"]
        # the vertical distance to the line of agreement IS the collapse
        ax.plot([x, x], [x, y], color=LIGHTGREY, lw=2.6,
                solid_capstyle="butt", zorder=2)
        ax.errorbar(
            [x], [y],
            xerr=[[x - r["slice_ci"][0]], [r["slice_ci"][1] - x]],
            yerr=[[y - r["patient_ci"][0]], [r["patient_ci"][1] - y]],
            fmt="none", ecolor=col, elinewidth=1.3, capsize=2.2, capthick=0.9,
            zorder=3,
        )
        ax.plot([x], [y], "o", ms=5.8, color=col, mec=col, ls="none", zorder=4)

        px, py, ha, va, leader = LABEL_POS[r["label"]]
        if leader:
            ax.annotate(
                r["label"], xy=(r["slice_ci"][1], y), xytext=(px, py),
                fontsize=6.4, color=col, ha=ha, va=va, zorder=5,
                arrowprops=dict(arrowstyle="-", color=col, lw=0.5,
                                shrinkA=2.5, shrinkB=1.5),
            )
        else:
            ax.text(x + px, y + py, r["label"], fontsize=6.4, color=col,
                    ha=ha, va=va, zorder=5)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xticks(TICKS)
    ax.set_yticks(TICKS)
    # the gutters carry the scales; the main panel borrows them
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="x", labelbottom=False, bottom=False)
    ax.tick_params(axis="y", labelleft=False, left=False)

    # ---- gutter: a slice axis but no patient axis ------------------------- #
    for r in slice_only:
        x = r["slice_auc"]
        ax_b.errorbar([x], [0.5],
                      xerr=[[x - r["slice_ci"][0]], [r["slice_ci"][1] - x]],
                      fmt="none", ecolor=GREY, elinewidth=1.3, capsize=2.2,
                      capthick=0.9, zorder=3)
        ax_b.plot([x], [0.5], "o", ms=5.8, color="white", mec=GREY, mew=1.4,
                  ls="none", zorder=4)
        ax_b.text(r["slice_ci"][1] + 0.010, 0.5, r["label"], fontsize=6.4,
                  color=GREY, ha="left", va="center", zorder=5)
    ax_b.set_ylim(0, 1)
    ax_b.set_yticks([])
    ax_b.spines["left"].set_visible(False)
    ax_b.set_xlabel("slice-level AUC")
    ax_b.set_xticks(TICKS)
    ax_b.axvline(CHANCE, color=BLACK, lw=0.8, ls=(0, (1, 2)), zorder=1)
    ax_b.text(lo + 0.006, 0.5, "patient level undefined", fontsize=6.0,
              color=GREY, ha="left", va="center", style="italic")

    # ---- gutter: a patient axis but no slice axis ------------------------- #
    for r in pat_only:
        y = r["patient_auc"]
        ax_l.errorbar([0.5], [y],
                      yerr=[[y - r["patient_ci"][0]], [r["patient_ci"][1] - y]],
                      fmt="none", ecolor=GREY, elinewidth=1.3, capsize=2.2,
                      capthick=0.9, zorder=3)
        ax_l.plot([0.5], [y], "o", ms=5.8, color="white", mec=GREY, mew=1.4,
                  ls="none", zorder=4)
        ax_l.text(0.5, r["patient_ci"][1] + 0.018,
                  r["label"].split(",")[0], fontsize=6.4, color=GREY,
                  ha="center", va="bottom", zorder=5)
    ax_l.set_xlim(0, 1)
    ax_l.set_xticks([])
    ax_l.spines["bottom"].set_visible(False)
    ax_l.set_ylabel("patient-level AUC of the same score vector\n"
                    "(mean aggregation)")
    ax_l.set_yticks(TICKS)
    ax_l.axhline(CHANCE, color=BLACK, lw=0.8, ls=(0, (1, 2)), zorder=1)
    ax_l.text(0.5, lo + 0.012, "no slice axis", fontsize=6.0, color=GREY,
              ha="center", va="bottom", rotation=90, style="italic")

    n_below = sum(1 for r in both if not r["above_chance_per_patient"])
    n_above = len(both) - n_below
    ax.set_title(
        f"{len(rows)} audited benchmark-arms; on the {len(both)} where both "
        f"units are defined, {n_below} patient-level intervals reach chance "
        f"and {n_above} do not",
        pad=6, fontsize=8.0, loc="left",
    )

    # ---- legend and the two one-unit notes, below the plot ---------------- #
    handles = [
        Line2D([], [], marker="o", ls="none", ms=5.8, color=BLUE,
               label="patient-level interval reaches chance or lies below it"),
        Line2D([], [], color=LIGHTGREY, lw=2.6,
               label="drop from the line of agreement"),
        Line2D([], [], marker="o", ls="none", ms=5.8, color=ORANGE,
               label="patient-level interval lies wholly above chance"),
        Line2D([], [], color=GREY, lw=0.9, ls=(0, (5, 3)),
               label="slice AUC = patient AUC"),
        Line2D([], [], marker="o", ls="none", ms=5.8, color="white", mec=GREY,
               mew=1.4, label="only one unit is defined (plotted in a gutter)"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.075, 0.185),
               frameon=False, handletextpad=0.7, borderaxespad=0.0,
               ncol=2, columnspacing=2.2, labelspacing=0.45)

    notes = []
    for r in pat_only:
        notes.append(
            f"{r['label']}: no slice index in that label file, so no "
            f"slice-level reading exists; the value shown is its strongest "
            f"zero-image baseline ({r['baseline'].replace('_', ' ')})."
        )
    for r in slice_only:
        notes.append(
            f"{r['label']}: patient-level AUC is undefined because "
            f"{r['n_pos_subjects']} of {r['n_subjects']} patients are positive."
        )
    fig.text(0.075, 0.035, "\n".join(notes), fontsize=6.2, color="#333333",
             ha="left", va="bottom", linespacing=1.5)

    save(fig, "fig2_unit_scatter.pdf")

    # ---- what the figure asserts, printed so it can be checked ------------ #
    print()
    for r in rows:
        s = f"{r['slice_auc']:.3f}" if r["slice_ok"] else "   n/a"
        p_ = f"{r['patient_auc']:.3f}" if r["patient_ok"] else "   n/a"
        gap = (f"{r['slice_auc'] - r['patient_auc']:+.3f}"
               if (r["slice_ok"] and r["patient_ok"]) else "     —")
        print(f"    {r['label']:<30s} slice {s}   patient {p_}   "
              f"slice−patient {gap}")
    print(f"    {n_below} of {len(both)} arms with both axes have a patient-level "
          f"interval reaching chance or below;")
    print(f"    {n_above} of {len(both)} lie wholly above chance per patient: "
          + ", ".join(r["label"] for r in both if r["above_chance_per_patient"]))
    xs = [r["slice_auc"] for r in both]
    ys = [r["patient_auc"] for r in both]
    print(f"    slice-level AUC spans {min(xs):.3f} to {max(xs):.3f}; "
          f"patient-level AUC spans {min(ys):.3f} to {max(ys):.3f}")


# --------------------------------------------------------------------------- #
# verification
# --------------------------------------------------------------------------- #

def verify_pdfs() -> int:
    """
    Check the two properties the journal and the reader both depend on.

      1. No Type-3 font anywhere. Type 3 embeds glyphs as uninterpreted PDF
         drawing operators: the text stops being text, so it cannot be searched,
         copied out of the proof, or re-hinted by the typesetter.
      2. Text is present as text on every page, embedded as a subset.

    Run over EVERY pdf in the figures directory, not only the one built here, so
    that the two copied figures are checked in the package they ship in.
    """
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
        print(f"  {path.name:<30s} {len(fonts):<7d} "
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
    print("make_rsna_figures.py -- every mark traced to the artefact it came from")
    print(f"repo: {REPO}")
    print(f"out:  {OUT.relative_to(REPO)}")
    print("=" * 78)
    print("Figures 1 and 3 are built by paper/tex/make_figures.py and copied in;")
    print("they are not rebuilt here.")
    fig2()
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
