#!/usr/bin/env python3
"""
Emit every number the RSNA submission prints, formatted, straight out of
revised_numbers.json.

    venv/bin/python paper/tex/rsna/make_rsna_numbers.py            # number sheet
    venv/bin/python paper/tex/rsna/make_rsna_numbers.py --tables   # table bodies

NOTHING IS TYPED BY HAND. Every value below is read from
paper/tex/rsna/revised_numbers.json, which carries its own source ledger with a
sha256 for each input file. The formatting rules are the manuscript's:

  * three decimals everywhere except the depth-conditioned block, which prints
    four, because the claim there is about an interval that excludes 0.500 by
    0.0013 and three decimals would hide it;
  * a null is never rounded toward significance: 0.500 stays 0.500 and
    -0.002 stays -0.002;
  * an exactly-degenerate value prints as "0.500 exactly" rather than as an
    estimate with an interval.

The companion checker is check_rsna_numbers.py, which reads main.tex back and
fails if it contains a number this file cannot produce.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NUM = HERE / "revised_numbers.json"

D = json.loads(NUM.read_text())
R = D["rsna_ich"]
OA = D["other_arms"]
TF = D["trivial_fraction_locked"]

LABELS = ["any", "epidural", "intraparenchymal", "intraventricular",
          "subarachnoid", "subdural"]
PRETTY = {
    "any": "Any hemorrhage",
    "epidural": "Epidural",
    "intraparenchymal": "Intraparenchymal",
    "intraventricular": "Intraventricular",
    "subarachnoid": "Subarachnoid",
    "subdural": "Subdural",
}


# --------------------------------------------------------------------------- #
# formatting
# --------------------------------------------------------------------------- #

def f3(x) -> str:
    if x is None:
        return "---"
    return f"{x:.3f}".replace("-", "$-$")


def f4(x) -> str:
    if x is None:
        return "---"
    return f"{x:.4f}".replace("-", "$-$")


def ci3(c) -> str:
    if not c or c[0] is None:
        return "---"
    return f"({f3(c[0])}, {f3(c[1])})"


def ci4(c) -> str:
    if not c or c[0] is None:
        return "---"
    return f"({f4(c[0])}, {f4(c[1])})"


def vci3(v, c) -> str:
    return f"{f3(v)} {ci3(c)}"


def vci4(v, c) -> str:
    return f"{f4(v)} {ci4(c)}"


def num(x) -> str:
    return f"\\num{{{int(x)}}}"


# --------------------------------------------------------------------------- #
# the number sheet
# --------------------------------------------------------------------------- #

def sheet() -> None:
    a = R["labels"]["any"]
    sp = R["across_holdout_spread"]["any"]
    ag = R["aggregation_sensitivity"]["labels"]["any"]
    sec = R["baseline_lock"]["secondary"]["any"]
    nul = a["within_series_permutation_null"]

    P = print
    P("=" * 78)
    P("COHORT AND DESIGN")
    P("=" * 78)
    P(f"  slices {R['cohort']['n_slices']}  series {R['cohort']['n_series']}  "
      f"patients {R['cohort']['n_patients']}")
    P(f"  estimator            {R['estimator']}")
    P(f"  holdout fraction     {R['holdout_fraction']}   seed {R['primary_seed']}")
    P(f"  train patients       {a['n_train_patients']}   slices {a['n_train_slices']}")
    P(f"  held-out patients    {a['n_test_patients']}   slices {a['n_test_slices']}")
    P(f"  bootstrap replicates {R['n_boot']}   bins {R['n_bins']}")
    P(f"  primary baseline     {R['primary_baseline']}")
    P(f"  depth definition     {R['depth_definition']}")

    P()
    P("=" * 78)
    P("R1  TWO UNITS, SIX LABELS, FROZEN HOLDOUT")
    P("=" * 78)
    for k in LABELS:
        L = R["labels"][k]
        P(f"  {k:<17s} slice {vci3(L['slice_auc'], L['slice_ci'])}   "
          f"patient {vci3(L['patient_auc_mean'], L['patient_auc_mean_ci'])}   "
          f"gap {f3(L['gap_slice_minus_patient'])}   "
          f"depthfixed {vci4(L['patient_auc_mean_depthfixed'], L['patient_auc_mean_depthfixed_ci'])}")
    sl = [R["labels"][k]["slice_auc"] for k in LABELS]
    pa = [R["labels"][k]["patient_auc_mean"] for k in LABELS]
    gp = [R["labels"][k]["gap_slice_minus_patient"] for k in LABELS]
    df = [R["labels"][k]["patient_auc_mean_depthfixed"] for k in LABELS]
    P(f"  slice range        {f3(min(sl))} to {f3(max(sl))}")
    P(f"  patient range      {f3(min(pa))} to {f3(max(pa))}")
    P(f"  gap range          {f3(min(gp))} to {f3(max(gp))}")
    P(f"  depth-fixed range  {f4(min(df))} to {f4(max(df))}")
    P("  patient CI wholly below 0.5: "
      + ", ".join(k for k in LABELS if R["labels"][k]["patient_auc_mean_ci"][1] < 0.5))
    P("  patient CI wholly above 0.5: "
      + (", ".join(k for k in LABELS if R["labels"][k]["patient_auc_mean_ci"][0] > 0.5) or "none"))
    P("  patient CI covering 0.5:     "
      + ", ".join(k for k in LABELS
                  if R["labels"][k]["patient_auc_mean_ci"][0] <= 0.5 <= R["labels"][k]["patient_auc_mean_ci"][1]))
    P("  depth-fixed CI excluding 0.5: "
      + ", ".join(k for k in LABELS
                  if not (R["labels"][k]["patient_auc_mean_depthfixed_ci"][0] <= 0.5
                          <= R["labels"][k]["patient_auc_mean_depthfixed_ci"][1])))
    P(f"  constant predictor slice/patient: "
      f"{f3(a['constant_predictor_slice_auc'])}/{f3(a['constant_predictor_patient_auc'])} "
      f"on all six labels: "
      f"{set(R['labels'][k]['constant_predictor_slice_auc'] for k in LABELS)}, "
      f"{set(R['labels'][k]['constant_predictor_patient_auc'] for k in LABELS)}")
    P(f"  perm null slice   {f3(nul['slice_mean'])} range {ci4(nul['slice_range'])}")
    P(f"  perm null patient {f3(nul['patient_mean'])} range {ci3(nul['patient_range'])}")
    P(f"  perm null patient, depth fixed {f4(nul['patient_depthfixed_mean'])} "
      f"range {ci4(nul['patient_depthfixed_range'])}")
    P(f"  slice excess over null {f3(nul['slice_excess_over_null'])}")
    P(f"  observed patient below its own null by "
      f"{f3(nul['patient_mean'] - a['patient_auc_mean'])}")
    P(f"  n_perm {nul['n_perm']}")

    P()
    P("=" * 78)
    P("R2  THE MECHANISM: STACK DEPTH")
    P("=" * 78)
    P(f"  depth alone, patient      {vci3(a['depth_alone_patient_auc'], a['depth_alone_patient_ci'])}")
    P(f"  mean agg, unstratified    {vci3(a['patient_auc_mean'], a['patient_auc_mean_ci'])}  "
      f"pairs {int(a['pairs_unstratified'])}")
    P(f"  within exact depth        {vci4(a['patient_auc_mean_depthfixed'], a['patient_auc_mean_depthfixed_ci'])}  "
      f"pairs {int(a['pairs_depthfixed'])}  strata {a['n_depth_strata_informative']}")
    P(f"  within 5-slice strata     {vci4(a['patient_auc_mean_depth5'], a['patient_auc_mean_depth5_ci'])}  "
      f"pairs {int(a['pairs_depth5'])}")
    P(f"  its own null at fixed depth {f4(nul['patient_depthfixed_mean'])} "
      f"{ci4(nul['patient_depthfixed_range'])}  -- observed sits INSIDE: "
      f"{nul['patient_depthfixed_range'][0] <= a['patient_auc_mean_depthfixed'] <= nul['patient_depthfixed_range'][1]}")

    P()
    P("=" * 78)
    P("R3  ACROSS-HOLDOUT SPREAD, 24 DRAWS")
    P("=" * 78)
    P(f"  n_draws {sp['n_draws']}  seeds {sp['seeds'][0]}..{sp['seeds'][-1]}")
    for key, fmt in [("slice_auc", f4), ("patient_auc_mean", f4),
                     ("patient_auc_mean_depthfixed", f4),
                     ("depth_alone_patient_auc", f4),
                     ("constant_slice_auc", f3), ("constant_patient_auc", f3)]:
        s = sp[key]
        P(f"  {key:<28s} mean {fmt(s['mean'])}  sd {s['sd']:.4f}  "
          f"[{fmt(s['min'])}, {fmt(s['max'])}]")
    P("  bootstrap width vs spread, patient: bootstrap "
      f"{a['patient_auc_mean_ci'][1] - a['patient_auc_mean_ci'][0]:.4f}, "
      f"spread {sp['patient_auc_mean']['max'] - sp['patient_auc_mean']['min']:.4f}")
    P("  depth-fixed family range covers 0.500: "
      f"{sp['patient_auc_mean_depthfixed']['min'] <= 0.5 <= sp['patient_auc_mean_depthfixed']['max']}")
    P("  primary draw ranks (of 24, ascending):")
    for k in LABELS:
        pr = R["primary_draw_position_in_family"][k]
        P(f"    {k:<17s} slice {pr['slice_auc']['rank_of_24_ascending']:>2d}  "
          f"patient {pr['patient_auc_mean']['rank_of_24_ascending']:>2d}  "
          f"depthfixed {pr['patient_auc_mean_depthfixed']['rank_of_24_ascending']:>2d}  "
          f"depth-alone {pr['depth_alone_patient_auc']['rank_of_24_ascending']:>2d}")

    P()
    P("=" * 78)
    P("R4  AGGREGATION SENSITIVITY, ANY")
    P("=" * 78)
    for op in ["mean", "max", "topk1_mean", "topk3_mean", "topk5_mean", "p75", "p90"]:
        P(f"  {op:<12s} {f4(ag['auc'][op])} {ci4(ag['ci'][op])}   "
          f"distinct patient scores {ag['n_distinct_patient_scores'][op]}   "
          f"CI excludes 0.5: {not (ag['ci'][op][0] <= 0.5 <= ag['ci'][op][1])}")

    P()
    P("=" * 78)
    P("R5  SECONDARY ZERO-IMAGE BASELINES, ANY")
    P("=" * 78)
    for k, v in sec.items():
        if k.startswith("_"):
            continue
        P(f"  {k:<28s} slice {f3(v['slice_auc'])}  patient {f3(v['patient_auc_mean'])}  "
          f"distinct slice scores {v['n_distinct_slice_scores']}")
    P(f"  locked primary is strongest of five on all six labels: "
      f"{R['baseline_lock']['note'][:110]}")

    P()
    P("=" * 78)
    P("R6  EVERY OTHER ARM")
    P("=" * 78)
    for key, v in OA.items():
        if key == "deeplesion_8class_lesion_type":
            P(f"  {v['display_name']}: 8-class accuracy {f3(v['baseline'])} "
              f"{ci3(v['baseline_ci'])} against chance {v['chance']}; "
              f"estimator {v['estimator'][:60]}")
            continue
        pooled = v.get("estimator_is_pooled_out_of_fold")
        s = v.get("slice_auc")
        p = v.get("patient_auc_mean")
        p = None if (isinstance(p, float) and p != p) else p
        cs = v.get("constant_predictor_slice_auc")
        P(f"  {v['display_name']:<38s} slice {vci3(s, v.get('slice_ci'))}  "
          f"patient {vci3(p, v.get('patient_auc_mean_ci') if p is not None else None)}  "
          f"const {f3(cs)}  pooled={pooled}  reps={v.get('n_boot')}")
    dl = [k for k in OA if k.startswith("deeplesion_") and k.endswith("_vs_rest")]
    P(f"  DeepLesion body-part arms: {len(dl)}; slice "
      f"{f3(min(OA[k]['slice_auc'] for k in dl))} to {f3(max(OA[k]['slice_auc'] for k in dl))}; "
      f"patient {f3(min(OA[k]['patient_auc_mean'] for k in dl))} to "
      f"{f3(max(OA[k]['patient_auc_mean'] for k in dl))}")
    pic = OA["picai_case_level"]
    P(f"  PI-CAI locked positional {f3(pic['positional_slice_auc'])}/"
      f"{f3(pic['positional_patient_auc'])}; secondary metadata tree "
      f"{vci3(pic['secondary_metadata_tree_patient_auc'], pic['secondary_metadata_tree_patient_ci'])}")
    P("  still pooled out of fold: "
      + ", ".join(f"{OA[k]['display_name']} {f4(OA[k]['constant_predictor_slice_auc'])}"
                  for k in OA if OA[k].get("estimator_is_pooled_out_of_fold")))
    t2, dw = OA["fastmri_prostate_t2"], OA["fastmri_prostate_dwi"]
    for v in (t2, dw):
        P(f"  {v['display_name']}: depth-fixed {f3(v['patient_auc_mean_depthfixed'])} "
          f"{ci3(v['patient_auc_mean_depthfixed_ci'])} on {int(v['pairs_depthfixed'])} pairs; "
          f"n_test_subjects {v['n_test_subjects']}, slices {v['n_test_slices']}")

    P()
    P("=" * 78)
    P("R7  DESCRIPTIVE CROSS-STUDY COMPARISON (locked baseline)")
    P("=" * 78)
    P(f"  status: {TF['status']}")
    prim = primary_tf_rows()
    for r in prim:
        P(f"  {r['benchmark']:<18s} {r['arm']:<14s} {r['metric']:<26s} "
          f"chance {r['chance']:<7g} published {r['published']:<7g} "
          f"locked TF {f4(r['locked_trivial_fraction'])} "
          f"{ci4(r.get('locked_trivial_fraction_ci'))}")
    P("  RSNA ICH per label (strongest comparator per label):")
    for k in LABELS:
        rows = [r for r in TF["rows"]
                if r["benchmark"] == "RSNA ICH" and r["arm"] == k]
        best = max(rows, key=lambda r: r["published"])
        P(f"    {k:<17s} published {best['published']}  "
          f"TF {f3(best['locked_trivial_fraction'])} "
          f"{ci3(best.get('locked_trivial_fraction_ci'))}")
    pre = [r for r in TF["rows"] if "preprint" in r["peer_reviewed"]]
    P(f"  preprint worked example, strongest comparator per arm:")
    for arm in ["T2", "DWI"]:
        rows = [r for r in pre if r["arm"] == arm]
        best = max(rows, key=lambda r: r["published"])
        P(f"    {arm:<4s} published {best['published']}  "
          f"TF {f3(best['locked_trivial_fraction'])} "
          f"{ci3(best.get('locked_trivial_fraction_ci'))}")
    P(f"  peer-reviewed rows matched (TF >= 1): "
      f"{sum(1 for r in TF['rows'] if 'peer-reviewed' in r['peer_reviewed'] and r['locked_trivial_fraction'] >= 1)}")


def primary_tf_rows() -> list[dict]:
    """One row per benchmark: the strongest peer-reviewed comparator on it."""
    out = []
    for bench, arm in [("RSNA ICH", "any"), ("DeepLesion", "8-class lesion type"),
                       ("PI-CAI", "case level"), ("LUNA16", "FP-reduction track")]:
        rows = [r for r in TF["rows"]
                if r["benchmark"] == bench and r["arm"] == arm
                and "peer-reviewed" in r["peer_reviewed"]]
        out.append(max(rows, key=lambda r: r["published"]))
    return out


# --------------------------------------------------------------------------- #
# table bodies
# --------------------------------------------------------------------------- #

def table1() -> None:
    print("% ---- Table 1 upper block")
    for k in LABELS:
        L = R["labels"][k]
        print(f"{PRETTY[k]:<17s}& {L['test_slice_prevalence']:.3f} & "
              f"{L['test_patient_prevalence']:.3f} & "
              f"{vci3(L['slice_auc'], L['slice_ci'])} & "
              f"{vci3(L['patient_auc_mean'], L['patient_auc_mean_ci'])} & "
              f"{f3(L['gap_slice_minus_patient'])} & "
              f"{vci4(L['patient_auc_mean_depthfixed'], L['patient_auc_mean_depthfixed_ci'])}\\\\")

    a = R["labels"]["any"]
    print("% ---- Table 1 lower block")
    print(f"Mean aggregation, unstratified & "
          f"{vci4(a['patient_auc_mean'], a['patient_auc_mean_ci'])} & "
          f"{num(a['pairs_unstratified'])}\\\\")
    print(f"\\quad within exact stack depth ({a['n_depth_strata_informative']} strata) & "
          f"{vci4(a['patient_auc_mean_depthfixed'], a['patient_auc_mean_depthfixed_ci'])} & "
          f"{num(a['pairs_depthfixed'])}\\\\")
    print(f"\\quad within 5-slice depth strata & "
          f"{vci4(a['patient_auc_mean_depth5'], a['patient_auc_mean_depth5_ci'])} & "
          f"{num(a['pairs_depth5'])}\\\\")
    print(f"Stack depth alone & "
          f"{vci4(a['depth_alone_patient_auc'], a['depth_alone_patient_ci'])} & "
          f"{num(a['pairs_unstratified'])}\\\\")
    nul = a["within_series_permutation_null"]
    print(f"Permutation null at fixed depth & "
          f"{f4(nul['patient_depthfixed_mean'])} "
          f"[{f4(nul['patient_depthfixed_range'][0])}, "
          f"{f4(nul['patient_depthfixed_range'][1])}] & ---\\\\")


def table2() -> None:
    sp = R["across_holdout_spread"]["any"]
    rep = R["replication_of_eight_holdout_result"]
    a = R["labels"]["any"]
    ag = R["aggregation_sensitivity"]["labels"]["any"]
    sec = R["baseline_lock"]["secondary"]["any"]

    print("% ---- Table 2 block A: 24 draws")
    rows = [("Slice AUC", "slice_auc", "slice_auc"),
            ("Patient AUC, mean aggregation", "patient_auc_mean", "patient_auc_mean"),
            ("Patient AUC, stack depth fixed", "patient_auc_mean_depthfixed",
             "patient_auc_mean_depthfixed"),
            ("Stack depth alone, patient", "depth_alone_patient_auc", None),
            ("Constant predictor, slice", "constant_slice_auc", "constant_slice_auc"),
            ("Constant predictor, patient", "constant_patient_auc", None)]
    eight = rep["reported_by_operator_8_draws"]
    for name, key, ekey in rows:
        s = sp[key]
        e = eight.get(ekey) if ekey else None
        e_s = f4(e) if isinstance(e, (int, float)) else (e if e else "---")
        print(f"{name} & {f4(s['mean'])} & {s['sd']:.4f} & "
              f"[{f4(s['min'])}, {f4(s['max'])}] & {e_s}\\\\")

    print("% ---- Table 2 block B: aggregation")
    names = {"mean": "Mean (pre-specified)", "max": "Maximum",
             "topk1_mean": "Top-1 mean", "topk3_mean": "Top-3 mean",
             "topk5_mean": "Top-5 mean", "p75": "75th percentile",
             "p90": "90th percentile"}
    for op in ["mean", "max", "topk1_mean", "topk3_mean", "topk5_mean", "p75", "p90"]:
        v, c = ag["auc"][op], ag["ci"][op]
        if c[0] == c[1]:
            cell = f"{f4(v)} (degenerate)"
        else:
            cell = vci4(v, c)
        print(f"{names[op]} & {cell} & {ag['n_distinct_patient_scores'][op]}\\\\")

    print("% ---- Table 2 block C: secondary baselines")
    nm = {"volume_size": "Volume size (stack depth)",
          "metadata_tree": "Metadata tree",
          "combined_position_metadata": "Position $+$ metadata tree",
          "column[plane]": "Imaging plane alone",
          "column[rescale_slope]": "Reconstruction slope alone",
          "column[rescale_intercept]": "Reconstruction intercept alone"}
    print(f"Positional, 20 bins (locked primary) & {f3(a['slice_auc'])} & "
          f"{f3(a['patient_auc_mean'])}\\\\")
    for k, label in nm.items():
        v = sec[k]
        print(f"{label} & {f3(v['slice_auc'])} & {f3(v['patient_auc_mean'])}\\\\")


ARM_ORDER = [
    ("rsna_ich_any", "RSNA ICH, any hemorrhage"),
    ("fastmri_prostate_t2", None),
    ("fastmri_prostate_dwi", None),
    ("deeplesion_pelvis_vs_rest", None),
    ("deeplesion_mediastinum_vs_rest", None),
    ("deeplesion_abdomen_vs_rest", None),
    ("deeplesion_kidney_vs_rest", None),
    ("deeplesion_liver_vs_rest", None),
    ("deeplesion_lung_vs_rest", None),
    ("deeplesion_softtissue_vs_rest", None),
    ("deeplesion_bone_vs_rest", None),
    ("fastmriplus_knee_meniscus_tear", None),
    ("fastmriplus_knee_any_finding", None),
    ("duke_breast_owner_slice_task", None),
    ("luna16_fp_reduction_candidates", None),
    ("picai_case_level", None),
]

# Estimator short names and the three-way input class of the field the locked
# baseline reads on that arm. Both are properties of the label file, recorded in
# revised_numbers.json (estimator) and in the taxonomy table (class).
EST = {"frozen": "Frozen holdout", "published": "Published split",
       "pooled": "Pooled out of fold", "repeat": "Repeated holdout"}


def table3() -> None:
    print("% ---- Table 3, 16 arms")
    for key, override in ARM_ORDER:
        if key == "rsna_ich_any":
            L = R["labels"]["any"]
            est, cls = EST["frozen"], "G"
            s, sci = L["slice_auc"], L["slice_ci"]
            p, pci = L["patient_auc_mean"], L["patient_auc_mean_ci"]
            cst = L["constant_predictor_slice_auc"]
            name = override
        else:
            v = OA[key]
            name = override or v["display_name"]
            pooled = v.get("estimator_is_pooled_out_of_fold")
            est = EST["pooled"] if pooled else EST["published"]
            s, sci = v.get("slice_auc"), v.get("slice_ci")
            p = v.get("patient_auc_mean")
            p = None if (isinstance(p, float) and p != p) else p
            pci = v.get("patient_auc_mean_ci") if p is not None else None
            cst = v.get("constant_predictor_slice_auc")
            cls = {"fastmri_prostate_t2": "G, A", "fastmri_prostate_dwi": "G, A",
                   "duke_breast_owner_slice_task": "G, A",
                   "picai_case_level": "G, A, C"}.get(key, "G")
            if key.startswith("deeplesion_"):
                cls = "G, C"
        pcell = vci3(p, pci) if p is not None else "Undefined"
        scell = vci3(s, sci) if s is not None else "Not applicable"
        print(f"{name} & {scell} & {pcell} & {est} & {f3(cst)} & {cls}\\\\")


def main() -> None:
    if "--tables" in sys.argv:
        table1()
        print()
        table2()
        print()
        table3()
    else:
        sheet()


if __name__ == "__main__":
    main()
