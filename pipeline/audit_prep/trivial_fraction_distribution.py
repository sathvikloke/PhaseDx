"""The trivial fraction as a CONTINUOUS measure, across every benchmark and every
published comparator on disk.

WHY THIS FILE EXISTS
    `paper/audit_results.md` used to lead with a count of verdicts -- so many MATCHED, so
    many PARTIAL, so many NOT MATCHED. That headline is weak in two independent ways.
    (1) It throws away the information in the PARTIAL rows: a benchmark at trivial fraction
    0.61 and a benchmark at 0.31 are both "PARTIAL", and the difference between them is the
    whole point of having a continuous statistic. (2) It makes the paper's strength hostage
    to one threshold and, through it, to one preprint -- because the only rows that cross
    the MATCHED threshold have a preprint comparator.

    So the headline becomes the DISTRIBUTION of

        trivial fraction = (best zero-image baseline - chance) / (published - chance)

    over every (benchmark, published comparator) pair the audit holds, with intervals, and
    the categorical verdict is kept as a secondary column rather than deleted. Rows where
    the baseline does NOT fire are part of the distribution, not an appendix: a measure
    that always returns a large number measures nothing.

WHAT IS AND IS NOT COMPUTED HERE
    This script does not re-fit any baseline. Every baseline value it uses is READ from an
    artefact produced elsewhere and named in `source` on every row, so the distribution can
    be regenerated but not silently invented. The published values are transcribed from the
    cited papers and carry their peer-review status.

    Rows are NOT independent observations. One benchmark can contribute several rows
    because a paper reports several systems in one table (Rempe et al.'s Table II has three
    arms; Burduja et al.'s Table 3 has two model columns x six labels). Distribution
    summaries are therefore reported twice: over all rows, and over benchmarks, taking each
    benchmark's STRONGEST published comparator, which is the conservative choice because a
    stronger comparator makes the denominator larger and the fraction smaller.

Usage:
    python pipeline/audit_prep/trivial_fraction_distribution.py \
        [--out-json paper/trivial_fraction_distribution.json] \
        [--out-md paper/trivial_fraction_distribution.md]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pipeline"))
from s14_trivialbaselines import trivial_fraction  # noqa: E402

CARDS = ROOT / "pipeline_out" / "trivial_baselines"
LOGS = ROOT / "pipeline_out" / "audit_logs"

# --------------------------------------------------------------------------
# The verdict rule, unchanged from paper/audit_results.md section 1. It is kept as a
# SECONDARY column. It is not the headline any more, and it is not deleted either.
# --------------------------------------------------------------------------

def verdict(tf: dict, baseline_ci, published: float) -> str:
    """The rule applied MECHANICALLY, with no case-by-case judgement.

    Applying it mechanically is deliberate, and it disagrees with the hand-assigned
    verdicts in an earlier revision of paper/audit_results.md on the two PI-CAI rows:
    their trivial fractions are 0.467 and 0.532, i.e. >= 0.30, so the written rule
    returns PARTIAL, while the document recorded NOT MATCHED on the strength of a cohort
    caveat that the rule has no slot for. Both readings are defensible and neither is
    hidden here. That a threshold-and-judgement verdict can flip while the underlying
    fraction does not move at all is itself an argument for demoting the verdict to a
    secondary column, which is what this file does.
    """
    if tf.get("value") is None:
        return "UNDEFINED"
    hi = (baseline_ci or [None, None])[1]
    if hi is not None and np.isfinite(hi) and hi >= published:
        return "MATCHED"
    v = tf["value"]
    if v >= 0.30:
        return "PARTIAL"
    return "NOT MATCHED"


# Extremes of the definition, exercised on the tool's own implementation so the limits
# section of the paper is a measurement rather than an assertion.
EXTREME_CASES = [
    ("published far above chance, baseline mid-range", 0.7374, 0.5, 0.9843),
    ("published just above chance (headroom 0.021)", 0.60, 0.5, 0.521),
    ("published exactly at chance", 0.60, 0.5, 0.50),
    ("published BELOW chance", 0.60, 0.5, 0.45),
    ("baseline ABOVE published", 0.854, 0.5, 0.714),
    ("baseline exactly at chance", 0.50, 0.5, 0.90),
    ("baseline BELOW chance", 0.4533, 0.5, 0.9843),
    ("baseline exactly equals published", 0.90, 0.5, 0.90),
    ("published near-perfect", 0.7374, 0.5, 1.00),
    ("non-AUROC chance anchor (average precision, prevalence 0.1434)",
     0.26, 0.1434, 0.60),
]


def check_extremes() -> list[dict]:
    out = []
    for nm, b, c, p in EXTREME_CASES:
        r = trivial_fraction(b, c, p, baseline_ci=(b - 0.02, b + 0.02))
        out.append({"case": nm, "baseline": b, "chance": c, "published": p,
                    "value": r["value"], "value_clipped": r["value_clipped"],
                    "ci": r["ci"], "reason": r["reason"]})
    return out


# --------------------------------------------------------------------------
# Baselines read from artefacts
# --------------------------------------------------------------------------

def card(name: str) -> dict:
    return json.loads((CARDS / f"{name}.json").read_text())


def card_best(name: str, metric: str = "slice_auc"):
    """(value, ci, baseline name, source) for the best zero-image baseline on a card."""
    d = card(name)
    ev = d["evaluations"][d["headline_evaluation"]]
    ci_key = "slice_ci_clustered" if metric.startswith("slice") else "patient_ci_clustered"
    best, val = None, -np.inf
    for bn, b in ev["baselines"].items():
        if bn == "prevalence":
            continue
        v = b.get(metric)
        if v is not None and np.isfinite(v) and v > val:
            best, val = bn, float(v)
    return (val, ev["baselines"][best][ci_key], best,
            f"pipeline_out/trivial_baselines/{name}.json")


def parse_burduja_log(path: Path) -> dict:
    """Six rows from the split-geometry replication. Format is fixed by the script that
    writes it; the published column is asserted against the transcription below so a
    silently changed log cannot walk into the table."""
    out = {}
    pat = re.compile(
        r"^(Any|Epidural \(EPH\)|Intraparenchymal \(IPH\)|Intraventricular \(IVH\)|"
        r"Subarachnoid \(SAH\)|Subdural \(SDH\))\s+"
        r"([\d.]+)\s+([\d.]+)\s+\[([\d.]+),([\d.]+)\]\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$")
    for line in path.read_text().splitlines():
        m = pat.match(line.strip())
        if not m:
            continue
        nm = m.group(1).split(" (")[0].lower()
        out[nm] = {"prevalence": float(m.group(2)), "slice_auc": float(m.group(3)),
                   "ci": [float(m.group(4)), float(m.group(5))],
                   "patient_disjoint_slice_auc": float(m.group(6)),
                   "published": float(m.group(7))}
    return out


# --------------------------------------------------------------------------
# Published comparators, transcribed with their peer-review status
# --------------------------------------------------------------------------

REMPE = ("Rempe et al. 2024, arXiv:2407.06165, Table II", "preprint (arXiv only, "
         "re-queried 2026-07-29: no journal-ref, no DOI, no Europe PMC record)")
YAN = ("Yan et al., CVPR 2018, Table 1", "peer-reviewed (IEEE/CVF conference proceedings)")
SETIO = ("Setio et al., Medical Image Analysis 2017;42:1-13", "peer-reviewed (journal)")
SAHA = ("Saha et al., Lancet Oncology 2024;25:879-887", "peer-reviewed (journal)")
BURDUJA = ("Burduja, Ionescu & Verga, Sensors 2020;20(19):5611, Table 3",
           "peer-reviewed (journal, PubMed-indexed)")

ICH_PUB = {  # label -> (BiLSTM, plain ResNeXt-101) slice ROC AUC, Burduja Table 3
    "any": (0.9843, 0.9752), "epidural": (0.9851, 0.9703),
    "intraparenchymal": (0.9927, 0.9883), "intraventricular": (0.9970, 0.9953),
    "subarachnoid": (0.9821, 0.9644), "subdural": (0.9682, 0.9576),
}


def build_rows() -> list[dict]:
    rows: list[dict] = []

    def add(**kw):
        kw.setdefault("estimator_variant_of", None)
        tf = trivial_fraction(kw["baseline"], kw["chance"], kw["published"],
                              baseline_ci=kw.get("baseline_ci"))
        kw["trivial_fraction"] = tf.get("value")
        kw["trivial_fraction_ci"] = tf.get("ci")
        kw["trivial_fraction_undefined_reason"] = tf.get("reason")
        kw["verdict"] = verdict(tf, kw.get("baseline_ci"), kw["published"])
        rows.append(kw)

    # ---- RSNA ICH, full cohort, 5-fold subject-disjoint CV -----------------
    coll = json.loads((CARDS / "rsna_ich_unit_collapse.json").read_text())
    for lab, r in coll["labels"].items():
        pub_lstm, pub_plain = ICH_PUB[lab]
        assert abs(r["published_slice_auc"] - pub_lstm) < 1e-9, lab
        for pub, which, primary in ((pub_lstm, "ResNeXt-101+BiLSTM", True),
                                    (pub_plain, "ResNeXt-101 (no LSTM)", False)):
            add(benchmark="RSNA ICH", arm=lab, metric="slice ROC AUC",
                unit="slice", chance=0.5,
                baseline=r["slice_auc"], baseline_ci=r["slice_ci_clustered"],
                baseline_model="positional 20-bin",
                estimator="5-fold subject-disjoint CV, pooled out of fold, "
                          "2,000 patient-clustered bootstrap reps, all 18,938 patients",
                published=pub, published_system=which, primary_comparator=primary,
                comparator=BURDUJA[0], peer_reviewed=BURDUJA[1],
                n_subjects=coll["n_patients"], n_rows=coll["n_slices"],
                source="pipeline_out/trivial_baselines/rsna_ich_unit_collapse.json")

    # ---- RSNA ICH, their split geometry (200 re-draws of 744 held-out scans) ----
    bur = parse_burduja_log(LOGS / "rsna_ich_burduja_conditions.log")
    for lab, r in bur.items():
        pub_lstm, _ = ICH_PUB[lab]
        assert abs(r["published"] - pub_lstm) < 1e-9, f"log/transcription mismatch {lab}"
        add(benchmark="RSNA ICH", arm=f"{lab} (their split geometry)",
            estimator_variant_of=f"RSNA ICH / {lab}",
            metric="slice ROC AUC", unit="slice", chance=0.5,
            baseline=r["slice_auc"], baseline_ci=r["ci"],
            baseline_model="positional 20-bin",
            estimator="200 random re-draws of their 744-scan held-out split; the "
                      "interval is split-to-split spread, NOT sampling error",
            published=pub_lstm, published_system="ResNeXt-101+BiLSTM",
            primary_comparator=False,
            comparator=BURDUJA[0], peer_reviewed=BURDUJA[1],
            n_subjects=18938, n_rows=752802,
            source="pipeline_out/audit_logs/rsna_ich_burduja_conditions.log")

    # ---- fastMRI Prostate, both arms, three published systems each --------
    for arm, cname in (("T2", "fastmri_prostate_t2_published"),
                       ("DWI", "fastmri_prostate_dwi_published")):
        val, ci, bname, src = card_best(cname)
        d = card(cname)
        for pub, which, primary in ((0.861, "image + k-space (gold standard)", True),
                                    (0.809, "PCA x2 magnitude + phase", False),
                                    (0.714, "R=16 PCA coil combination", False)):
            add(benchmark="fastMRI Prostate", arm=arm, metric="slice AUROC",
                unit="slice", chance=0.5, baseline=val, baseline_ci=ci,
                baseline_model=bname.replace("_", " "),
                estimator="the authors' own in-file split, patient-clustered bootstrap",
                published=pub, published_system=which, primary_comparator=primary,
                comparator=REMPE[0], peer_reviewed=REMPE[1],
                n_subjects=d["n_subjects"], n_rows=d["n_rows"], source=src)

    # ---- DeepLesion, Yan et al.'s reconstructed conditions ----------------
    # 200 random patient-disjoint 25/25/50 partitions of the 9,816 type-labelled rows.
    # Value transcribed from paper/audit_results.md section 3.2, produced by
    # pipeline/audit_prep/deeplesion_yan_conditions.py.
    for pub, which, primary in (
            (0.905, "triplet + type + location + size", True),
            (0.862, "multi-scale ImageNet feature", False),
            (0.597, "their own image-derived Location feature baseline", False)):
        add(benchmark="DeepLesion", arm="8-class lesion type",
            metric="8-class accuracy", unit="lesion", chance=0.2361,
            baseline=0.5571, baseline_ci=[0.5243, 0.5778],
            baseline_model="positional 20-bin on published normalised z",
            estimator="200 random patient-disjoint 25/25/50 partitions; the interval is "
                      "partition-to-partition spread",
            published=pub, published_system=which, primary_comparator=primary,
            comparator=YAN[0], peer_reviewed=YAN[1],
            n_subjects=None, n_rows=9816,
            source="paper/audit_results.md section 3.2 "
                   "(pipeline/audit_prep/deeplesion_yan_conditions.py)")

    # ---- PI-CAI, at the unit its authors report ---------------------------
    val, ci, bname, src = card_best("picai_case_level", metric="patient_auc")
    d = card("picai_case_level")
    for pub, which, primary in ((0.91, "AI system", True),
                                (0.86, "62 radiologists, PI-RADS 2.1", False)):
        add(benchmark="PI-CAI", arm="case level", metric="case AUROC",
            unit="case/patient", chance=0.5, baseline=val, baseline_ci=ci,
            baseline_model=bname.replace("_", " "),
            estimator="the benchmark's own official 5-fold splits, clustered bootstrap; "
                      "DIFFERENT COHORT from the published number (see audit_results 3.5)",
            published=pub, published_system=which, primary_comparator=primary,
            comparator=SAHA[0], peer_reviewed=SAHA[1],
            n_subjects=d["n_subjects"], n_rows=d["n_rows"], source=src)

    # ---- LUNA16, on the competition's own metric --------------------------
    # CPM 0.0020 and sensitivity 0.0006 at 1 FP/scan against a random-score reference of
    # 0.0027, from pipeline/audit_prep/luna16_cpm.py (audit_results section 3.6).
    add(benchmark="LUNA16", arm="FP-reduction track",
        metric="sensitivity at 1 FP/scan", unit="candidate", chance=0.0027,
        baseline=0.0006, baseline_ci=None,
        baseline_model="positional 20-bin, scored on the challenge metric",
        estimator="scan-disjoint 5-fold, out of fold; CPM 0.0020 against a random-score "
                  "reference CPM of 0.0027",
        published=0.95, published_system="combined challenge solutions",
        primary_comparator=True,
        comparator=SETIO[0], peer_reviewed=SETIO[1],
        n_subjects=888, n_rows=754975,
        source="paper/audit_results.md section 3.6 "
               "(pipeline/audit_prep/luna16_cpm.py)")
    return rows


# --------------------------------------------------------------------------

def summarise(rows: list[dict], label: str) -> dict:
    v = np.array([r["trivial_fraction"] for r in rows
                  if r["trivial_fraction"] is not None], float)
    if not len(v):
        return {"set": label, "n": 0}
    return {
        "set": label, "n": int(len(v)),
        "min": float(v.min()), "q25": float(np.percentile(v, 25)),
        "median": float(np.median(v)), "q75": float(np.percentile(v, 75)),
        "max": float(v.max()),
        "n_at_or_below_0.05": int((v <= 0.05).sum()),
        "n_in_0.30_to_0.70": int(((v >= 0.30) & (v <= 0.70)).sum()),
        "n_at_or_above_1": int((v >= 1.0).sum()),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("Usage:")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-json", default=str(ROOT / "paper" /
                                              "trivial_fraction_distribution.json"))
    ap.add_argument("--out-md", default=str(ROOT / "paper" /
                                            "trivial_fraction_distribution.md"))
    a = ap.parse_args(argv)

    allrows = build_rows()
    # Distribution summaries run on the primary estimator only. The split-geometry
    # replication measures a row already in the table a second way; counting it again
    # would inflate n without adding an observation. It is kept in the row table as a
    # robustness check and is excluded here.
    rows = [r for r in allrows if r["estimator_variant_of"] is None]
    primary = [r for r in rows if r["primary_comparator"]]
    peer = [r for r in rows if not r["peer_reviewed"].startswith("preprint")]
    peer_primary = [r for r in peer if r["primary_comparator"]]
    pre = [r for r in rows if r["peer_reviewed"].startswith("preprint")]

    payload = {
        "tool": "trivial_fraction_distribution",
        "version": "1.0",
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "definition": "(best zero-image baseline - chance) / (published - chance)",
        "row_independence_warning":
            "rows are NOT independent: one benchmark contributes several rows when a "
            "paper reports several systems in one table. Read the per-benchmark summary, "
            "which uses each benchmark's STRONGEST published comparator, as the primary "
            "distribution.",
        "summaries": [
            summarise(rows, "all rows"),
            summarise(primary, "strongest published comparator per benchmark-arm"),
            summarise(peer, "peer-reviewed comparators, all rows"),
            summarise(peer_primary,
                      "peer-reviewed comparators, strongest per benchmark-arm"),
            summarise(pre, "preprint comparator (Rempe et al.) only"),
        ],
        "verdict_counts": {v: sum(1 for r in rows if r["verdict"] == v)
                           for v in ("MATCHED", "PARTIAL", "NOT MATCHED", "UNDEFINED")},
        "verdict_counts_peer_reviewed_only":
            {v: sum(1 for r in peer if r["verdict"] == v)
             for v in ("MATCHED", "PARTIAL", "NOT MATCHED", "UNDEFINED")},
        "verdict_rule": "applied mechanically; see verdict() docstring for the two PI-CAI "
                        "rows where the mechanical rule and the earlier hand-assigned "
                        "verdict in paper/audit_results.md disagree",
        "extremes_check": check_extremes(),
        "rows": allrows,
    }
    Path(a.out_json).write_text(json.dumps(payload, indent=2))

    # ---- markdown ----
    def f(x, nd=3):
        return "—" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"

    def ci(p, nd=3):
        return "—" if not p else f"[{p[0]:.{nd}f}, {p[1]:.{nd}f}]"

    L = ["# Trivial fraction as a continuous measure — every benchmark, every comparator",
         "",
         f"Generated {payload['generated_utc']} by "
         "`pipeline/audit_prep/trivial_fraction_distribution.py`. "
         "No baseline is re-fitted here; every value is read from the artefact named in "
         "the last column.", "",
         "> trivial fraction = (best zero-image baseline − chance) / (published − chance)",
         "",
         "**Rows are not independent.** One benchmark contributes several rows when a "
         "paper reports several systems in one table. Read the *strongest published "
         "comparator per benchmark-arm* line as the primary distribution: a stronger "
         "comparator makes the denominator larger and the fraction smaller, so it is the "
         "conservative choice.", "",
         "## Distribution", "",
         "| set | n | min | Q1 | median | Q3 | max | ≤0.05 | 0.30–0.70 | ≥1 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for s in payload["summaries"]:
        if not s["n"]:
            continue
        L.append(f"| {s['set']} | {s['n']} | {f(s['min'])} | {f(s['q25'])} | "
                 f"**{f(s['median'])}** | {f(s['q75'])} | {f(s['max'])} | "
                 f"{s['n_at_or_below_0.05']} | {s['n_in_0.30_to_0.70']} | "
                 f"{s['n_at_or_above_1']} |")
    L += ["", "## Rows", "",
          "| benchmark | arm | published | system | peer-reviewed? | zero-image baseline "
          "| **trivial fraction** [CI] | verdict (secondary) | source |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(allrows, key=lambda r: (r["benchmark"], r["arm"], -r["published"])):
        pr = "**no — preprint**" if r["peer_reviewed"].startswith("preprint") else "yes"
        tfs = ("undefined" if r["trivial_fraction"] is None
               else f"**{f(r['trivial_fraction'])}** {ci(r['trivial_fraction_ci'])}")
        arm = r["arm"] + (" *(variant, not counted)*" if r["estimator_variant_of"] else "")
        L.append(f"| {r['benchmark']} | {arm} | {f(r['published'], 4)} "
                 f"| {r['published_system']} | {pr} "
                 f"| {f(r['baseline'], 4)} {ci(r['baseline_ci'])} | {tfs} "
                 f"| {r['verdict']} | `{r['source']}` |")
    L += ["", "## Behaviour at the extremes of the definition", "",
          "Run on the tool's own `trivial_fraction()` so the limits section is a "
          "measurement, not an assertion. Baseline CI is baseline ± 0.02 throughout.", "",
          "| case | baseline | chance | published | value | clipped | note |",
          "|---|---|---|---|---|---|---|"]
    for e in payload["extremes_check"]:
        val = "**undefined**" if e["value"] is None else f(e["value"], 4)
        clp = "—" if e["value_clipped"] is None else f(e["value_clipped"], 4)
        note = e["reason"] or ("clipped copy differs from value" if
                               e["value_clipped"] != e["value"] else "")
        L.append(f"| {e['case']} | {f(e['baseline'], 4)} | {f(e['chance'], 4)} "
                 f"| {f(e['published'], 4)} | {val} | {clp} | {note} |")
    L += ["", "## Verdict counts (secondary, kept not deleted)", "",
          "| set | MATCHED | PARTIAL | NOT MATCHED |", "|---|---|---|---|",
          f"| all rows | {payload['verdict_counts']['MATCHED']} | "
          f"{payload['verdict_counts']['PARTIAL']} | "
          f"{payload['verdict_counts']['NOT MATCHED']} |",
          f"| peer-reviewed comparators only | "
          f"{payload['verdict_counts_peer_reviewed_only']['MATCHED']} | "
          f"{payload['verdict_counts_peer_reviewed_only']['PARTIAL']} | "
          f"{payload['verdict_counts_peer_reviewed_only']['NOT MATCHED']} |", ""]
    Path(a.out_md).write_text("\n".join(L))

    for s in payload["summaries"]:
        if s["n"]:
            print(f"{s['set']:<52} n={s['n']:<3} median {s['median']:.3f}  "
                  f"range [{s['min']:.3f}, {s['max']:.3f}]  "
                  f"IQR [{s['q25']:.3f}, {s['q75']:.3f}]")
    print()
    print("verdicts, all rows            :", payload["verdict_counts"])
    print("verdicts, peer-reviewed only  :", payload["verdict_counts_peer_reviewed_only"])
    print(f"\nwrote {a.out_json}\nwrote {a.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
