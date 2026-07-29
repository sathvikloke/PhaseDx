#!/usr/bin/env python3
"""
summarise_archzoo.py -- read one statistics.json per architecture tree and answer
the two headline questions in numbers.

  Q1  IS THE CLINICAL NULL ROBUST TO ARCHITECTURE?
      -> does ANY architecture put the LOWER BOUND of the subject-clustered
         pooled out-of-fold patient-level CI above 0.500 for `phase` on a
         clinical cohort?
  Q2  IS THE CONFOUND RESULT ROBUST TO ARCHITECTURE?
      -> the brain receive-coil-count AUC per architecture.

Nothing is rounded up and nothing is imputed: a cell that did not run is printed
as MISSING, not as a blank.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "pipeline_out" / "results_arch"
TREES = ["resnet18", "complex_small", "densenet121", "resnet50", "convnext_tiny",
         "vit_b_16", "resnet18_scratch"]
CLINICAL = ["prostate_t2", "prostate_dwi"]
CONDITIONS = ["magnitude", "phase", "both"]


def load(tree: str):
    p = ROOT / tree / "statistics.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def cell(stats, cohort, condition):
    """
    The one estimate for (cohort, condition) in this architecture's tree.

    `_scratch/` is run_tagged's per-run temp directory. If stage 4 runs while a
    cell is still training, s04's rglob sees the in-flight payload there and
    builds it into its OWN split family -- which would otherwise be picked up
    here as if it were the cohort estimate. Exclude it by path.
    """
    if stats is None:
        return None
    hits = [r for r in stats["runs"]
            if r["cohort"] == cohort and r["condition"] == condition
            and "/_scratch/" not in str(r.get("path", ""))]
    if not hits:
        return None
    # prefer the pooled out-of-fold estimate when both forms are present
    hits.sort(key=lambda r: (not r.get("pooled", False), -(r.get("n_folds") or 0)))
    return hits[0]


def fmt(r, level="patient_level_mean"):
    if r is None:
        return "MISSING"
    d = r[level]
    if d.get("auc") is None:
        return f"n/a ({d.get('reason')})"
    s = f"{d['auc']:.3f} [{d['ci_lo']:.3f}, {d['ci_hi']:.3f}]"
    # A pool over fewer than the expected folds is NOT the full cohort. Say so
    # in the cell rather than letting a partial estimate read as a complete one.
    nf = r.get("n_folds")
    if nf is not None and nf != 5:
        s += f" !{nf}/5f"
    return s


def init_of(stats):
    """What initialisation the runs in this tree ACTUALLY got."""
    if stats is None:
        return "?"
    inits = set()
    for r in stats["runs"]:
        for p in str(r.get("path", "")).split(" + "):
            stem = Path(p).stem
            if "__" in stem:
                inits.add(stem.split("__")[2])
    return ",".join(sorted(inits)) or "?"


def main() -> int:
    loaded = {t: load(t) for t in TREES}

    for level, title in (("patient_level_mean", "PATIENT-LEVEL (mean over slices), "
                                                "subject-clustered 95% CI"),
                         ("slice_level", "SLICE-LEVEL, subject-clustered 95% CI")):
        print("=" * 108)
        print(f"POOLED OUT-OF-FOLD AUC -- {title}")
        print("=" * 108)
        for cohort in CLINICAL:
            print(f"\n  cohort = {cohort}   (5-fold subject-level CV, seed 42, "
                  f"pooled out-of-fold)")
            print(f"  {'architecture':<18}{'init':<10}" +
                  "".join(f"{c:<32}" for c in CONDITIONS))
            for t in TREES:
                s = loaded[t]
                row = "".join(f"{fmt(cell(s, cohort, c), level):<32}" for c in CONDITIONS)
                print(f"  {t:<18}{init_of(s):<10}{row}")
        print()

    # ---------------- brain: the mechanism ----------------
    print("=" * 108)
    print("BRAIN CONFOUND COHORT -- phase/magnitude predicting RECEIVE-COIL COUNT")
    print("official split, 454 subjects cached / test split clustered by subject")
    print("=" * 108)
    print(f"  {'architecture':<18}{'init':<10}{'condition':<12}"
          f"{'slice AUC [CI]':<30}{'patient AUC [CI]':<30}{'n_subj':>7}")
    for t in TREES:
        s = loaded[t]
        for c in ("magnitude", "phase"):
            r = cell(s, "brain", c)
            n = r["patient_level_mean"]["n_clusters"] if r else "-"
            print(f"  {t:<18}{init_of(s):<10}{c:<12}{fmt(r, 'slice_level'):<30}"
                  f"{fmt(r, 'patient_level_mean'):<30}{str(n):>7}")

    # ---------------- the two verdicts ----------------
    print()
    print("=" * 108)
    print("Q1  DOES ANY ARCHITECTURE LIFT A PHASE CI LOWER BOUND ABOVE 0.500 "
          "ON A CLINICAL COHORT?")
    print("=" * 108)
    breaches, checked, missing = [], 0, []
    for t in TREES:
        s = loaded[t]
        for cohort in CLINICAL:
            for level in ("patient_level_mean", "slice_level"):
                r = cell(s, cohort, "phase")
                if r is None:
                    missing.append(f"{t}/{cohort}/phase/{level}")
                    continue
                lo = r[level].get("ci_lo")
                if lo is None:
                    continue
                checked += 1
                if lo > 0.500:
                    breaches.append((t, cohort, level, r[level]["auc"], lo,
                                     r[level]["ci_hi"]))
    print(f"  phase cells with a computable CI: {checked}"
          f"   (missing: {len(missing)})")
    if breaches:
        print("  CI LOWER BOUND ABOVE 0.500 IN:")
        for b in breaches:
            print(f"    {b[0]:<18}{b[1]:<14}{b[2]:<20}AUC {b[3]:.3f} "
                  f"[{b[4]:.3f}, {b[5]:.3f}]")
    else:
        print("  NO. Every phase CI lower bound that was computed sits at or below 0.500.")

    # max phase AUC observed, and the maximum-over-architectures caveat
    best = None
    for t in TREES:
        s = loaded[t]
        for cohort in CLINICAL:
            r = cell(s, cohort, "phase")
            if r and r["patient_level_mean"].get("auc") is not None:
                v = r["patient_level_mean"]["auc"]
                if best is None or v > best[0]:
                    best = (v, t, cohort, r["patient_level_mean"])
    if best:
        d = best[3]
        print(f"\n  highest phase patient-level AUC over all architectures x clinical "
              f"cohorts: {best[0]:.3f} [{d['ci_lo']:.3f}, {d['ci_hi']:.3f}]"
              f"  ({best[1]}, {best[2]})")
        print("  (the maximum over a sweep is biased upward; its nominal CI is not a p-value)")

    print()
    print("=" * 108)
    print("Q2  IS THE CONFOUND (phase -> receive-coil count) ROBUST TO ARCHITECTURE?")
    print("=" * 108)
    vals = []
    for t in TREES:
        r = cell(loaded[t], "brain", "phase")
        if r and r["slice_level"].get("auc") is not None:
            vals.append((t, r["slice_level"]["auc"], r["slice_level"]["ci_lo"],
                         r["slice_level"]["ci_hi"]))
    if vals:
        print(f"  brain phase slice-level AUC across {len(vals)} architectures: "
              f"min {min(v[1] for v in vals):.3f}, max {max(v[1] for v in vals):.3f}")
        n_above = sum(1 for v in vals if v[2] > 0.500)
        print(f"  architectures whose brain-phase CI lower bound clears 0.500: "
              f"{n_above}/{len(vals)}")
    else:
        print("  no brain phase cells available")

    if missing:
        print("\nMISSING CELLS (not run / stage 4 not yet over them):")
        for m in sorted(set(m.rsplit('/', 1)[0] for m in missing)):
            print(f"    {m}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
