#!/usr/bin/env python3
"""
Generate the two example label tables shipped with trivialbaselines.

Both are SYNTHETIC. No patient data of any kind is redistributed with this package.
They exist so that the quickstart runs in one command on a clean machine, and so that
the tool can be seen to produce a POSITIVE result on a benchmark that has a shortcut in
it and a NULL result on one that does not.

    shortcut_benchmark.csv
        A per-patient disease state, plus two artefacts of the sort that occur in real
        releases:
          * lesion-positive slices cluster in the middle third of the stack, so
            P(label | relative slice position) is informative;
          * the label rate differs between download batches, so an administrative
            field predicts the label.
        Expect: high slice-level positional AUROC, near-chance PATIENT-level positional
        AUROC, and a metadata baseline above its own permutation null.

    clean_benchmark.csv
        The same per-patient disease state, the same schema, the same prevalence -- but
        lesion slices are spread uniformly through the stack and the batch is assigned
        independently of the label. Expect every baseline near 0.5. This is the case
        that matters most for the tool's credibility: a benchmark where the null models
        do NOT reach the headline is a real and reportable outcome, and if the tool
        could not produce one, it would be measuring its own optimism.

Both files carry a subject-level `data_split` column, so the audit uses the dataset's
own split rather than inventing one -- which is what you want when auditing somebody
else's benchmark.

Regenerate with:  python make_examples.py
The seeds are fixed, so the output is byte-identical on any platform.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

N_PATIENTS = 200
PREVALENCE = 0.40
BATCHES = ("2019_release1", "2020_release1", "2021_release2", "2022_release2")
SCANNERS = ("Aera", "Skyra", "Prisma")


def _split_for(i: int) -> str:
    """Subject-level split: 140 train / 20 val / 40 test, assigned by index."""
    if i < 140:
        return "training"
    if i < 160:
        return "validation"
    return "test"


def build(kind: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(N_PATIENTS):
        depth = int(rng.integers(24, 37))
        diseased = bool(rng.random() < PREVALENCE)

        if kind == "shortcut":
            # Release batch tracks the label: later batches were enriched for
            # positives, exactly the way a benchmark grown by successive uploads
            # tends to be. This is the metadata shortcut.
            p_late = 0.75 if diseased else 0.25
            batch = BATCHES[2 + int(rng.random() < 0.5)] if rng.random() < p_late \
                else BATCHES[int(rng.random() < 0.5)]
        else:
            batch = BATCHES[int(rng.integers(0, len(BATCHES)))]

        scanner = SCANNERS[int(rng.integers(0, len(SCANNERS)))]
        tr = float(np.round(rng.normal(4000, 150), 1))

        # Which slices carry the finding.
        #
        # A CONTIGUOUS run cannot be positioned uniformly in a finite stack. Draw its
        # centre over the whole stack and clip, and edge lesions come out truncated, so
        # positives concentrate mid-stack; draw the centre only where the run fits
        # whole, and positives concentrate mid-stack even harder. Both were tried here
        # and both left the "clean" file with a centrality AUROC of 0.60-0.67 -- i.e. a
        # positional shortcut in the file that is supposed to have none. So the clean
        # file marks slices INDEPENDENTLY at a fixed rate, which is uniform in relative
        # position by construction, and only the shortcut file uses a contiguous run.
        if not diseased:
            lesion = set()
        elif kind == "shortcut":
            n_lesion = int(rng.integers(3, 8))
            half = n_lesion // 2
            lo, hi = int(0.35 * depth), max(int(0.65 * depth), int(0.35 * depth) + 1)
            centre = int(rng.integers(lo, hi))
            lesion = {min(max(centre + d, 0), depth - 1) for d in range(-half, half + 1)}
        else:
            rate = float(rng.integers(3, 8)) / depth
            lesion = {s for s in range(depth) if rng.random() < rate}
            if not lesion:                      # a diseased patient has >= 1 lesion slice
                lesion = {int(rng.integers(0, depth))}

        for s in range(depth):
            rows.append({
                "patient_id": f"P{i:04d}",
                "slice": s,
                "lesion": int(s in lesion),
                "data_split": _split_for(i),
                "release_batch": batch,
                "scanner_model": scanner,
                "TR": tr,
            })
    return pd.DataFrame(rows)


def main() -> None:
    for kind, seed, name in (("shortcut", 20260729, "shortcut_benchmark.csv"),
                             ("clean", 20260730, "clean_benchmark.csv")):
        df = build(kind, seed)
        out = HERE / name
        df.to_csv(out, index=False)
        pat = df.groupby("patient_id")["lesion"].max()
        print(f"{name}: {len(df)} slices, {df.patient_id.nunique()} patients, "
              f"{int(df.lesion.sum())} positive slices, "
              f"{int(pat.sum())} positive patients "
              f"({df.lesion.mean():.3f} slice / {pat.mean():.3f} patient prevalence)")


if __name__ == "__main__":
    main()
