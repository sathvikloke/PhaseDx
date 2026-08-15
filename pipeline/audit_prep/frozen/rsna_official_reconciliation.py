#!/usr/bin/env python3
"""Reconcile the pixel-free tabular mirror against the OFFICIAL RSNA ICH release.

    venv/bin/python pipeline/audit_prep/frozen/rsna_official_reconciliation.py \
        /path/to/stage_2_train.csv

Why this exists
---------------
Every RSNA ICH value in the manuscript rests on a slice ordering and a subject
identifier that the official release does not contain; both come from a public
pixel-free tabular mirror. Table 2 reports internal-consistency tests of that
mapping and states their limit in terms: they can falsify a bad mapping and can
never certify a good one, because the official label file had not been obtained.

This obtains it and does the check the internal tests could not: a row-for-row
comparison of the mirror's per-image labels against the official ones, joined on
the SOP instance identifier, for all six official labels.

What it CAN establish: that the mirror reproduces the official label of every
image it claims to hold, and which images it omits.

What it still CANNOT establish: that the mirror's SERIES and PATIENT groupings,
or its within-series slice ORDER, are correct. The official file carries neither
a subject identifier nor a slice index -- that absence is the whole reason the
mirror is used. Label agreement therefore bounds one failure mode and leaves the
ordering evidence resting on the run-length and orientation tests in Table 2.
Do not let a clean result here be read as validating the ordering.

Emits pipeline_out/trivial_baselines/rsna_official_reconciliation.json.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
MIRROR = REPO / "pipeline_out" / "audit_data" / "rsna_ich_slices.csv"
OUT = REPO / "pipeline_out" / "trivial_baselines" / "rsna_official_reconciliation.json"

SUBTYPES = ["epidural", "intraparenchymal", "intraventricular",
            "subarachnoid", "subdural", "any"]
# the mirror calls the 'any' column 'label'
MIRROR_COL = {s: ("label" if s == "any" else s) for s in SUBTYPES}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main() -> None:
    t0 = time.time()
    official_path = Path(sys.argv[1])

    print("reading the official release ...")
    off = pd.read_csv(official_path)
    n_raw = len(off)

    # The official release ships a handful of EXACT duplicate rows. They are
    # dropped, but only after checking that no duplicated ID carries conflicting
    # labels -- a conflict would be a defect in the release itself and must not be
    # silently resolved. Recorded in the report either way.
    dup_mask = off.duplicated("ID", keep=False)
    dup_ids = int(off.loc[dup_mask, "ID"].nunique())
    conflicting = int((off[dup_mask].groupby("ID").Label.nunique() > 1).sum())
    if conflicting:
        raise SystemExit(f"{conflicting} duplicated IDs carry conflicting labels; "
                         "refusing to guess which is authoritative")
    off = off.drop_duplicates("ID", keep="first")
    print(f"  official: {n_raw:,} rows, {n_raw - len(off):,} exact duplicate rows "
          f"dropped over {dup_ids} IDs (no conflicts)")

    # ID_<sop>_<subtype>  ->  two columns
    parts = off["ID"].str.rsplit("_", n=1, expand=True)
    off["sop"] = parts[0]
    off["subtype"] = parts[1]
    wide = off.pivot(index="sop", columns="subtype", values="Label")
    wide = wide[SUBTYPES]
    print(f"            {len(off):,} unique rows -> {len(wide):,} images "
          f"x {len(SUBTYPES)} labels")

    print("reading the mirror ...")
    mir = pd.read_csv(MIRROR, usecols=["image_id", "series_id", "patient_id"]
                              + [MIRROR_COL[s] for s in SUBTYPES])
    mir = mir.set_index("image_id")
    print(f"  mirror  : {len(mir):,} images")

    off_ids = set(wide.index)
    mir_ids = set(mir.index)
    both = sorted(off_ids & mir_ids)
    only_off = sorted(off_ids - mir_ids)
    only_mir = sorted(mir_ids - off_ids)
    print(f"\n  in both            : {len(both):,}")
    print(f"  official only      : {len(only_off):,}")
    print(f"  mirror only        : {len(only_mir):,}"
          f"{'   <-- MIRROR INVENTS IMAGES' if only_mir else ''}")

    W = wide.loc[both]
    M = mir.loc[both]

    per_label = {}
    total_disagree = 0
    print("\n  label agreement, row for row:")
    for s in SUBTYPES:
        a = W[s].to_numpy().astype(int)
        b = M[MIRROR_COL[s]].to_numpy().astype(int)
        ne = int((a != b).sum())
        total_disagree += ne
        per_label[s] = {
            "n_compared": int(len(a)),
            "n_disagree": ne,
            "disagreement_rate": ne / max(len(a), 1),
            "official_positives": int(a.sum()),
            "mirror_positives": int(b.sum()),
        }
        flag = "EXACT" if ne == 0 else f"{ne:,} MISMATCHES"
        print(f"    {s:<18s} {flag:<18s} official pos {int(a.sum()):>7,}  "
              f"mirror pos {int(b.sum()):>7,}")

    # is the mirror's 'any' the OR of the five subtypes, as the release defines it?
    five = W[[s for s in SUBTYPES if s != "any"]].to_numpy().astype(int)
    any_is_or = int((W["any"].to_numpy().astype(int) != (five.sum(1) > 0)).sum())

    report = {
        "tool": "rsna_official_reconciliation",
        "version": "1.0",
        "official_file": official_path.name,
        "official_sha256": sha256(official_path),
        "mirror_file": MIRROR.name,
        "mirror_sha256": sha256(MIRROR),
        "NOT_FOR_SUBMISSION": "working artefact; contains absolute local paths",
        "n_official_rows_raw": int(n_raw),
        "n_exact_duplicate_rows_dropped": int(n_raw - len(off)),
        "n_duplicated_ids": dup_ids,
        "n_duplicated_ids_with_conflicting_labels": conflicting,
        "n_official_images": int(len(wide)),
        "n_mirror_images": int(len(mir)),
        "n_in_both": len(both),
        "n_official_only": len(only_off),
        "n_mirror_only": len(only_mir),
        "official_only_ids": only_off[:50],
        "mirror_only_ids": only_mir[:50],
        "per_label": per_label,
        "total_label_disagreements": total_disagree,
        "official_any_equals_or_of_five_violations": any_is_or,
        "verdict": ("mirror labels reproduce the official release exactly on every "
                    "shared image" if total_disagree == 0 else
                    "MIRROR LABELS DISAGREE WITH THE OFFICIAL RELEASE"),
        "what_this_does_not_establish":
            "the mirror's series and patient groupings and its within-series slice "
            "order are NOT tested here; the official file carries neither a subject "
            "identifier nor a slice index, which is why the mirror is used at all",
    }
    OUT.write_text(json.dumps(report, indent=1))

    print(f"\n  total disagreements over all six labels: {total_disagree:,}")
    print(f"  'any' == OR(five subtypes) violations   : {any_is_or:,}")
    print(f"\n  VERDICT: {report['verdict']}")
    if only_off:
        print(f"\n  images in the official release the mirror does not hold "
              f"({len(only_off)}):")
        for i in only_off[:12]:
            print(f"    {i}")
    print(f"\n  wrote {OUT}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
