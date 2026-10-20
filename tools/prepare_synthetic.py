#!/usr/bin/env python3
"""Copy completed B25 source/configs and aggregate evidence; never run experiments."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

SUMMARIES = (
    "transport/transport_source_to_target_change_summary.csv",
    "transport/audit_pattern_vs_transport_loss.csv",
    "transport/transport_image_minus_geometry_summary.csv",
    "transport/transport_auroc_summary.csv",
    "precision/precision_108cells.csv",
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, help="Existing completed batch25")
    ap.add_argument("--out", required=True, help="New portable directory")
    a = ap.parse_args()
    source, out = Path(a.source), Path(a.out)
    lock = json.loads((source / "state/LOCK25.json").read_text())
    names = list(lock["hashes"]) + ["state/LOCK25.json"] + list(SUMMARIES)
    records = []
    # Validate all inputs before creating the destination.
    for name in names:
        payload = (source / name).read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if name in lock["hashes"] and digest != lock["hashes"][name]:
            raise RuntimeError("original source/config no longer matches LOCK25: " + name)
        records.append({"relative_path": name, "sha256": digest, "bytes": len(payload),
                        "scope": "original frozen source/config" if name in lock["hashes"]
                        else "original lock" if name.endswith("LOCK25.json") else "retained aggregate; no recomputation"})
    out.mkdir(parents=True, exist_ok=False)
    for rec in records:
        dest = out / rec["relative_path"]
        dest.parent.mkdir(exist_ok=True, parents=True)
        shutil.copyfile(source / rec["relative_path"], dest)
    (out / "PORTABLE_EVIDENCE.json").write_text(json.dumps({
        "origin": "PhaseDx Batch25, completed October 14 2026",
        "scope": "No experiments rerun. 265 previously reviewed selected arithmetic checks; not a full independent repeat.",
        "clinical_interpretation": "Synthetic counterexample and hypothetical precision; no external clinical performance.",
        "files": records}, indent=2) + "\n")
    print(f"Copied {len(records)} source/config/aggregate artifacts; no numerical experiment executed.")


if __name__ == "__main__":
    main()
