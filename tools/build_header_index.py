#!/usr/bin/env python3
"""Bind completed versioned header shards for manifest/selection reuse."""
import argparse
import json
from pathlib import Path

from phasedx_external.protocol28 import sha_file


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ("plans", "shards", "image-root", "out"):
        ap.add_argument("--" + key, required=True)
    a = ap.parse_args()
    rows, seen = [], set()
    for plan in sorted(Path(a.plans).glob("*.json")):
        # exFAT AppleDouble sidecars are filesystem metadata, not input plans.
        if plan.name.startswith("._"):
            continue
        d = json.loads(plan.read_text())
        root = Path(a.shards) / plan.stem
        summary, records = root/"SUMMARY.json", root/"headers.jsonl"
        s = json.loads(summary.read_text())
        if s["status"] != "complete" or s["identity_failures"] or s.get("io_failure"):
            raise RuntimeError("unfinished or unauthenticated header shard")
        if s["completed"] != len(d["files"]) or s["plan_sha256"] != sha_file(plan):
            raise RuntimeError("header count/plan mismatch")
        if Path(d["image_root"]).resolve() != Path(a.image_root).resolve():
            raise RuntimeError("different source root")
        for item in d["files"]:
            if item["relpath"] in seen:
                raise RuntimeError("duplicate planned source")
            seen.add(item["relpath"])
        row = {"plan": str(plan.resolve()), "summary": str(summary.resolve()),
               "records": str(records.resolve())}
        row.update({k+"_sha256": sha_file(v) for k, v in list(row.items())})
        rows.append(row)
    if not rows:
        raise RuntimeError("no shards")
    with open(a.out, "x") as f:
        json.dump({"image_root": str(Path(a.image_root).resolve()), "expected_files": len(seen),
                   "shards": rows, "scope": "Header records from source-hash-verified files; no clinical selection."}, f, indent=2)
    print(json.dumps({"shards": len(rows), "files": len(seen)}))


if __name__ == "__main__":
    main()
