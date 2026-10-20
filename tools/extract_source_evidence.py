#!/usr/bin/env python3
"""Extract exact, bounded text locators from explicitly selected local sources."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    spec = json.loads(Path(a.spec).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    rows, receipts = [], []
    failed = False
    for item in spec["sources"]:
        path = Path(item["path"])
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        matched = not item.get("expected_sha256") or digest == item["expected_sha256"]
        failed |= not matched
        receipts.append({"source_id": item["id"], "path": str(path), "sha256": digest,
                         "expected_sha256": item.get("expected_sha256"),
                         "identity_matches": matched, "evidence_class": item["evidence_class"]})
        if not matched:
            continue
        lines = data.decode("utf-8").splitlines()
        for query in item.get("queries", []):
            hits = [i for i, line in enumerate(lines) if re.search(query["pattern"], line)]
            for i in hits:
                lo, hi = max(0, i-query.get("context", 0)), min(len(lines), i+query.get("context", 0)+1)
                rows.append({"source_id": item["id"], "sha256": digest, "claim_key": query["id"],
                             "line_start": lo+1, "line_end": hi, "text": "\n".join(lines[lo:hi]),
                             "evidence_class": item["evidence_class"]})
            if not hits:
                rows.append({"source_id": item["id"], "sha256": digest, "claim_key": query["id"],
                             "line_start": "", "line_end": "", "text": "NO_MATCH",
                             "evidence_class": item["evidence_class"]})
    with (out / "SOURCE_EXCERPTS.csv").open("x", newline="") as f:
        w = csv.DictWriter(f, ["source_id", "sha256", "claim_key", "line_start", "line_end", "text", "evidence_class"])
        w.writeheader()
        w.writerows(rows)
    (out / "SOURCE_RECEIPTS.json").write_text(json.dumps({
        "scope": "Static text extraction only; no training, image reconstruction, or outcome calculation.",
        "sources": receipts, "excerpt_rows": len(rows), "identity_failure": failed}, indent=2) + "\n")
    print(json.dumps({"sources": len(receipts), "excerpts": len(rows), "identity_failure": failed}))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
