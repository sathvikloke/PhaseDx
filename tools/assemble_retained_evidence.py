#!/usr/bin/env python3
"""Copy explicitly selected retained evidence into a private writing-input directory.

The supplied JSON lists {name,path,kind,scope} sources. No row-level arrays,
statistics, inference or new resampling are computed by this command.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    spec = json.loads(Path(a.spec).read_text())
    rows = []
    for s in spec["sources"]:
        name = s["name"]
        if Path(name).name != name:
            raise ValueError("name must be a filename")
        p = Path(s["path"])
        payload = p.read_bytes()
        rec = dict(s, sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload))
        if p.suffix == ".csv":
            with p.open(newline="") as f:
                rd = csv.DictReader(f)
                rec.update(columns=rd.fieldnames, rows=sum(1 for _ in rd))
        rows.append(rec)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    for r in rows:
        shutil.copyfile(r["path"], out / r["name"])
    (out / "SOURCE_BINDINGS_PRIVATE.json").write_text(json.dumps({
        "scope": "Retained inputs copied without numerical recomputation. Previous verification scope does not expand.",
        "sources": rows}, indent=2) + "\n")
    (out / "README.md").write_text(
        "# Retained writing inputs\n\nPrivate assembly; not a release authorization.\n\n" +
        "\n".join(f"- `{r['name']}`: {r['scope']}" for r in rows) + "\n")
    print(json.dumps({"files": len(rows), "bytes": sum(r["bytes"] for r in rows),
                      "numerical_recomputation": False}))


if __name__ == "__main__":
    main()
