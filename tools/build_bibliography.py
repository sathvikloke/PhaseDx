#!/usr/bin/env python3
"""Build full-author BibTeX from retained Crossref records; no network or claims review."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re


def clean(v):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", str(v))).strip()


def tex(v):
    return clean(v).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    wanted = list(csv.DictReader(open(a.registry)))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    found = {}
    for path in sorted(Path(a.records).rglob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (ValueError, UnicodeDecodeError):
            continue
        msg = data.get("message", {}) if isinstance(data, dict) else {}
        items = msg.get("items", []) if isinstance(msg, dict) else []
        if isinstance(msg, dict) and msg.get("DOI"):
            items = [msg]
        for item in items:
            if item.get("DOI"):
                found.setdefault(item["DOI"].lower(), (item, path))
    entries, bindings, missing = [], [], []
    for row in wanted:
        doi = row["doi"].lower()
        if doi not in found:
            missing.append(row["id"])
            continue
        m, path = found[doi]
        title = clean(m.get("title", [""])[0])
        if title.casefold() != clean(row["title"]).casefold():
            # Retain the exact publisher metadata, but flag the mismatch for review.
            title_status = "registry_title_differs"
        else:
            title_status = "matches_registry"
        authors = " and ".join(
            tex(x.get("family", x.get("name", ""))) + (", " + tex(x["given"]) if x.get("given") else "")
            for x in m.get("author", []))
        date = m.get("published", m.get("issued", {})).get("date-parts", [[row["year"]]])[0]
        fields = {"title": "{"+tex(title)+"}", "author": authors, "year": str(date[0]),
                  "journal": tex(m.get("container-title", [""])[0]), "doi": doi,
                  "url": "https://doi.org/"+doi}
        for k in ("volume", "issue", "page"):
            if m.get(k):
                fields[{"issue": "number", "page": "pages"}.get(k, k)] = tex(m[k])
        entries.append("@article{" + row["id"] + ",\n" +
                       ",\n".join(f"  {k} = {{{v}}}" for k, v in fields.items() if v) + "\n}\n")
        bindings.append({"id": row["id"], "doi": doi, "source": str(path),
                         "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "title_status": title_status, "authors": len(m.get("author", [])),
                         "scope": "Bibliographic identity only; does not establish a scientific claim."})
    (out / "references.bib").write_text("\n".join(entries))
    (out / "BIBLIOGRAPHY_RECEIPT.json").write_text(json.dumps({
        "bindings": bindings, "missing": missing, "network_requests": 0}, indent=2)+"\n")
    print(json.dumps({"entries": len(entries), "missing": missing,
                      "title_mismatches": [b["id"] for b in bindings if b["title_status"] != "matches_registry"]}))
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
