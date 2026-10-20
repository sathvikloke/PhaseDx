"""Consume complete, source-authenticated header shards without re-reading images."""
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(index_path, image_root):
    """Return authenticated headers. A failed/partial I/O attempt is never a cohort."""
    index = json.loads(Path(index_path).read_text())
    if Path(index["image_root"]).resolve() != Path(image_root).resolve():
        raise ValueError("header cache belongs to another image root")
    rows, seen = [], set()
    for shard in index["shards"]:
        for field in ("plan", "summary", "records"):
            if sha(shard[field]) != shard[field + "_sha256"]:
                raise ValueError("header cache " + field + " identity changed")
        plan = json.loads(Path(shard["plan"]).read_text())
        summary = json.loads(Path(shard["summary"]).read_text())
        if (summary["status"] != "complete" or summary.get("identity_failures") != 0
                or summary.get("io_failure") or summary["completed"] != len(plan["files"])
                or summary["plan_sha256"] != shard["plan_sha256"]):
            raise ValueError("header cache is partial or contains source/I/O failures")
        if Path(plan["image_root"]).resolve() != Path(image_root).resolve():
            raise ValueError("header shard belongs to another source")
        expected = {x["relpath"]: x for x in plan["files"]}
        if len(expected) != len(plan["files"]):
            raise ValueError("duplicate planned source")
        found = set()
        with open(shard["records"]) as f:
            for line in f:
                r = json.loads(line)
                p = Path(r["relpath"])
                if p.is_absolute() or ".." in p.parts:
                    raise ValueError("unsafe source path")
                if r["relpath"] in seen or r["relpath"] not in expected:
                    raise ValueError("duplicate or unplanned header source")
                e = expected[r["relpath"]]
                if not r["hash_verified"] or any(r[k] != e[k] for k in
                        ("relpath", "sha256", "source_exam_id", "assignment")) or int(r["size"]) != int(e["size"]):
                    raise ValueError("header identity differs from frozen source")
                seen.add(r["relpath"])
                found.add(r["relpath"])
                rows.append(r)
        if found != set(expected):
            raise ValueError("missing planned headers")
    if len(rows) != index["expected_files"]:
        raise ValueError("header cache count differs from source inventory")
    return rows
