#!/usr/bin/env python3
"""Stage the original fixed B25 matrix into a NEW run. Execution requires --execute."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys

from run_external import run_queue


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evidence", default=str(Path(__file__).resolve().parents[1] / "reproducibility/synthetic_b25"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--deadline-utc", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    evidence, out = Path(a.evidence).resolve(), Path(a.out).resolve()
    lock = json.loads((evidence / "state/LOCK25.json").read_text())
    for name, expected in lock["hashes"].items():
        if hashlib.sha256((evidence / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError("frozen source/config changed: " + name)
    deadline = datetime.datetime.fromisoformat(a.deadline_utc.replace("Z", "+00:00"))
    if deadline.tzinfo is None:
        raise ValueError("deadline requires timezone")
    jobs = []
    for kind in ("transport", "precision"):
        for i in range(12):
            script = out / "scripts" / (kind + "25.py")
            config = out / "config" / f"{kind}_{i:02}.json"
            cfg = json.loads((evidence / "config" / config.name).read_text())
            output = out / kind / "shards" / f"{kind}_shard_{cfg['shard']}.csv"
            jobs.append({"id": f"{kind}_{i:02}", "command": [sys.executable, str(script), str(config)],
                         "cwd": str(out), "script": str(script), "config": str(config),
                         "inputs": [str(out / n) for n in lock["hashes"]],
                         "outputs": [str(output)], "memory_bytes": 384*1024**2, "reserve_bytes": 8*1024**2})
    if not a.execute:
        print(json.dumps({"jobs": len(jobs), "out": str(out), "writes": False,
                          "design": "fixed 12x200 transport configurations and 108x500 precision cells",
                          "note": "Use retained aggregates for the paper. Reproduction is optional, not a new queued study."}, indent=2))
        return 0
    if deadline.timestamp() <= datetime.datetime.now(datetime.timezone.utc).timestamp():
        raise ValueError("deadline has already passed")
    out.mkdir(exist_ok=False, parents=True)
    for name in list(lock["hashes"]) + ["state/LOCK25.json"]:
        p = out / name
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(evidence / name, p)
    (out / "JOBS.json").write_text(json.dumps({"jobs": jobs}, indent=2))
    result = run_queue({"jobs": jobs}, out, deadline.timestamp(), a.workers)
    print(json.dumps(result))
    return 0 if all(v == "done" for v in result.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
