#!/usr/bin/env python3
"""Progress of the stage-11 sweep: what is done, what failed, what is left."""
import collections
import glob
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_archzoo_sweep import ROOT, build_plan, expected_paths  # noqa: E402

plan = build_plan()
done, missing = [], []
for c in plan:
    (done if any(p.exists() for p in expected_paths(c)) else missing).append(c)

secs = collections.defaultdict(list)
fails = []
for f in sorted(glob.glob(str(ROOT / "_manifest_*.jsonl"))):
    for line in open(f):
        r = json.loads(line)
        if r["status"] == "ok":
            secs[(r["cell"]["tree"], r["cell"]["cohort"])].append(r["seconds"])
        else:
            fails.append(r)

print(f"CELLS: {len(done)}/{len(plan)} done, {len(missing)} remaining, {len(fails)} failures")
by = collections.Counter((c["tree"], c["cohort"]) for c in done)
tot = collections.Counter((c["tree"], c["cohort"]) for c in plan)
print(f"  {'tree':<18}{'cohort':<14}{'done/total':>12}{'mean s/run':>12}")
for k in sorted(tot):
    s = secs.get(k, [])
    print(f"  {k[0]:<18}{k[1]:<14}{by[k]:>5}/{tot[k]:<6}"
          f"{(sum(s) / len(s) if s else float('nan')):>12.0f}")
if fails:
    print("\nFAILURES:")
    for r in fails:
        c = r["cell"]
        print(f"  {c['tree']}/{c['cohort']}/{c['condition']}: {r['error'][:160]}")
live = subprocess.run(["pgrep", "-af", "run_archzoo_sweep.py"], capture_output=True, text=True)
print("\nLIVE:", len([l for l in live.stdout.splitlines() if l.strip()]), "sweep process(es)")
for l in live.stdout.splitlines():
    print("  ", l.split("run_archzoo_sweep.py")[-1].strip()[:110])
