"""
run_all.py
----------
One-command, resumable orchestrator for the whole PhaseDx pipeline.

    s01_labels  ->  s02_prostate + s02_breast  ->  s03_train (cohorts x
    conditions x seeds)  ->  s04_stats  ->  s05_controls  ->  s06_report

Design notes, in the order they will matter to you:

* **Resumable by output, not by bookkeeping.** A stage is "done" when the files
  it is supposed to produce exist and are non-empty. There is no hidden state
  file that can disagree with the disk. `--force` re-runs everything;
  `--force-stage NAME` re-runs one.

* **It refuses to fight a running build.** Stage 2 writes a multi-hundred-MB
  HDF5 in place. If a stage looks incomplete but its outputs were touched in the
  last few minutes, another process is probably mid-write, and starting a second
  writer would corrupt the cache. Such a stage is skipped with a loud message
  unless you pass --force. This is not hypothetical: the prostate cache is
  normally built in the background while the rest of the pipeline is developed.

* **Stages 4 and 5 may not exist yet.** They are declared optional. A missing
  script is logged as SKIPPED (missing script) and the run continues to stage 6,
  which degrades to an INCONCLUSIVE verdict and names what was missing. The
  pipeline never fabricates a control it did not run.

* **--quick is for wiring, not for results.** Few files, 2 epochs, 1 seed. Any
  numbers it produces are meaningless and stage 6 will say so via its sample
  size caveats. It exists so that a change to the plumbing can be tested in
  minutes without touching the full drive.

Logging: every line goes to `pipeline_out/run.log` in a parseable
`timestamp | LEVEL | stage=... | event=... | ...` form, plus per-stage stdout in
`pipeline_out/logs/<stage>.log`, plus a machine-readable `pipeline_out/run_state.json`
with per-stage timing, return codes and commands.

Usage:
    python pipeline/run_all.py                    # full pipeline, resume
    python pipeline/run_all.py --quick            # smoke run
    python pipeline/run_all.py --dry-run          # print the plan, run nothing
    python pipeline/run_all.py --force            # ignore existing outputs
    python pipeline/run_all.py --stages s03,s06   # only these
    python pipeline/run_all.py --from s04         # this stage onwards
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

PIPELINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PIPELINE_DIR.parent
OUT_ROOT = PROJECT_ROOT / "pipeline_out"

COHORTS = ("prostate_dwi", "prostate_t2", "breast")
CONDITIONS = ("magnitude", "phase", "both")
DEFAULT_SEEDS = (42, 123, 7)

# A stage whose outputs were touched this recently is assumed to be mid-write by
# another process, and is left alone.
RECENT_WRITE_SECONDS = 300

logger = logging.getLogger("run_all")


# --------------------------------------------------------------------------
# Stage model
# --------------------------------------------------------------------------

@dataclass
class Stage:
    name: str
    description: str
    script: str                       # relative to pipeline/
    argv: List[str] = field(default_factory=list)
    outputs: List[Path] = field(default_factory=list)
    # (directory, glob) pairs that must match at least one file. Needed for
    # stage 5, whose filenames depend on which acquisition arms and confound
    # columns a cohort turns out to have, so they cannot be predicted here.
    output_globs: List[tuple] = field(default_factory=list)
    optional: bool = False            # missing script => skip, do not fail
    always_run: bool = False          # cheap and depends on everything upstream
    enabled: bool = True

    @property
    def script_path(self) -> Path:
        return PIPELINE_DIR / self.script

    def command(self, python: str) -> List[str]:
        return [python, str(self.script_path), *self.argv]

    def missing_outputs(self) -> List[Path]:
        missing = [p for p in self.outputs
                   if not (p.exists() and (p.is_dir() or p.stat().st_size > 0))]
        for d, pattern in self.output_globs:
            if not any(Path(d).glob(pattern)):
                missing.append(Path(d) / pattern)
        return missing

    def n_expected(self) -> int:
        return len(self.outputs) + len(self.output_globs)

    def is_done(self) -> bool:
        return bool(self.outputs or self.output_globs) and not self.missing_outputs()

    def _existing_outputs(self) -> List[Path]:
        found = [p for p in self.outputs if p.exists()]
        for d, pattern in self.output_globs:
            found.extend(Path(d).glob(pattern))
        return found

    def recently_written(self, seconds: int = RECENT_WRITE_SECONDS) -> List[Path]:
        now = time.time()
        touched = []
        for p in self._existing_outputs():
            try:
                if (now - p.stat().st_mtime) < seconds:
                    touched.append(p)
            except OSError:
                pass
        return touched


def build_stages(args) -> List[Stage]:
    cohorts = [c for c in args.cohorts if c in COHORTS]
    seeds = list(args.seeds)
    conditions = list(args.conditions)
    cache = OUT_ROOT / "cache"
    cohorts_dir = OUT_ROOT / "cohorts"
    results = OUT_ROOT / "results"
    report = OUT_ROOT / "report"

    stages: List[Stage] = []

    # Every stage is given its paths EXPLICITLY. The stage scripts all default
    # to common.OUT_ROOT, so relying on those defaults would silently ignore
    # --out-root and, worse, let a rehearsal run read or overwrite the real
    # cache. Any new stage added here must pass its own paths too.

    # ---- stage 1 ---------------------------------------------------------
    stages.append(Stage(
        name="s01",
        description="build cohort tables and splits from the label files",
        script="s01_labels.py",
        argv=["--cohorts", *COHORTS, "--out", str(cohorts_dir)],
        outputs=[cohorts_dir / "s01_summary.json",
                 *[cohorts_dir / f"{c}_cohort.csv" for c in COHORTS]],
    ))

    # ---- stage 2 ---------------------------------------------------------
    prostate_cohorts = [c for c in cohorts if c.startswith("prostate")]
    if prostate_cohorts:
        which = "both" if len(prostate_cohorts) == 2 else prostate_cohorts[0]
        argv = ["--cohort", which, "--out", str(cache)]
        if args.quick:
            argv += ["--limit", str(args.quick_files)]
        stages.append(Stage(
            name="s02_prostate",
            description=f"reconstruct + cache prostate ({which})",
            script="s02_prostate.py",
            argv=argv,
            outputs=[p for c in prostate_cohorts
                     for p in (cache / f"{c}.h5", cache / f"{c}_index.csv")],
        ))
    if "breast" in cohorts:
        argv = ["--out-h5", str(cache / "breast.h5"),
                "--out-csv", str(cache / "breast_index.csv")]
        if args.quick:
            argv += ["--limit", str(args.quick_files)]
        stages.append(Stage(
            name="s02_breast",
            description="grid + cache breast radial acquisitions",
            script="s02_breast.py",
            argv=argv,
            outputs=[cache / "breast.h5", cache / "breast_index.csv"],
        ))

    # ---- stage 3: one stage per cohort, all conditions x seeds inside ----
    for cohort in cohorts:
        argv = ["--cohort", cohort,
                "--conditions", ",".join(conditions),
                "--seeds", ",".join(str(s) for s in seeds),
                "--cache-dir", str(cache),
                "--results-dir", str(results),
                "--ckpt-dir", str(OUT_ROOT / "checkpoints")]
        if args.quick:
            argv += ["--epochs", str(args.quick_epochs)]
        elif args.epochs:
            argv += ["--epochs", str(args.epochs)]
        stages.append(Stage(
            name=f"s03_{cohort}",
            description=f"train {cohort}: {len(conditions)} conditions x {len(seeds)} seeds",
            script="s03_train.py",
            argv=argv,
            outputs=[results / f"{cohort}_{cond}_seed{seed}.json"
                     for cond in conditions for seed in seeds],
        ))

    # ---- optional background-region falsification runs -------------------
    # s05_controls.py owns the controls. Until it exists, stage 3 can still
    # produce the one control it is capable of on its own -- the background-only
    # region run -- and s06 will pick it up, labelled as a fallback.
    if args.background_control:
        for cohort in cohorts:
            bg_dir = results / "controls" / "background"
            stages.append(Stage(
                name=f"s03bg_{cohort}",
                description=f"falsification control: {cohort} trained on background only",
                script="s03_train.py",
                argv=["--cohort", cohort, "--conditions", "phase",
                      "--seeds", ",".join(str(s) for s in seeds),
                      "--region", "background",
                      "--cache-dir", str(cache),
                      "--results-dir", str(bg_dir),
                      "--ckpt-dir", str(OUT_ROOT / "checkpoints" / "background")]
                + (["--epochs", str(args.quick_epochs)] if args.quick else []),
                outputs=[bg_dir / f"{cohort}_phase_seed{seed}.json" for seed in seeds],
            ))

    # ---- stage 4 ---------------------------------------------------------
    stages.append(Stage(
        name="s04",
        description="subject-clustered bootstrap CIs, DeLong comparisons, Holm",
        script="s04_stats.py",
        argv=["--results-dir", str(results),
              "--cache-dir", str(cache),
              "--cohort-dir", str(cohorts_dir),
              "--out", str(results / "statistics.json")],
        outputs=[results / "statistics.json"],
        optional=True,
    ))

    # ---- stage 5: one invocation per cohort ------------------------------
    # s05 requires --cohort and writes one JSON per control run into
    # pipeline_out/controls/results (a SIBLING of results/, not a child).
    # Its filenames encode the acquisition arms and confound columns each
    # cohort happens to have, so completeness is checked by glob per control.
    controls_results = OUT_ROOT / "controls" / "results"
    for cohort in cohorts:
        argv = ["--cohort", cohort,
                "--results-dir", str(controls_results),
                "--ckpt-dir", str(OUT_ROOT / "controls" / "checkpoints"),
                "--headline-dir", str(results),
                "--cache-dir", str(cache),
                "--cohort-dir", str(cohorts_dir),
                "--seeds", ",".join(str(s) for s in seeds[:1])]
        if args.quick:
            argv += ["--epochs", str(args.quick_epochs),
                     "--n-permutations", "4", "--n-boot", "300"]
        elif args.epochs:
            argv += ["--epochs", str(args.epochs)]
        stages.append(Stage(
            name=f"s05_{cohort}",
            description=f"falsification suite for {cohort}: permutation, background, "
                        f"phase-scramble, acquisition split, confound predictability",
            script="s05_controls.py",
            argv=argv,
            output_globs=[(controls_results, f"{cohort}__{ctrl}__*.json")
                          for ctrl in ("label_permutation", "background_only",
                                       "phase_scramble", "acquisition_split",
                                       "confound_predictability")],
            optional=True,
        ))

    # ---- stage 6 ---------------------------------------------------------
    stages.append(Stage(
        name="s06",
        description="figures, RESULTS.md and the verdict",
        script="s06_report.py",
        argv=["--results-dir", str(results), "--out", str(report),
              "--cache-dir", str(cache), "--cohorts-dir", str(cohorts_dir)],
        outputs=[report / "RESULTS.md", report / "verdict.json"],
        always_run=True,
    ))

    return stages


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------

class _StructuredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created))
        stage = getattr(record, "stage", "-")
        event = getattr(record, "event", "-")
        extra = getattr(record, "fields", "")
        base = f"{ts} | {record.levelname:<7} | stage={stage:<14} | event={event:<10} | {record.getMessage()}"
        return f"{base} | {extra}" if extra else base


def setup_logging(log_path: Path, verbose: bool) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(_StructuredFormatter())
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.DEBUG if verbose else logging.INFO)
    sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s",
                                      datefmt="%H:%M:%S"))
    logger.addHandler(sh)


def log(level: int, stage: str, event: str, msg: str, **fields) -> None:
    extra = {"stage": stage, "event": event,
             "fields": " ".join(f"{k}={v}" for k, v in fields.items())}
    logger.log(level, msg, extra=extra)


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------

def run_stage(stage: Stage, python: str, log_dir: Path) -> Dict:
    """Run one stage, tee-ing its output to console and to its own log file."""
    cmd = stage.command(python)
    log_dir.mkdir(parents=True, exist_ok=True)
    stage_log = log_dir / f"{stage.name}.log"

    log(logging.INFO, stage.name, "start", stage.description,
        cmd=shlex.join(cmd), log=str(stage_log))

    t0 = time.time()
    tail: List[str] = []
    with open(stage_log, "w") as lf:
        lf.write(f"# {shlex.join(cmd)}\n# started {time.strftime('%Y-%m-%dT%H:%M:%S')}\n\n")
        lf.flush()
        proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            lf.write(line)
            sys.stdout.write(f"    [{stage.name}] {line}")
            tail.append(line.rstrip("\n"))
            if len(tail) > 40:
                tail.pop(0)
        rc = proc.wait()
    elapsed = time.time() - t0

    missing = stage.missing_outputs()
    ok = (rc == 0) and not missing
    if ok:
        log(logging.INFO, stage.name, "done", "completed",
            seconds=f"{elapsed:.1f}", rc=rc)
    else:
        log(logging.ERROR, stage.name, "failed",
            f"return code {rc}" + (f", missing outputs: {[str(p) for p in missing]}"
                                   if missing else ""),
            seconds=f"{elapsed:.1f}", rc=rc, log=str(stage_log))
    return {"status": "ok" if ok else "failed", "returncode": rc,
            "seconds": elapsed, "command": cmd, "log": str(stage_log),
            "missing_outputs": [str(p) for p in missing], "tail": tail[-15:]}


def decide(stage: Stage, args) -> tuple[str, str]:
    """
    Resume logic. Returns (action, reason) with action in
    {"run", "skip-done", "skip-missing-script", "skip-in-progress", "skip-filtered"}.
    """
    if not stage.enabled:
        return "skip-filtered", "not selected by --stages/--from/--until"
    if not stage.script_path.exists():
        if stage.optional:
            return "skip-missing-script", f"{stage.script} has not been written yet"
        return "missing-script", f"{stage.script} does not exist"
    forced = args.force or (stage.name in args.force_stage)
    if forced:
        return "run", "forced"
    if stage.always_run:
        return "run", "always re-run (depends on every upstream output)"
    if stage.is_done():
        touched = stage.recently_written()
        if touched:
            return ("skip-done",
                    f"all {stage.n_expected()} outputs present but "
                    f"{touched[0].name} was written in the last "
                    f"{RECENT_WRITE_SECONDS}s -- a build may still be in "
                    f"progress and downstream stages may see a partial cache")
        return "skip-done", f"all {stage.n_expected()} outputs already on disk"
    touched = stage.recently_written()
    if touched:
        return ("skip-in-progress",
                f"outputs modified in the last {RECENT_WRITE_SECONDS}s "
                f"({', '.join(p.name for p in touched)}); another process is "
                f"probably still writing. Use --force to override.")
    n_missing = len(stage.missing_outputs())
    return "run", f"{n_missing}/{stage.n_expected()} outputs missing"


def apply_selection(stages: List[Stage], args) -> None:
    names = [s.name for s in stages]
    selected = set(names)
    if args.stages:
        want = {w.strip() for w in args.stages.split(",") if w.strip()}
        selected = {n for n in names if n in want or any(n.startswith(w) for w in want)}
        unknown = {w for w in want
                   if not any(n == w or n.startswith(w) for n in names)}
        if unknown:
            raise SystemExit(f"unknown stage(s): {sorted(unknown)}; known: {names}")
    if args.from_stage:
        try:
            i = next(k for k, n in enumerate(names) if n.startswith(args.from_stage))
        except StopIteration:
            raise SystemExit(f"unknown --from stage {args.from_stage!r}; known: {names}")
        selected &= set(names[i:])
    if args.until_stage:
        try:
            j = max(k for k, n in enumerate(names) if n.startswith(args.until_stage))
        except ValueError:
            raise SystemExit(f"unknown --until stage {args.until_stage!r}; known: {names}")
        selected &= set(names[:j + 1])
    for s in stages:
        s.enabled = s.name in selected


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def print_plan(stages: List[Stage], args, python: str) -> None:
    print()
    print("=" * 100)
    print(f"PLAN  ({'QUICK' if args.quick else 'FULL'} mode)   "
          f"python={python}   out={OUT_ROOT}")
    print("=" * 100)
    print(f"{'stage':<20}{'action':<22}{'reason'}")
    print("-" * 100)
    for s in stages:
        action, reason = decide(s, args)
        print(f"{s.name:<20}{action:<22}{reason[:58]}")
    print("-" * 100)
    print("commands that would run:")
    for s in stages:
        action, _ = decide(s, args)
        if action == "run":
            print(f"  {shlex.join(s.command(python))}")
    print("=" * 100)
    print()


def final_summary(records: Dict[str, dict], stages: List[Stage]) -> None:
    print()
    print("=" * 96)
    print("RUN SUMMARY")
    print("=" * 96)
    print(f"{'stage':<20}{'status':<24}{'seconds':>10}  detail")
    print("-" * 96)
    total = 0.0
    for s in stages:
        r = records.get(s.name)
        if r is None:
            continue
        secs = r.get("seconds", 0.0) or 0.0
        total += secs
        detail = r.get("reason", "")
        print(f"{s.name:<20}{r['status']:<24}{secs:>10.1f}  {detail[:40]}")
    print("-" * 96)
    print(f"{'TOTAL':<20}{'':<24}{total:>10.1f}")
    print("=" * 96)

    verdict_path = OUT_ROOT / "report" / "verdict.json"
    results_md = OUT_ROOT / "report" / "RESULTS.md"
    s06_ran = records.get("s06", {}).get("status") == "OK"
    if verdict_path.exists():
        try:
            v = json.loads(verdict_path.read_text())
            print()
            print("VERDICT" if s06_ran else
                  "VERDICT  ** STALE: stage 6 did not run in this invocation; this is "
                  "the previous report **")
            print("-" * 96)
            for cohort, blk in (v.get("cohorts") or {}).items():
                print(f"  {cohort:<16} {blk.get('verdict')}")
                print(f"  {'':<16} {blk.get('reason')}")
            if not v.get("cohorts"):
                print("  no cohort could be evaluated")
            for d in v.get("degraded", []):
                print(f"  NOTE: {d}")
        except Exception as exc:  # noqa: BLE001
            print(f"  (could not read {verdict_path}: {exc})")
    print()
    if results_md.exists():
        print(f"  READ THIS NEXT:  {results_md}")
        print(f"  figures:         {OUT_ROOT / 'report' / 'figures'}")
    else:
        print(f"  RESULTS.md was not produced; see {OUT_ROOT / 'run.log'}")
    print(f"  full log:        {OUT_ROOT / 'run.log'}")
    print(f"  per-stage logs:  {OUT_ROOT / 'logs'}")
    print(f"  machine state:   {OUT_ROOT / 'run_state.json'}")
    print("=" * 92)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="PhaseDx: run the whole pipeline, resumably",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="A stage is skipped when its outputs already exist. Use --force to "
               "re-run everything, --force-stage NAME to re-run one, or --dry-run "
               "to see the plan without executing anything.",
    )
    p.add_argument("--cohorts", default=",".join(COHORTS),
                   help="comma-separated subset of " + ",".join(COHORTS))
    p.add_argument("--conditions", default=",".join(CONDITIONS),
                   help="comma-separated subset of " + ",".join(CONDITIONS))
    p.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS),
                   help="comma-separated training seeds")
    p.add_argument("--epochs", type=int, default=None,
                   help="override stage-3 epochs (default: s03_train.py's own)")

    p.add_argument("--quick", action="store_true",
                   help="smoke mode: few files, few epochs, one seed. Produces "
                        "wiring confidence, not results")
    p.add_argument("--quick-files", type=int, default=4,
                   help="--limit passed to stage 2 in --quick mode")
    p.add_argument("--quick-epochs", type=int, default=2,
                   help="stage-3 epochs in --quick mode")

    p.add_argument("--force", action="store_true", help="re-run every stage")
    p.add_argument("--force-stage", default="", help="comma-separated stage names to re-run")
    p.add_argument("--stages", default="", help="only run these stages (prefix match)")
    p.add_argument("--from", dest="from_stage", default="", help="start at this stage")
    p.add_argument("--until", dest="until_stage", default="", help="stop after this stage")
    p.add_argument("--background-control", action="store_true",
                   help="also run stage 3 with --region background (the falsification "
                        "control stage 3 can produce on its own, until s05 exists)")

    p.add_argument("--dry-run", action="store_true",
                   help="print the plan and exit without running anything")
    p.add_argument("--keep-going", action="store_true",
                   help="continue after a required stage fails")
    p.add_argument("--python", default=sys.executable, help="interpreter for the stages")
    p.add_argument("--out-root", default=None,
                   help="override pipeline_out (used to rehearse a run, or to keep "
                        "an experiment out of the main tree)")
    p.add_argument("--log", default=None,
                   help="run log path (default: <out-root>/run.log)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    if args.out_root:
        # Rebind the module-level root before any stage is built, so every path
        # in the plan, the logs and the state file agree.
        global OUT_ROOT
        OUT_ROOT = Path(args.out_root).resolve()
    if not args.log:
        args.log = str(OUT_ROOT / "run.log")

    args.cohorts = [c.strip() for c in args.cohorts.split(",") if c.strip()]
    args.conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    args.seeds = [int(s) for s in str(args.seeds).split(",") if str(s).strip()]
    args.force_stage = {s.strip() for s in args.force_stage.split(",") if s.strip()}
    if args.quick:
        args.seeds = args.seeds[:1]
    bad = [c for c in args.cohorts if c not in COHORTS]
    if bad:
        raise SystemExit(f"unknown cohort(s) {bad}; known: {list(COHORTS)}")
    bad = [c for c in args.conditions if c not in CONDITIONS]
    if bad:
        raise SystemExit(f"unknown condition(s) {bad}; known: {list(CONDITIONS)}")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    stages = build_stages(args)
    apply_selection(stages, args)

    if args.dry_run:
        print_plan(stages, args, args.python)
        return 0

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    setup_logging(Path(args.log), args.verbose)
    t_start = time.time()
    log(logging.INFO, "-", "run-start",
        f"PhaseDx pipeline, {'quick' if args.quick else 'full'} mode",
        cohorts=",".join(args.cohorts), conditions=",".join(args.conditions),
        seeds=",".join(str(s) for s in args.seeds), python=args.python)
    print_plan(stages, args, args.python)

    records: Dict[str, dict] = {}
    failed_required = False

    for stage in stages:
        action, reason = decide(stage, args)

        if action == "skip-filtered":
            continue
        if action == "missing-script":
            log(logging.ERROR, stage.name, "missing", reason)
            records[stage.name] = {"status": "MISSING SCRIPT", "seconds": 0.0,
                                   "reason": reason}
            failed_required = True
            if not args.keep_going:
                break
            continue
        if action == "skip-missing-script":
            log(logging.WARNING, stage.name, "skip", reason + " -- downstream stages "
                "will report the resulting gap rather than assume it away")
            records[stage.name] = {"status": "SKIPPED (no script)", "seconds": 0.0,
                                   "reason": reason}
            continue
        if action == "skip-done":
            log(logging.INFO, stage.name, "skip", reason, outputs=len(stage.outputs))
            records[stage.name] = {"status": "SKIPPED (done)", "seconds": 0.0,
                                   "reason": reason}
            continue
        if action == "skip-in-progress":
            log(logging.WARNING, stage.name, "skip", reason)
            records[stage.name] = {"status": "SKIPPED (in progress)", "seconds": 0.0,
                                   "reason": reason}
            continue

        rec = run_stage(stage, args.python, OUT_ROOT / "logs")
        rec["reason"] = reason
        rec["status"] = "OK" if rec["status"] == "ok" else "FAILED"
        records[stage.name] = rec
        _write_state(records, args, t_start)

        if rec["status"] == "FAILED":
            if stage.optional:
                log(logging.WARNING, stage.name, "failed",
                    "optional stage failed; continuing")
                continue
            failed_required = True
            print()
            print(f"  last lines of {rec['log']}:")
            for line in rec.get("tail", []):
                print(f"    | {line}")
            if not args.keep_going:
                log(logging.ERROR, stage.name, "abort",
                    "required stage failed; stopping (use --keep-going to continue)")
                break

    _write_state(records, args, t_start)
    log(logging.INFO, "-", "run-end", "pipeline finished",
        seconds=f"{time.time() - t_start:.1f}",
        failed=str(failed_required))
    final_summary(records, stages)
    return 1 if failed_required else 0


def _write_state(records: Dict[str, dict], args, t_start: float) -> None:
    state = {
        "started": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t_start)),
        "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_seconds": time.time() - t_start,
        "mode": "quick" if args.quick else "full",
        "cohorts": args.cohorts, "conditions": args.conditions, "seeds": args.seeds,
        "stages": records,
    }
    (OUT_ROOT / "run_state.json").write_text(json.dumps(state, indent=2, default=str))


if __name__ == "__main__":
    raise SystemExit(main())
