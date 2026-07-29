"""
s13_expand.py
-------------
Cohort expansion: what is on the drive, what would fit, how to get it in, and
what adding it destroys.

The clinical cohorts are small -- 45 usable prostate DWI subjects, 67 prostate
T2, 70 breast -- and the smallest of them is the one Rempe et al. 2024 used, so
it is the one a reproduction needs. Expanding it is the top data priority. This
module makes that a single command and, more importantly, makes it safe.

Four things go wrong when a cohort is expanded by hand, and each has a section
below:

1. YOU DOWNLOAD THE WRONG TARS. The release ships in fixed ID chunks; the drive
   holds a subset; the gap is not obvious from `ls` because the chunk
   directories are named by ID RANGE while the files inside are named by ID, and
   several ranges are incomplete (one AXDIFF file absent from every DIFF chunk
   we hold, nine AXT2 files absent across the four T2 chunks). `inventory`
   prints the exact absent IDs and the exact tar names.

2. YOU RUN OUT OF DISK MID-EXTRACTION. A prostate DWI patient is 8.4 GB measured
   and a chunk is ~93 GB, and extraction needs the tar AND its contents resident
   at the same time. `budget` measures per-patient cost from the files already
   present, reads real free space, models the transient tar-plus-extract peak,
   and refuses to recommend anything that would take the drive below a safety
   margin. It also reports the 818 GB of already-extracted tar files sitting on
   the drive, because that is the largest single lever available and it costs
   nothing but an `rm`.

3. YOU EXTRACT GARBAGE. macOS writes AppleDouble sidecars (`._name.h5`) onto
   exFAT; the drive already carries 1425 of them and they are not HDF5. Worse,
   nine prostate files that ARE on the drive at full size fail to open
   ("free block size is zero?") -- interrupted copies that stayed silent until
   stage 1 probed them, costing 5 DWI and 4 T2 subjects. `extract` filters the
   sidecars and opens every extracted file before declaring it good, and reports
   the corrupt ones instead of aborting the batch.

4. YOU APPEND RESULTS ACROSS A FOLD CHANGE. This is the one that would actually
   corrupt the paper. The 5-fold CV split is derived at stage 1 from the set of
   subjects present. Add a patient and every fold membership moves, so the
   existing out-of-fold predictions are predictions over a DIFFERENT partition.
   They cannot be pooled with new ones and they cannot be topped up: the whole
   102-run headline tree has to be re-run. `recache` says so in the loudest
   terms available in a terminal and prints the exact commands.

Nothing here downloads anything. Downloads require the NYU fastMRI data-use
agreement links, which are the user's and only the user's.

Usage:
    python pipeline/s13_expand.py --dry-run             # sections 1 + 2, no writes
    python pipeline/s13_expand.py inventory
    python pipeline/s13_expand.py budget --margin-gb 100 [--reclaim-tars]
    python pipeline/s13_expand.py extract --staging /Volumes/Research/staging
    python pipeline/s13_expand.py recache --cohort prostate_dwi
    python pipeline/s13_expand.py --self-test
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tarfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import h5py

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import common  # noqa: E402

GB = 1e9
GiB = float(2 ** 30)

# Below this much free space the drive is not safe to keep filling: the OS, the
# pipeline_out cache and any in-flight extraction all need room, and exFAT gives
# no warning before it wedges.
DEFAULT_MARGIN_GB = 100.0

# How many downloaded tars sit in staging at once. Extraction needs the tar and
# its expanded contents resident simultaneously, so the transient peak is
# (extracted so far) + (concurrency x tar size). One at a time is the cheapest
# and the default; raise it only if the download is genuinely parallel.
DEFAULT_STAGING_CONCURRENCY = 1


# ---------------------------------------------------------------------------
# The releases
# ---------------------------------------------------------------------------
#
# `stride` and `dir_prefix` are how NYU chunks the release into tars. Both are
# VERIFIED against the chunk directories already on the drive (see
# verify_chunk_grid); they are not assumed. What cannot be verified from the
# drive is the name of the final, short chunk -- 312 prostate patients do not
# divide evenly by 11 or by 20 -- so chunk names beyond the last one present are
# printed with an explicit "inferred" marker.

@dataclass(frozen=True)
class Release:
    cohort: str
    organ: str
    dir_prefix: str          # 'fastMRI_prostate_DIFF_IDS_'
    n_patients: int          # size of the full public release
    stride: int              # patient IDs per tar
    acq_per_patient: int     # h5 files per patient
    required: tuple          # datasets every good file must carry
    roots: tuple             # trees stage 1 scans, in DATA_ROOT-relative form
    dest: str                # DATA_ROOT-relative tree new chunks extract into
    files_for: Callable      # patient id -> [expected h5 basenames]
    id_of: Callable          # h5 basename -> patient id, or None
    priority: int            # download order across organs (lower goes first)

    @property
    def root_paths(self) -> list:
        return [common.DATA_ROOT / r for r in self.roots]

    @property
    def dest_path(self) -> Path:
        return common.DATA_ROOT / self.dest


def _prostate_id(tag: str) -> Callable:
    rx = re.compile(rf"^file_prostate_{tag}_(\d+)\.h5$")

    def f(name: str):
        m = rx.match(name)
        return int(m.group(1)) if m else None
    return f


_BREAST_RX = re.compile(r"^fastMRI_breast_(\d+)_(\d+)\.h5$")


def _breast_id(name: str):
    m = _BREAST_RX.match(name)
    return int(m.group(1)) if m else None


RELEASES = {
    # Diffusion first: weakest cohort (45 usable subjects) and the one Rempe et
    # al. 2024 used, so it is what a reproduction needs.
    "prostate_dwi": Release(
        cohort="prostate_dwi", organ="prostate",
        dir_prefix="fastMRI_prostate_DIFF_IDS_",
        n_patients=312, stride=11, acq_per_patient=1,
        # coil_sens_maps is required, not optional: s02_prostate does a real
        # GRAPPA reconstruction and then a SENSE-style coil combination, and a
        # DWI file without sensitivity maps cannot be phase-combined at all.
        required=("kspace", "coil_sens_maps", "calibration_data", "ismrmrd_header"),
        roots=("prostate",), dest="prostate",
        files_for=lambda i: [f"file_prostate_AXDIFF_{i:03d}.h5"],
        id_of=_prostate_id("AXDIFF"), priority=0,
    ),
    "prostate_t2": Release(
        cohort="prostate_t2", organ="prostate",
        dir_prefix="fastMRI_prostate_T2_IDS_",
        n_patients=312, stride=20, acq_per_patient=1,
        required=("kspace", "calibration_data", "ismrmrd_header", "reconstruction_rss"),
        roots=("prostate",), dest="prostate",
        files_for=lambda i: [f"file_prostate_AXT2_{i:03d}.h5"],
        id_of=_prostate_id("AXT2"), priority=1,
    ),
    "breast": Release(
        cohort="breast", organ="breast",
        dir_prefix="fastMRI_breast_IDS_",
        n_patients=300, stride=10, acq_per_patient=2,
        required=("kspace", "temptv"),
        # Stage 1 scans BOTH breast trees and asserts on duplicate
        # (patient_id, acq) pairs, so a chunk extracted into the wrong one is
        # not a cosmetic mistake -- it hard-fails s01.
        roots=("breast_updated", "breast"), dest="breast_updated/breast",
        # Zero-padded to three digits: the release ships fastMRI_breast_001_1.h5,
        # not fastMRI_breast_1_1.h5, and stage 1 derives patient_id straight from
        # that digit string, so the padding is load-bearing.
        files_for=lambda i: [f"fastMRI_breast_{i:03d}_1.h5", f"fastMRI_breast_{i:03d}_2.h5"],
        id_of=_breast_id, priority=2,
    ),
}

COHORT_ORDER = sorted(RELEASES, key=lambda c: RELEASES[c].priority)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def fmt_ranges(ids: Iterable[int]) -> str:
    """[1,2,3,7,9,10] -> '001-003, 007, 009-010'. Empty -> '(none)'."""
    ids = sorted(set(int(i) for i in ids))
    if not ids:
        return "(none)"
    out, start, prev = [], ids[0], ids[0]
    for i in ids[1:]:
        if i == prev + 1:
            prev = i
            continue
        out.append(f"{start:03d}" if start == prev else f"{start:03d}-{prev:03d}")
        start = prev = i
    out.append(f"{start:03d}" if start == prev else f"{start:03d}-{prev:03d}")
    return ", ".join(out)


def chunk_grid(rel: Release) -> list:
    """The (first_id, last_id) span of every tar in the release, in order."""
    spans = []
    a = 1
    while a <= rel.n_patients:
        spans.append((a, min(a + rel.stride - 1, rel.n_patients)))
        a += rel.stride
    return spans


def chunk_dirname(rel: Release, a: int, b: int) -> str:
    return f"{rel.dir_prefix}{a:03d}_{b:03d}"


def disk_free(path: Path) -> tuple:
    total, used, free = shutil.disk_usage(path)
    return total, used, free


def probe_h5(path: Path, required: tuple) -> dict:
    """
    Open one file and decide whether it is usable.

    Deliberately stricter than "h5py.File did not raise": nine files on this
    drive are full-size and openable-looking yet fail on first link lookup, so
    the check reads the dataset list AND one element out of kspace. A truncated
    copy that keeps a valid superblock dies on the read, not on the open.
    """
    rec = {"path": str(path), "ok": False, "error": "", "bytes": 0, "kspace_shape": ""}
    try:
        rec["bytes"] = path.stat().st_size
    except OSError as e:
        rec["error"] = f"stat failed: {e}"
        return rec
    try:
        with h5py.File(path, "r") as f:
            keys = set(f.keys())
            missing = [k for k in required if k not in keys]
            if missing:
                rec["error"] = f"missing datasets: {','.join(missing)}"
                return rec
            ks = f["kspace"]
            rec["kspace_shape"] = str(tuple(ks.shape))
            if ks.ndim < 3:
                rec["error"] = f"kspace has {ks.ndim} dims, expected >= 3"
                return rec
            _ = ks[tuple(0 for _ in ks.shape)]      # forces a real chunk read
    except Exception as e:  # noqa: BLE001 -- any failure here is data loss
        rec["error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"
        return rec
    rec["ok"] = True
    return rec


# ---------------------------------------------------------------------------
# 1. INVENTORY
# ---------------------------------------------------------------------------

@dataclass
class Inventory:
    rel: Release
    present_ids: set = field(default_factory=set)      # >=1 file on disk
    complete_ids: set = field(default_factory=set)     # all acquisitions, all open
    corrupt: list = field(default_factory=list)        # probe records, ok=False
    partial_ids: set = field(default_factory=set)      # some but not all acqs
    files: dict = field(default_factory=dict)          # pid -> [Path]
    chunks_present: list = field(default_factory=list)  # dirnames on disk
    grid_ok: bool = True
    grid_note: str = ""
    sizes: list = field(default_factory=list)          # bytes, per file
    probed: bool = False
    stray: list = field(default_factory=list)          # h5 that parse but sit outside the grid

    @property
    def missing_ids(self) -> list:
        return sorted(set(range(1, self.rel.n_patients + 1)) - self.present_ids)

    @property
    def usable_ids(self) -> set:
        return self.complete_ids


def verify_chunk_grid(rel: Release, dirnames: Iterable[str]) -> tuple:
    """
    Check that the chunk directories on disk lie on the stride grid we assume.

    If they do not, the tar names this module prints would be wrong, which is
    the one failure mode that wastes the user's bandwidth rather than their
    time. Return (ok, note).
    """
    grid = {chunk_dirname(rel, a, b) for a, b in chunk_grid(rel)}
    off = sorted(d for d in dirnames if d not in grid)
    if off:
        return False, (f"{len(off)} chunk dir(s) do not lie on the assumed "
                       f"stride-{rel.stride} grid: {off[:4]}")
    return True, ""


def take_inventory(cohort: str, probe: bool = True, verbose: bool = True) -> Inventory:
    rel = RELEASES[cohort]
    inv = Inventory(rel=rel, probed=probe)

    seen = set()
    for root in rel.root_paths:
        if not root.exists():
            if verbose:
                print(f"    NOTE: {root} not present, skipping")
            continue
        for p in common.iter_h5(root):
            pid = rel.id_of(p.name)
            if pid is None:
                continue
            if p in seen:
                continue
            seen.add(p)
            inv.files.setdefault(pid, []).append(p)
            inv.sizes.append(p.stat().st_size)

    inv.present_ids = set(inv.files)
    for pid in sorted(inv.present_ids):
        if pid < 1 or pid > rel.n_patients:
            inv.stray.append(pid)

    dirs = sorted({p.parent.name for ps in inv.files.values() for p in ps})
    inv.chunks_present = dirs
    inv.grid_ok, inv.grid_note = verify_chunk_grid(rel, dirs)

    if probe:
        n = sum(len(v) for v in inv.files.values())
        done = 0
        for pid in sorted(inv.files):
            good = 0
            for p in inv.files[pid]:
                rec = probe_h5(p, rel.required)
                done += 1
                if verbose and (done % 50 == 0 or done == n):
                    print(f"    probed {done}/{n} {rel.cohort} files")
                if rec["ok"]:
                    good += 1
                else:
                    inv.corrupt.append(rec)
            if good == rel.acq_per_patient:
                inv.complete_ids.add(pid)
            elif good > 0 or len(inv.files[pid]) < rel.acq_per_patient:
                inv.partial_ids.add(pid)
    else:
        for pid, ps in inv.files.items():
            (inv.complete_ids if len(ps) >= rel.acq_per_patient
             else inv.partial_ids).add(pid)
    return inv


def chunk_status(rel: Release, inv: Inventory) -> list:
    """One row per tar in the release: what it would give us."""
    rows = []
    for a, b in chunk_grid(rel):
        span = list(range(a, b + 1))
        on_disk = [i for i in span if i in inv.present_ids]
        usable = [i for i in span if i in inv.usable_ids]
        broken = [i for i in span if i in inv.present_ids and i not in inv.usable_ids]
        absent = [i for i in span if i not in inv.present_ids]
        dirname = chunk_dirname(rel, a, b)
        rows.append({
            "a": a, "b": b, "dir": dirname, "tar": dirname + ".tar",
            "span": len(span), "on_disk": len(on_disk), "usable": len(usable),
            "broken_ids": broken, "absent_ids": absent,
            "held": dirname in inv.chunks_present,
            # Inferred names are the ones we have never seen on disk. Only the
            # final short chunk is genuinely uncertain, but the honest marker is
            # "we have not observed this name", not "we think it is fine".
            "name_verified": dirname in inv.chunks_present,
            "new_patients": len(absent) if dirname not in inv.chunks_present else 0,
            "recoverable": len(broken) + (len(absent) if dirname in inv.chunks_present else 0),
        })
    return rows


def print_inventory(invs: dict) -> None:
    print("=" * 78)
    print("1. INVENTORY -- what is on the drive vs what the release contains")
    print("=" * 78)
    print(f"data root: {common.DATA_ROOT}")
    for cohort in [c for c in COHORT_ORDER if c in invs]:
        inv = invs[cohort]
        rel = inv.rel
        rows = chunk_status(rel, inv)
        held = [r for r in rows if r["held"]]
        print()
        print(f"-- {cohort}  ({rel.organ}; release = {rel.n_patients} patients, "
              f"{rel.acq_per_patient} acq/patient, {rel.stride} patients/tar, "
              f"{len(rows)} tars)")
        if not inv.grid_ok:
            print(f"   !! CHUNK GRID MISMATCH: {inv.grid_note}")
            print("      Tar names below are unreliable; check the NYU manifest by hand.")
        if inv.stray:
            print(f"   !! patient ids outside 1..{rel.n_patients} on disk: {inv.stray}")
        print(f"   patients on disk        : {len(inv.present_ids):>4} / {rel.n_patients}")
        if inv.probed:
            print(f"   patients usable         : {len(inv.usable_ids):>4}   "
                  f"(this is the cohort n the paper reports)")
            print(f"   patients lost to corrupt: {len(inv.present_ids) - len(inv.usable_ids):>4}"
                  f"   ({len(inv.corrupt)} unreadable file(s))")
        else:
            print("   patients usable         :  not probed (--no-probe)")
        print(f"   tars held               : {len(held):>4} / {len(rows)}")

        print(f"   MISSING patient IDs     : {fmt_ranges(inv.missing_ids)}")

        incomplete = [r for r in held if r["absent_ids"] or r["broken_ids"]]
        if incomplete:
            print("   tars we hold that are INCOMPLETE:")
            for r in incomplete:
                bits = []
                if r["absent_ids"]:
                    bits.append(f"absent {fmt_ranges(r['absent_ids'])}")
                if r["broken_ids"]:
                    bits.append(f"corrupt {fmt_ranges(r['broken_ids'])}")
                usable = (f", {r['usable']}/{r['span']} usable" if inv.probed
                          else " (usability not checked)")
                print(f"      {r['tar']:<44} {r['on_disk']}/{r['span']} on disk"
                      f"{usable}  ({'; '.join(bits)})")

        absent_tars = [r for r in rows if not r["held"]]
        print(f"   tars NOT on the drive   : {len(absent_tars)}"
              f"  ({sum(r['span'] for r in absent_tars)} patients)")
        print(f"      (names below are generated from the stride-{rel.stride} grid that the "
              f"{len(held)} chunk")
        print("       directories on disk confirm; only the short final chunk is a guess)")
        for r in absent_tars:
            mark = "   [SHORT FINAL CHUNK: NYU may name or split this one differently]" \
                if r["span"] != rel.stride else ""
            print(f"      {r['tar']:<44} ids {r['a']:03d}-{r['b']:03d} "
                  f"({r['span']} patients){mark}")

    print()
    print("   Note on 'absent' inside a tar we hold: the file is listed in the")
    print("   label CSV but is not on disk. Either the release omits that series")
    print("   for that patient or our extraction lost it -- re-extracting the tar")
    print("   is the only way to tell. Do not assume it is free.")


# ---------------------------------------------------------------------------
# 2. BUDGET
# ---------------------------------------------------------------------------

def measure_cost(inv: Inventory) -> dict:
    """
    Per-patient bytes, measured from the files actually on this drive.

    Mean rather than median: a budget that is right on average and wrong on the
    tails is what the safety margin is for, whereas a median under-books a
    cohort whose size distribution is right-skewed (DWI runs 6.1-11.5 GB).
    """
    rel = inv.rel
    n = len(inv.sizes)
    if n == 0:
        return {"n_files": 0, "per_patient": 0.0, "per_tar": 0.0,
                "min": 0.0, "max": 0.0, "measured": False}
    total = float(sum(inv.sizes))
    per_file = total / n
    per_patient = per_file * rel.acq_per_patient
    return {
        "n_files": n, "per_file": per_file, "per_patient": per_patient,
        "per_tar": per_patient * rel.stride,
        "min": float(min(inv.sizes)), "max": float(max(inv.sizes)),
        "measured": True,
    }


def find_reclaimable_tars(invs: dict) -> list:
    """
    Tars still sitting on the drive whose contents are already extracted.

    A tar is only reported reclaimable when EVERY patient in its span is present
    on disk and (if probed) usable. Deleting a tar whose extraction was partial
    or corrupt would destroy the only copy, so the bar is deliberately high.
    """
    out = []
    for cohort in COHORT_ORDER:
        rel = RELEASES[cohort]
        inv = invs[cohort]
        for root in {p for p in rel.root_paths} | {rel.dest_path, rel.dest_path.parent}:
            if not root.exists():
                continue
            for tar in sorted(root.glob("*.tar")):
                if tar.name.startswith("._") or not tar.name.startswith(rel.dir_prefix):
                    continue
                m = re.match(rf"{re.escape(rel.dir_prefix)}(\d+)_(\d+)\.tar$", tar.name)
                if not m:
                    continue
                a, b = int(m.group(1)), int(m.group(2))
                span = set(range(a, b + 1))
                if not inv.probed:
                    # Deleting the only copy of 90 GB on the strength of an
                    # unverified `ls` is not a trade worth offering.
                    out.append({"tar": tar, "bytes": tar.stat().st_size, "safe": False,
                                "why": "run without --no-probe before deleting anything"})
                    continue
                pool = inv.usable_ids
                if span - pool:
                    out.append({"tar": tar, "bytes": tar.stat().st_size,
                                "safe": False,
                                "why": f"{len(span - pool)} of {len(span)} patients "
                                       f"not verified on disk"})
                else:
                    out.append({"tar": tar, "bytes": tar.stat().st_size,
                                "safe": True, "why": "all patients extracted and verified"})
    return out


def build_plan(invs: dict, free_bytes: float, margin_gb: float,
               concurrency: int, extra_bytes: float = 0.0) -> dict:
    """
    Greedy prioritised download list. Organ order is fixed by Release.priority
    (DWI, then T2, then breast); within an organ, ascending patient ID.

    The constraint is not "sum of tars <= free". Extraction needs the tar and
    its expanded contents on the drive at the same time, so the binding
    constraint at every step is

        already_extracted + concurrency * this_tar <= free + reclaimed - margin

    which is what stops the plan one tar earlier than a naive sum would.
    """
    budget = free_bytes + extra_bytes - margin_gb * GB
    costs = {c: measure_cost(invs[c]) for c in COHORT_ORDER}

    plan, skipped, running, unmeasurable = [], [], 0.0, []
    stop_reason = None
    for cohort in COHORT_ORDER:
        rel, inv = RELEASES[cohort], invs[cohort]
        cost = costs[cohort]
        if not cost["measured"]:
            # No files of this cohort on the drive, so there is no measured
            # per-patient size. Guessing one would put free tars in the plan;
            # refusing to plan is the only honest option.
            unmeasurable.append(cohort)
            continue
        for r in chunk_status(rel, inv):
            if r["held"]:
                continue                     # a tar we already hold is a REPAIR, not an expansion
            tar_bytes = cost["per_patient"] * r["span"]
            peak = running + concurrency * tar_bytes
            if peak > budget:
                if stop_reason is None:
                    stop_reason = (
                        f"{r['tar']} ({tar_bytes / GB:.1f} GB) would push the peak to "
                        f"{(peak + margin_gb * GB) / GB:.1f} GB against "
                        f"{(free_bytes + extra_bytes) / GB:.1f} GB available, "
                        f"leaving less than the {margin_gb:.0f} GB margin")
                skipped.append({"cohort": cohort, **r, "bytes": tar_bytes})
                continue
            running += tar_bytes
            plan.append({"cohort": cohort, **r, "bytes": tar_bytes,
                         "running": running,
                         "free_after": free_bytes + extra_bytes - running})
    return {"plan": plan, "skipped": skipped, "budget": budget, "costs": costs,
            "running": running, "stop_reason": stop_reason,
            "unmeasurable": unmeasurable,
            "margin": margin_gb * GB, "concurrency": concurrency,
            "free": free_bytes, "extra": extra_bytes}


def print_budget(invs: dict, margin_gb: float, concurrency: int,
                 use_reclaim: bool) -> dict:
    total, used, free = disk_free(common.DATA_ROOT)
    reclaim = find_reclaimable_tars(invs)
    safe_reclaim = sum(r["bytes"] for r in reclaim if r["safe"])

    print()
    print("=" * 78)
    print("2. BUDGET -- what actually fits")
    print("=" * 78)
    vol = common.DATA_ROOT
    print(f"   volume                     : {vol}")
    print(f"   capacity                   : {total / GB:>9.1f} GB  ({total / GiB:.1f} GiB)")
    print(f"   used                       : {used / GB:>9.1f} GB")
    print(f"   free NOW                   : {free / GB:>9.1f} GB  ({free / GiB:.1f} GiB)")
    print(f"   safety margin              : {margin_gb:>9.1f} GB  (--margin-gb)")
    print(f"   staging concurrency        : {concurrency:>9d}    tar(s) resident during extraction")

    print()
    print("   measured per-patient cost (from the files already on this drive)")
    print(f"   {'cohort':<14}{'files':>7}{'GB/file':>10}{'GB/patient':>12}"
          f"{'GB/tar':>10}   {'min-max GB/file'}")
    for cohort in COHORT_ORDER:
        c = measure_cost(invs[cohort])
        if not c["measured"]:
            print(f"   {cohort:<14}      -   no files on disk; cannot measure, cohort skipped")
            continue
        print(f"   {cohort:<14}{c['n_files']:>7}{c['per_file'] / GB:>10.2f}"
              f"{c['per_patient'] / GB:>12.2f}{c['per_tar'] / GB:>10.1f}"
              f"   {c['min'] / GB:.2f}-{c['max'] / GB:.2f}")

    if reclaim:
        print()
        print("   RECLAIMABLE -- tars still on the drive whose contents are already extracted")
        for r in sorted(reclaim, key=lambda x: -x["bytes"]):
            flag = "reclaimable" if r["safe"] else "KEEP"
            print(f"      {flag:<12}{r['bytes'] / GB:>8.1f} GB  {r['tar']}")
            if not r["safe"]:
                print(f"                                  {r['why']}")
        print(f"      -> {safe_reclaim / GB:.1f} GB recoverable with `rm`, no download needed."
              f"  This is the largest lever available.")

    extra = safe_reclaim if use_reclaim else 0.0
    if use_reclaim:
        print(f"\n   --reclaim-tars given: counting {safe_reclaim / GB:.1f} GB of tar deletions"
              f" as available.")
    else:
        print("\n   Counting free space ONLY (pass --reclaim-tars to spend the tar space too).")

    res = build_plan(invs, free, margin_gb, concurrency, extra)
    avail = free + extra
    print()
    print(f"   available for extracted data : {avail / GB:.1f} - {margin_gb:.1f} margin "
          f"= {res['budget'] / GB:.1f} GB, minus one tar in flight at each step")

    print()
    print("   RECOMMENDED DOWNLOAD ORDER  (prostate DWI -> prostate T2 -> breast)")
    if not res["plan"]:
        print("      NOTHING FITS. Free space is already inside the safety margin.")
    else:
        print(f"      {'#':>3}  {'tar':<44}{'pts':>5}{'GB':>9}{'running':>10}{'free after':>12}")
        for i, r in enumerate(res["plan"], 1):
            print(f"      {i:>3}  {r['tar']:<44}{r['span']:>5}{r['bytes'] / GB:>9.1f}"
                  f"{r['running'] / GB:>10.1f}{r['free_after'] / GB:>12.1f}")
        by_cohort = {}
        for r in res["plan"]:
            e = by_cohort.setdefault(r["cohort"], [0, 0, 0.0])
            e[0] += 1
            e[1] += r["span"]
            e[2] += r["bytes"]
        print()
        for cohort in COHORT_ORDER:
            if cohort not in by_cohort:
                continue
            n_tar, n_pat, b = by_cohort[cohort]
            inv = invs[cohort]
            now = len(inv.usable_ids) if inv.probed else len(inv.present_ids)
            print(f"      {cohort:<14} +{n_tar:>2} tars = +{n_pat:>3} patients "
                  f"({b / GB:>7.1f} GB)   cohort n: {now} -> up to {now + n_pat}"
                  f"   [before any of them turn out corrupt]")
        print(f"      TOTAL           {len(res['plan'])} tars, "
              f"{sum(r['span'] for r in res['plan'])} patients, "
              f"{res['running'] / GB:.1f} GB")
    if res["unmeasurable"]:
        print()
        print(f"   NOT PLANNED (no files of this cohort on the drive, so no measured "
              f"per-patient size): {', '.join(res['unmeasurable'])}")

    if res["stop_reason"]:
        print()
        print(f"   STOP: {res['stop_reason']}.")
        left = {}
        for r in res["skipped"]:
            e = left.setdefault(r["cohort"], [0, 0])
            e[0] += 1
            e[1] += r["span"]
        for cohort, (n_tar, n_pat) in left.items():
            print(f"      not affordable: {cohort:<14} {n_tar:>2} tars / {n_pat:>3} patients")
        if not use_reclaim and safe_reclaim > 0:
            alt = build_plan(invs, free, margin_gb, concurrency, safe_reclaim)
            gain = len(alt["plan"]) - len(res["plan"])
            gain_p = (sum(r["span"] for r in alt["plan"])
                      - sum(r["span"] for r in res["plan"]))
            print(f"      Deleting the {safe_reclaim / GB:.1f} GB of already-extracted tars "
                  f"would buy {gain} more tars / {gain_p} more patients (--reclaim-tars).")

    # Repairs are a separate question: they buy back subjects we have already
    # paid for, but at a terrible rate, and saying so is the point.
    print()
    print("   REPAIR OPTION -- tars we already hold that contain corrupt or absent files")
    any_repair = False
    for cohort in COHORT_ORDER:
        inv = invs[cohort]
        cost = measure_cost(inv)
        if not cost["measured"]:
            continue
        rows = [r for r in chunk_status(RELEASES[cohort], inv)
                if r["held"] and r["recoverable"]]
        if not rows:
            continue
        any_repair = True
        # The comparison that matters is against a NEW tar of the SAME cohort,
        # because that is the alternative use of the same gigabytes.
        print(f"      {cohort}  (a new {cohort} tar costs "
              f"{cost['per_patient'] / GB:.1f} GB per patient)")
        for r in rows:
            tar_b = cost["per_patient"] * r["span"]
            each = tar_b / r["recoverable"]
            print(f"        {r['tar']:<42}{tar_b / GB:>8.1f} GB  recovers "
                  f"{r['recoverable']} patient(s) = {each / GB:.1f} GB each"
                  f"   ({each / cost['per_patient']:.0f}x a new patient)")
    if not any_repair:
        print("      (none)")
    else:
        print("      Repair means re-downloading a whole chunk to rescue one or two files,")
        print("      so it buys patients at several times the price of a new tar. Do the")
        print("      new tars first; repair only if the drive still has room afterwards.")

    cache_free = disk_free(common.OUT_ROOT)[2] if common.OUT_ROOT.exists() else 0.0
    planned = sum(r["span"] for r in res["plan"])
    print()
    print(f"   Local disk (the cache, not the drive): {common.OUT_ROOT} has "
          f"{cache_free / GB:.1f} GB free.")
    print(f"   The caches are ~5 MB per prostate subject, so +{planned} patients adds "
          f"well under 10 GB. Not a constraint.")

    print()
    print("   No download happens here and none can. The NYU fastMRI data-use")
    print("   agreement links are yours; fetch the tars listed above into a staging")
    print("   directory, then run `s13_expand.py extract --staging <dir>`.")
    return res


# ---------------------------------------------------------------------------
# 3. EXTRACT
# ---------------------------------------------------------------------------

def classify_tar(name: str):
    """Map a tar filename onto (cohort, first_id, last_id), or None."""
    for cohort in COHORT_ORDER:
        rel = RELEASES[cohort]
        m = re.match(rf"{re.escape(rel.dir_prefix)}(\d+)_(\d+)\.tar$", name)
        if m:
            return cohort, int(m.group(1)), int(m.group(2))
    return None


def is_appledouble(name: str) -> bool:
    base = name.rsplit("/", 1)[-1]
    return base.startswith("._") or base == ".DS_Store"


def extract_one(tar_path: Path, cohort: str, a: int, b: int, invs: dict,
                dry: bool, delete_tar: bool) -> dict:
    """
    Extract one tar into the right release directory and verify what came out.

    Members are placed by BASENAME into <dest>/<chunk_dir>/, not by the path
    stored in the archive. That is deliberate: it makes path traversal
    impossible, it makes the result independent of whether NYU wrapped the chunk
    in an extra directory level, and it means an archive that disagrees with its
    own name still lands where stage 1 will look.
    """
    rel = RELEASES[cohort]
    inv = invs[cohort]
    dest = rel.dest_path / chunk_dirname(rel, a, b)
    rep = {"tar": str(tar_path), "cohort": cohort, "dest": str(dest),
           "written": [], "skipped": [], "good": [], "corrupt": [],
           "collisions": [], "error": ""}

    span = set(range(a, b + 1))

    # A breast chunk extracted into the second tree while its patients already
    # live in the first makes s01 raise on duplicate (patient_id, acq) pairs.
    # Catch it here, where the fix is free.
    for pid in sorted(span & inv.present_ids):
        for p in inv.files.get(pid, []):
            if p.parent != dest:
                rep["collisions"].append(str(p))
    if rep["collisions"]:
        print(f"   !! {len(rep['collisions'])} patient(s) in this span already exist "
              f"OUTSIDE {dest}:")
        for c in rep["collisions"][:6]:
            print(f"        {c}")
        print("      Extracting here would give stage 1 two copies of the same "
              "(patient, acq) and it will refuse to build the cohort.")
        print("      Remove the old copy first, or extract over it, then re-run.")
        rep["error"] = "destination collision"
        return rep

    size = tar_path.stat().st_size
    _, _, free = disk_free(common.DATA_ROOT)
    if free < size * 1.02:
        rep["error"] = (f"not enough free space: {free / GB:.1f} GB free, "
                        f"needs ~{size / GB:.1f} GB to expand")
        print(f"   !! {rep['error']}")
        return rep

    print(f"   -> {dest}")
    if dry:
        print(f"      DRY RUN: would extract {size / GB:.1f} GB, "
              f"excluding AppleDouble (._*) entries")
        return rep

    dest.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        # Stream mode: one sequential pass over the archive, which is what a
        # 90 GB tar on a spinning external drive wants.
        with tarfile.open(tar_path, "r|") as tf:
            for m in tf:
                if not m.isfile():
                    continue
                base = m.name.rsplit("/", 1)[-1]
                if is_appledouble(m.name):
                    rep["skipped"].append(m.name)
                    continue
                if not base.endswith(".h5") or rel.id_of(base) is None:
                    rep["skipped"].append(m.name)
                    continue
                src = tf.extractfile(m)
                if src is None:
                    rep["skipped"].append(m.name)
                    continue
                out = dest / base
                with open(out, "wb") as fh:
                    shutil.copyfileobj(src, fh, length=8 << 20)
                rep["written"].append(base)
                print(f"      {base:<40}{out.stat().st_size / GB:>7.2f} GB")
    except Exception as e:  # noqa: BLE001
        rep["error"] = f"{type(e).__name__}: {e}"
        print(f"   !! extraction failed: {rep['error']}")
        print("      Files already written are kept and verified below.")

    print(f"      extracted {len(rep['written'])} file(s) in {time.time() - t0:.0f}s; "
          f"skipped {len(rep['skipped'])} non-HDF5/AppleDouble entr(ies)")

    for base in rep["written"]:
        recd = probe_h5(dest / base, rel.required)
        (rep["good"] if recd["ok"] else rep["corrupt"]).append(recd)
    for recd in rep["corrupt"]:
        print(f"      CORRUPT  {Path(recd['path']).name}: {recd['error']}")
    print(f"      verified {len(rep['good'])}/{len(rep['written'])} file(s) open "
          f"and carry {list(rel.required)}")

    expected = sorted({n for pid in span for n in rel.files_for(pid)})
    got = set(rep["written"])
    absent = [n for n in expected if n not in got]
    if absent:
        print(f"      NOT IN THIS TAR ({len(absent)}): {', '.join(absent[:6])}"
              f"{' ...' if len(absent) > 6 else ''}")

    if delete_tar:
        if rep["corrupt"] or rep["error"]:
            print("      keeping the tar: verification did not come back clean")
        else:
            tar_path.unlink()
            print(f"      deleted {tar_path.name} (+{size / GB:.1f} GB free)")
    return rep


def cmd_extract(args, invs: dict) -> int:
    staging = Path(args.staging)
    print("=" * 78)
    print("3. EXTRACT")
    print("=" * 78)
    if not staging.exists():
        print(f"ERROR: staging directory does not exist: {staging}", file=sys.stderr)
        return 2

    tars = [p for p in sorted(staging.rglob("*.tar")) if not p.name.startswith("._")]
    if not tars:
        print(f"No .tar files under {staging}. Nothing to do.")
        return 0

    print(f"staging   : {staging}")
    print(f"tars found: {len(tars)}")
    known, unknown = [], []
    for p in tars:
        cls = classify_tar(p.name)
        (known if cls else unknown).append((p, cls))
    for p, _ in unknown:
        print(f"   ?? unrecognised, will not touch: {p.name}")
        print("      Expected one of: "
              + ", ".join(RELEASES[c].dir_prefix + "<a>_<b>.tar" for c in COHORT_ORDER))

    reports = []
    for p, (cohort, a, b) in known:
        print()
        print(f"-- {p.name}  ({p.stat().st_size / GB:.1f} GB -> {cohort}, ids {a:03d}-{b:03d})")
        reports.append(extract_one(p, cohort, a, b, invs, args.dry_run, args.delete_tar))

    print()
    print("-" * 78)
    print("EXTRACTION SUMMARY")
    n_written = sum(len(r["written"]) for r in reports)
    n_good = sum(len(r["good"]) for r in reports)
    n_bad = sum(len(r["corrupt"]) for r in reports)
    n_err = sum(1 for r in reports if r["error"])
    print(f"   tars processed  : {len(reports)}  ({n_err} with an error)")
    print(f"   files written   : {n_written}")
    print(f"   files verified  : {n_good}")
    print(f"   files CORRUPT   : {n_bad}")
    if n_bad:
        print("   corrupt files, by tar:")
        for r in reports:
            for c in r["corrupt"]:
                print(f"      {Path(r['tar']).name}: {Path(c['path']).name} -- {c['error']}")
        print("   These patients will be dropped by stage 1 (h5_ok=0) exactly as the")
        print("   nine already on the drive are. Re-download the affected tars if the")
        print("   patients matter; do not re-run extraction over a good copy.")
    affected = sorted({r["cohort"] for r in reports if r["good"]})
    if affected and not args.dry_run:
        print()
        print(f"   NEXT: python pipeline/s13_expand.py recache --cohort {' '.join(affected)}")
    return 1 if (n_bad or n_err) else 0


# ---------------------------------------------------------------------------
# 4. RECACHE
# ---------------------------------------------------------------------------

STAGE2 = {
    "prostate_dwi": "python pipeline/s02_prostate.py --cohort prostate_dwi",
    "prostate_t2": "python pipeline/s02_prostate.py --cohort prostate_t2",
    # s02_breast resumes by default. After a fold re-derivation that is exactly
    # wrong: the surviving rows would keep their OLD cv*_split values while the
    # new rows get the new ones, producing an index whose folds are internally
    # inconsistent and whose leakage no check would catch. Force a rebuild.
    "breast": "python pipeline/s02_breast.py --no-resume",
}


def stale_result_dirs(cohort: str, results_dir: Path) -> list:
    if not results_dir.exists():
        return []
    return sorted(d for d in results_dir.iterdir()
                  if d.is_dir() and d.name.startswith(cohort))


def print_invalidation(cohorts: list, results_dir: Path) -> None:
    bar = "!" * 78
    print()
    print(bar)
    print("!! ADDING PATIENTS INVALIDATES EVERY EXISTING RESULT FOR THESE COHORTS")
    print(bar)
    print("""
   The 5-fold CV partition is derived at stage 1 from the set of subjects
   present. stratified_subject_kfold deals subjects into folds in a
   deterministic order that depends on WHICH subjects exist -- so adding even
   one patient moves other patients between folds.

   Consequences, all of them hard:

     * Every existing out-of-fold prediction was made on a different partition.
       Pooling old and new predictions pools two different experiments.
     * You cannot top up. There is no "run the new patients and append": the
       old runs' training sets now contain subjects that belong to the new
       test folds, which is leakage.
     * The pooled AUCs, the subject-clustered bootstrap CIs, the DeLong tests,
       the falsification suite and the s06 verdict all inherit the old
       partition and must be recomputed from scratch.
     * The headline number changes for reasons that have nothing to do with
       phase. Do not compare a 45-subject result to a 250-subject result and
       call the difference a finding.

   The result is a NEW experiment on a LARGER cohort, reported alongside the
   old one, not an update to it. If the expanded cohort moves prostate_dwi
   above 0.500, that is a new result requiring the same falsification suite --
   background-only, permutation, acquisition-stratified splits -- before it
   means anything. A null does not get rounded up because n went up.
""")
    for c in cohorts:
        stale = stale_result_dirs(c, results_dir)
        print(f"   {c}: {len(stale)} stale results director(ies)")
        for d in stale:
            n = len(list(d.glob('*.json')))
            print(f"      {d}  ({n} run(s))")
    print()


def print_rerun_commands(cohorts: list, epochs: int, seeds: str) -> None:
    print("-" * 78)
    print("COMMANDS TO RE-RUN, IN ORDER")
    print("-" * 78)
    print()
    print("  # 0. archive the old results so they cannot be pooled with the new ones")
    print("  #    (s13_expand.py recache --archive-stale does this for you)")
    for c in cohorts:
        print(f"  mv pipeline_out/results/{c}_cv? pipeline_out/results_pre_expansion/")
    print()
    print("  # 1. stage 1: rebuild the cohort tables and RE-DERIVE the CV folds.")
    print("  #    Run it over ALL cohorts, not just the expanded one: s01 rewrites")
    print("  #    s01_summary.json wholesale, so a subset run silently drops the")
    print("  #    other cohorts' accounting from the file the report reads.")
    print("  #    Do NOT pass --no-probe: probing is what detects corrupt h5 files,")
    print("  #    and corrupt files are how we lost 9 patients already.")
    print("  ./venv/bin/python pipeline/s01_labels.py")
    print()
    print("  # 2. stage 2: rebuild the cache for the expanded cohort(s).")
    for c in cohorts:
        print(f"  ./venv/bin/{STAGE2[c]}")

    print()
    print("  # 3. re-run the headline tree. Every fold, every condition, every seed.")
    for c in cohorts:
        print(f"  for fold in 0 1 2 3 4; do \\")
        print(f"    ./venv/bin/python pipeline/s03_train.py --cohort {c} \\")
        print(f"      --conditions all --seeds {seeds} --epochs {epochs} \\")
        print(f"      --split-col cv${{fold}}_split \\")
        print(f"      --results-dir pipeline_out/results/{c}_cv${{fold}}; \\")
        print(f"  done")
    print()
    print("  # 4. stats, falsification suite, report -- all of them, not just stats.")
    print("  ./venv/bin/python pipeline/s04_stats.py")
    for c in cohorts:
        print(f"  ./venv/bin/python pipeline/s05_controls.py --cohort {c} --controls all")
    print("  ./venv/bin/python pipeline/s06_report.py")
    print()
    print("  # 5. the suites that must stay green (s04 95, s06 58, s05 3, s07 89)")
    print("  ./venv/bin/python pipeline/s04_stats.py --self-test")
    print("  ./venv/bin/python pipeline/s06_report.py --self-test")
    print()


def cmd_recache(args, invs: dict) -> int:
    import subprocess

    cohorts = list(args.cohort)
    results_dir = common.RESULTS_DIR
    print("=" * 78)
    print("4. RE-CACHE")
    print("=" * 78)
    for c in cohorts:
        inv = invs[c]
        tail = (f"{len(inv.usable_ids)} usable" if inv.probed
                else "usability not checked (--no-probe)")
        print(f"   {c:<14} {len(inv.present_ids)} patients on disk, {tail}")

    print_invalidation(cohorts, results_dir)
    print_rerun_commands(cohorts, args.epochs, args.seeds)

    if args.dry_run:
        print("DRY RUN: nothing was executed. Drop --dry-run to run steps 1 and 2.")
        return 0
    if not args.run:
        print("Stages 1 and 2 were NOT run. Add --run to execute steps 1 and 2 above")
        print("(steps 0 and 3-5 stay manual: they are hours of GPU time and an")
        print("explicit decision to discard the old results).")
        return 0

    if args.archive_stale:
        dest = results_dir.parent / f"results_pre_expansion_{time.strftime('%Y%m%d_%H%M%S')}"
        dest.mkdir(parents=True, exist_ok=True)
        for c in cohorts:
            for d in stale_result_dirs(c, results_dir):
                shutil.move(str(d), str(dest / d.name))
                print(f"   archived {d.name} -> {dest}")

    py = str(Path(sys.executable))
    steps = [[py, "pipeline/s01_labels.py"]]
    for c in cohorts:
        steps.append([py] + STAGE2[c].split()[1:])
    for cmd in steps:
        print(f"\n$ {' '.join(cmd)}")
        rc = subprocess.call(cmd, cwd=str(common.PROJECT_ROOT))
        if rc != 0:
            print(f"FAILED (exit {rc}). Stopping; the cache is now in an "
                  f"inconsistent state and must not be trained on.", file=sys.stderr)
            return rc
    print()
    print("Stages 1 and 2 done. The cache and the folds are new. Nothing in")
    print("pipeline_out/results reflects them yet -- re-run step 3 onwards.")
    return 0


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------

def self_test() -> bool:
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  {'PASS' if cond else 'FAIL'}  {name}")
        ok = ok and bool(cond)

    check("fmt_ranges collapses runs",
          fmt_ranges([1, 2, 3, 7, 9, 10]) == "001-003, 007, 009-010")
    check("fmt_ranges empty", fmt_ranges([]) == "(none)")
    check("fmt_ranges dedups", fmt_ranges([5, 5, 6]) == "005-006")

    dwi = RELEASES["prostate_dwi"]
    grid = chunk_grid(dwi)
    check("dwi grid starts at the observed chunk", grid[0] == (1, 11))
    check("dwi grid covers every patient exactly once",
          sum(b - a + 1 for a, b in grid) == dwi.n_patients)
    check("dwi grid last chunk is clamped to the release size",
          grid[-1][1] == dwi.n_patients)
    check("dwi observed chunk names reproduce",
          chunk_dirname(dwi, 45, 55) == "fastMRI_prostate_DIFF_IDS_045_055")
    t2 = RELEASES["prostate_t2"]
    check("t2 observed chunk names reproduce",
          chunk_dirname(t2, 61, 80) == "fastMRI_prostate_T2_IDS_061_080")
    br = RELEASES["breast"]
    check("breast grid ends exactly on 300", chunk_grid(br)[-1] == (291, 300))

    check("id_of parses AXDIFF",
          dwi.id_of("file_prostate_AXDIFF_007.h5") == 7)
    check("id_of rejects AXT2 for the DWI release",
          dwi.id_of("file_prostate_AXT2_007.h5") is None)
    check("id_of parses breast", br.id_of("fastMRI_breast_131_2.h5") == 131)
    check("files_for breast gives both acquisitions",
          br.files_for(131) == ["fastMRI_breast_131_1.h5", "fastMRI_breast_131_2.h5"])
    # The release zero-pads to three digits; an unpadded name would make the
    # "not in this tar" report claim every low-ID patient was missing.
    check("files_for breast zero-pads low ids",
          br.files_for(21) == ["fastMRI_breast_021_1.h5", "fastMRI_breast_021_2.h5"])
    check("files_for prostate zero-pads",
          dwi.files_for(7) == ["file_prostate_AXDIFF_007.h5"])
    # Round-trip: every expected name must parse back to the id it came from,
    # for every id in the release. This is what catches a padding change.
    check("every expected filename round-trips through id_of",
          all(r.id_of(n) == i
              for r in RELEASES.values()
              for i in range(1, r.n_patients + 1)
              for n in r.files_for(i)))

    check("classify_tar routes DWI",
          classify_tar("fastMRI_prostate_DIFF_IDS_056_066.tar") == ("prostate_dwi", 56, 66))
    check("classify_tar routes breast",
          classify_tar("fastMRI_breast_IDS_021_030.tar") == ("breast", 21, 30))
    check("classify_tar rejects junk", classify_tar("random.tar") is None)

    check("AppleDouble filter catches the sidecars",
          is_appledouble("a/b/._file.h5") and is_appledouble(".DS_Store"))
    check("AppleDouble filter passes real files",
          not is_appledouble("dir/file_prostate_AXDIFF_001.h5"))

    # The budget must stop one tar EARLY, because extraction needs the tar and
    # its contents resident at once. A plan that just sums tar sizes overfills.
    fake = Inventory(rel=dwi, probed=True)
    fake.sizes = [10 * GB]                       # 10 GB/patient -> 110 GB/tar
    fake.present_ids = set(range(1, 12))
    fake.complete_ids = set(range(1, 12))
    fake.chunks_present = ["fastMRI_prostate_DIFF_IDS_001_011"]
    empt = {c: Inventory(rel=RELEASES[c], probed=True) for c in COHORT_ORDER}
    empt["prostate_dwi"] = fake
    res = build_plan(empt, free_bytes=350 * GB, margin_gb=100.0, concurrency=1)
    # budget = 350 - 100 = 250. Tar 1: peak 110 <= 250 (running 110).
    # Tar 2: peak 220 <= 250 (running 220). Tar 3: peak 330 > 250 -> stop.
    check("budget admits exactly the tars that fit", len(res["plan"]) == 2)
    check("budget reserves for the tar in flight",
          res["running"] == 220 * GB)
    check("budget explains why it stopped", bool(res["stop_reason"]))
    check("budget never plans a tar we already hold",
          all(not r["held"] for r in res["plan"]))
    res0 = build_plan(empt, free_bytes=50 * GB, margin_gb=100.0, concurrency=1)
    check("budget recommends nothing when free < margin", res0["plan"] == [])

    check("every cohort has a stage-2 command", set(STAGE2) == set(RELEASES))
    check("breast stage 2 forces a rebuild rather than resuming",
          "--no-resume" in STAGE2["breast"])
    return ok


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="PhaseDx stage 13: inventory, budget, extract and re-cache "
                    "a cohort expansion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Downloads are NOT performed and cannot be: the NYU fastMRI "
               "data-use-agreement links are the user's.")
    p.add_argument("--dry-run", action="store_true",
                   help="print the plan and touch nothing. With no subcommand this "
                        "runs inventory + budget.")
    p.add_argument("--no-probe", action="store_true",
                   help="skip opening every h5. Faster, but then 'usable' counts are "
                        "file counts and corrupt files stay invisible.")
    # Comma-separated, not nargs='+': a greedy list argument before a subcommand
    # eats the subcommand name.
    p.add_argument("--cohorts", default=",".join(COHORT_ORDER),
                   help="comma-separated subset of " + ",".join(COHORT_ORDER))
    p.add_argument("--self-test", action="store_true")
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("inventory", help="section 1 only")

    b = sub.add_parser("budget", help="sections 1 + 2")
    b.add_argument("--margin-gb", type=float, default=DEFAULT_MARGIN_GB,
                   help=f"never plan below this much free space (default {DEFAULT_MARGIN_GB:.0f})")
    b.add_argument("--staging-concurrency", type=int, default=DEFAULT_STAGING_CONCURRENCY,
                   help="tars resident at once during extraction")
    b.add_argument("--reclaim-tars", action="store_true",
                   help="count already-extracted tar files on the drive as free space")

    e = sub.add_parser("extract", help="section 3")
    e.add_argument("--staging", required=True, help="directory holding downloaded .tar files")
    e.add_argument("--delete-tar", action="store_true",
                   help="delete each tar after ALL of its files verify clean")

    r = sub.add_parser("recache", help="section 4")
    r.add_argument("--cohort", nargs="+", required=True, choices=COHORT_ORDER,
                   help="cohort(s) whose cache and folds must be rebuilt")
    r.add_argument("--run", action="store_true",
                   help="actually execute stage 1 and stage 2 (default: print only)")
    r.add_argument("--archive-stale", action="store_true",
                   help="move the now-invalid results directories aside before running")
    r.add_argument("--epochs", type=int, default=20)
    r.add_argument("--seeds", default="42,123")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        print("s13_expand self-test")
        return 0 if self_test() else 1

    if not common.DATA_ROOT.exists():
        print(f"ERROR: data root not mounted: {common.DATA_ROOT}", file=sys.stderr)
        print("Is the drive plugged in? Check `ls /Volumes`.", file=sys.stderr)
        return 2

    cmd = args.cmd or ("budget" if args.dry_run else None)
    if cmd is None:
        build_parser().print_help()
        return 0

    want = [c.strip() for c in args.cohorts.split(",") if c.strip()]
    bad = [c for c in want if c not in RELEASES]
    if bad:
        print(f"ERROR: unknown cohort(s) {bad}; choose from {COHORT_ORDER}", file=sys.stderr)
        return 2

    probe = not args.no_probe
    print("=" * 78)
    print(f"PhaseDx cohort expansion -- {time.strftime('%Y-%m-%d %H:%M:%S')}")
    if args.dry_run:
        print("DRY RUN: nothing on disk will be created, moved or deleted.")
    print("=" * 78)
    print("scanning the drive"
          f"{' and opening every h5 (this is the slow part)' if probe else ''} ...")
    # extract and recache need every cohort's inventory regardless of --cohorts:
    # the collision guard has to see files that live under another cohort's tree.
    full = cmd in ("extract", "recache")
    invs = {c: (take_inventory(c, probe=probe) if (full or c in want)
                else Inventory(rel=RELEASES[c], probed=probe))
            for c in COHORT_ORDER}

    print_inventory({c: invs[c] for c in want})

    if cmd == "inventory":
        return 0
    if cmd == "budget":
        print_budget(invs, getattr(args, "margin_gb", DEFAULT_MARGIN_GB),
                     getattr(args, "staging_concurrency", DEFAULT_STAGING_CONCURRENCY),
                     getattr(args, "reclaim_tars", False))
        if args.dry_run:
            print()
            print("=" * 78)
            print("DRY RUN COMPLETE -- nothing was downloaded, extracted or deleted.")
            print("=" * 78)
        return 0
    if cmd == "extract":
        return cmd_extract(args, invs)
    if cmd == "recache":
        return cmd_recache(args, invs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
