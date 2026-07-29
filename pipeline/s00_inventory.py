"""
s00_inventory.py
----------------
Stage 0 of the PhaseDx pipeline: discovery.

Walks the FastMRI drive and records, for every .h5 file, the exact dataset
keys, shapes, dtypes, and attributes. Also finds every CSV/XLSX that could
carry labels. Nothing downstream can be written correctly until we know the
real on-disk layout, so this runs first and writes a machine-readable manifest.

This stage is read-only. It never modifies the drive.

Usage:
    python pipeline/s00_inventory.py --root /Volumes/Research/fastmridatasets
    python pipeline/s00_inventory.py --root /Volumes/Research/fastmridatasets --full
"""

import argparse
import json
import re
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np

# --------------------------------------------------------------------------
# Organ attribution
# --------------------------------------------------------------------------
#
# The previous implementation lowercased the WHOLE path and asked whether an
# organ word appeared anywhere in it. That is wrong in two ways that both bite
# on this drive:
#
#   1. It matches the BASENAME. knee/val on this drive holds a duplicate copy of
#      every brain file (file_brain_*.h5), so 460 files sitting in the knee tree
#      were reported as organ='brain' -- and, because dict iteration order put
#      'brain' before 'knee', that was also true of the *directory* attribution
#      used for the per-organ file counts. The inventory therefore claimed the
#      knee folder contained brain data and undercounted the knee release.
#   2. It matches anything above the data root. A drive mounted under
#      /Volumes/brain-backup, or a checkout in ~/knee/, relabels every file on
#      the disk. Nothing in the old rule was anchored.
#
# The fix is to derive the organ from the path COMPONENT immediately after the
# data-root directory, and to fall back to exact component matches (never
# substrings) if the root is not recognisable. Where a file's own name follows a
# fastMRI naming convention we ALSO record what the name implies, so a file
# sitting in the wrong organ tree is visible in the manifest rather than being
# silently reassigned. Directory and filename attribution are two separate
# fields precisely because on this drive they disagree for 460 files.

DATA_ROOT_NAME = "fastmridatasets"

KNOWN_ORGANS = ("prostate", "breast", "brain", "knee")

# Release directories whose name is not exactly the organ name.
ORGAN_DIR_ALIASES = {"breast_updated": "breast"}

# fastMRI naming conventions. Used only to detect misfiled copies -- never to
# override the directory attribution.
ORGAN_FILENAME_PATTERNS = (
    (re.compile(r"^file_brain_.+\.h5$", re.I), "brain"),
    (re.compile(r"^file\d+\.h5$", re.I), "knee"),
    (re.compile(r"^fastMRI_breast_\d+_\d+\.h5$", re.I), "breast"),
    (re.compile(r"^file_prostate_.+\.h5$", re.I), "prostate"),
)


def _normalise_organ(component: str) -> str:
    comp = component.lower()
    comp = ORGAN_DIR_ALIASES.get(comp, comp)
    return comp if comp in KNOWN_ORGANS else ""


def organ_from_path(path, root_name: str = DATA_ROOT_NAME) -> str:
    """
    Organ implied by the DIRECTORY the file sits in.

    Anchored on the data-root component: `.../fastmridatasets/<organ>/...`. If
    the root is not present (a relocated copy, a unit test), fall back to the
    deepest path component that exactly names a known organ. Components are
    compared for equality, so 'file_brain_AXT2_...h5' inside knee/val is NOT
    read as brain, and a mount point called /Volumes/brain does not relabel a
    prostate file.
    """
    parts = [p.lower() for p in Path(str(path)).parts]
    root_hits = [i for i, p in enumerate(parts) if p == root_name.lower()]
    if root_hits and root_hits[-1] + 1 < len(parts):
        organ = _normalise_organ(parts[root_hits[-1] + 1])
        if organ:
            return organ
        return "unknown"
    # Deepest-first so /a/brain/val/x.h5 resolves to the directory it is in.
    for comp in reversed(parts):
        organ = _normalise_organ(comp)
        if organ:
            return organ
    return "unknown"


def organ_from_filename(path) -> str:
    """Organ implied by the fastMRI FILENAME convention, or 'unknown'."""
    name = Path(str(path)).name
    for rx, organ in ORGAN_FILENAME_PATTERNS:
        if rx.match(name):
            return organ
    return "unknown"


def guess_organ(path, root_name: str = DATA_ROOT_NAME) -> str:
    """Organ attribution used for grouping. Directory-derived; see organ_from_path."""
    return organ_from_path(path, root_name)


def jsonable(value):
    """Coerce an h5 attribute value into something json.dump can handle."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.size > 32:
            return f"<ndarray shape={value.shape} dtype={value.dtype}>"
        return [jsonable(v) for v in value.tolist()]
    return value


def describe_h5(path: Path, read_header: bool = False) -> dict:
    """
    Open one h5 file and record its structure without loading bulk arrays.

    Returns a dict with keys/shapes/dtypes/attrs, or an 'error' key if the
    file is unreadable (truncated and corrupt files are expected — the
    manuscript already reports excluding four prostate files for this).
    """
    dir_organ = organ_from_path(path)
    name_organ = organ_from_filename(path)
    record = {
        "path": str(path),
        "organ": dir_organ,
        "organ_from_filename": name_organ,
        # 1 when the file's own name says it belongs to a different release than
        # the directory it is stored in. On this drive that fires for the brain
        # files duplicated into knee/val.
        "organ_mismatch": int(name_organ != "unknown" and name_organ != dir_organ),
        "size_bytes": path.stat().st_size,
        "datasets": {},
        "attrs": {},
    }
    try:
        with h5py.File(path, "r") as f:
            record["attrs"] = {k: jsonable(v) for k, v in f.attrs.items()}

            def visit(name, obj):
                if isinstance(obj, h5py.Dataset):
                    record["datasets"][name] = {
                        "shape": list(obj.shape),
                        "dtype": str(obj.dtype),
                    }

            f.visititems(visit)

            # ismrmrd_header holds acquisition params (sequence, field strength,
            # scanner model). We need these for the confound-control analysis,
            # so pull the raw XML when asked.
            if read_header and "ismrmrd_header" in f:
                raw = f["ismrmrd_header"][()]
                if isinstance(raw, bytes):
                    record["ismrmrd_header"] = raw.decode("utf-8", errors="replace")
                else:
                    record["ismrmrd_header"] = str(raw)
    except Exception as exc:  # noqa: BLE001 - we want to log, not crash the scan
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def find_label_files(root: Path) -> list:
    """Locate candidate label sources (CSV/XLSX/JSON) anywhere under root."""
    out = []
    for pattern in ("*.csv", "*.CSV", "*.xlsx", "*.XLSX", "*.json"):
        for p in root.rglob(pattern):
            try:
                size = p.stat().st_size
            except OSError:
                continue
            entry = {"path": str(p), "size_bytes": size}
            # Grab the header row of CSVs so we can see the column names now
            # rather than after another round-trip to the drive.
            if p.suffix.lower() == ".csv":
                try:
                    with open(p, "r", encoding="utf-8", errors="replace") as fh:
                        entry["header"] = fh.readline().strip()[:2000]
                        entry["first_row"] = fh.readline().strip()[:2000]
                except OSError as exc:
                    entry["error"] = str(exc)
            out.append(entry)
    return out


def main():
    parser = argparse.ArgumentParser(description="PhaseDx stage 0: inventory the FastMRI drive")
    parser.add_argument("--root", required=True, help="Root of the FastMRI data on the drive")
    parser.add_argument("--out", default="pipeline_out/inventory",
                        help="Directory for the manifest output")
    parser.add_argument("--full", action="store_true",
                        help="Probe every file. Default probes a sample per organ folder.")
    parser.add_argument("--sample-per-dir", type=int, default=2,
                        help="Files to probe per directory when not using --full")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"ERROR: root does not exist: {root}", file=sys.stderr)
        print("Is the drive mounted? Check `ls /Volumes`.", file=sys.stderr)
        sys.exit(2)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {root} ...")
    all_h5 = sorted(root.rglob("*.h5"))
    print(f"  found {len(all_h5)} .h5 files")

    # Group by parent directory so a sampled scan still touches every folder.
    by_dir = defaultdict(list)
    for p in all_h5:
        by_dir[p.parent].append(p)

    if args.full:
        to_probe = all_h5
    else:
        to_probe = []
        for d, files in sorted(by_dir.items()):
            to_probe.extend(files[: args.sample_per_dir])
    print(f"  probing {len(to_probe)} files "
          f"({'full scan' if args.full else 'sampled'}) across {len(by_dir)} directories")

    records = []
    for i, p in enumerate(to_probe, 1):
        if i % 50 == 0 or i == len(to_probe):
            print(f"    {i}/{len(to_probe)}")
        records.append(describe_h5(p, read_header=True))

    # ---- Summarise the distinct layouts we saw, per organ -------------------
    # This is the part that actually drives the reader code: if prostate files
    # are all (S, D, C, H, W) and breast are all (2, S, kx, C, ky), we can write
    # exact readers. Any outlier shape becomes an explicit exclusion.
    layouts = defaultdict(Counter)
    attr_keys = defaultdict(Counter)
    errors = []
    for r in records:
        organ = r["organ"]
        if "error" in r:
            errors.append({"path": r["path"], "organ": organ, "error": r["error"]})
            continue
        signature = "; ".join(
            f"{k}{tuple(v['shape'])}:{v['dtype']}"
            for k, v in sorted(r["datasets"].items())
        )
        layouts[organ][signature] += 1
        for k in r["attrs"]:
            attr_keys[organ][k] += 1

    file_counts = Counter(guess_organ(p) for p in all_h5)

    # Directory organ vs filename organ, over EVERY .h5 on the drive (not just
    # the probed sample). This is the cell that exposes the duplication hazard:
    # ('knee', 'brain') is non-empty because knee/val holds a copy of the brain
    # release, and a naive glob over both trees double-counts every brain
    # patient and can put one patient in both train and test.
    misfiled = Counter()
    for p in all_h5:
        n_org = organ_from_filename(p)
        if n_org != "unknown" and n_org != guess_organ(p):
            misfiled[(guess_organ(p), n_org)] += 1

    # Basenames that occur in more than one directory anywhere under root.
    by_basename = defaultdict(list)
    for p in all_h5:
        by_basename[p.name].append(str(p))
    dup_basenames = {n: v for n, v in by_basename.items() if len(v) > 1}

    summary = {
        "root": str(root),
        "total_h5_files": len(all_h5),
        "files_probed": len(to_probe),
        "full_scan": args.full,
        "files_per_organ": dict(file_counts),
        "files_per_organ_by_filename": dict(
            Counter(organ_from_filename(p) for p in all_h5)
        ),
        "misfiled_files_dir_organ_to_filename_organ": {
            f"{d}->{n}": c for (d, n), c in misfiled.most_common()
        },
        "duplicate_basenames": len(dup_basenames),
        "duplicate_basename_examples": {
            n: v for n, v in list(sorted(dup_basenames.items()))[:5]
        },
        "directories_per_organ": dict(
            Counter(guess_organ(d) for d in by_dir)
        ),
        "layouts_per_organ": {
            organ: [{"signature": sig, "count": n} for sig, n in c.most_common()]
            for organ, c in layouts.items()
        },
        "attr_keys_per_organ": {
            organ: dict(c.most_common()) for organ, c in attr_keys.items()
        },
        "unreadable_files": errors,
        "label_file_candidates": find_label_files(root),
    }

    (out_dir / "manifest.json").write_text(json.dumps(records, indent=2))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # ---- Human-readable report ---------------------------------------------
    print("\n" + "=" * 72)
    print("INVENTORY SUMMARY")
    print("=" * 72)
    print(f"Total .h5 files: {len(all_h5)}")
    for organ, n in file_counts.most_common():
        print(f"  {organ:<10} {n:>6} files (by directory)")
    print("\nBy filename convention (disagreement = a misfiled or duplicated copy):")
    for organ, n in Counter(organ_from_filename(p) for p in all_h5).most_common():
        print(f"  {organ:<10} {n:>6} files")
    if misfiled:
        print("\n!! FILES WHOSE NAME AND DIRECTORY DISAGREE:")
        for (d, n_org), c in misfiled.most_common():
            print(f"   in {d}/ but named like {n_org}: {c} file(s)")
    if dup_basenames:
        print(f"\n!! {len(dup_basenames)} basename(s) appear in more than one directory. "
              f"A glob over both trees double-counts these patients.")
        for name, paths in list(sorted(dup_basenames.items()))[:3]:
            print(f"   {name}")
            for pp in paths:
                print(f"      {pp}")
    if errors:
        print(f"\nUnreadable files: {len(errors)}")
        for e in errors[:10]:
            print(f"  {e['path']}: {e['error']}")

    print("\nDistinct k-space layouts per organ:")
    for organ, entries in summary["layouts_per_organ"].items():
        print(f"\n  [{organ}]")
        for e in entries[:5]:
            print(f"    x{e['count']:<4} {e['signature'][:300]}")

    print("\nFile-level attribute keys per organ:")
    for organ, keys in summary["attr_keys_per_organ"].items():
        print(f"  [{organ}] {', '.join(keys) if keys else '(none)'}")

    print(f"\nLabel file candidates: {len(summary['label_file_candidates'])}")
    for c in summary["label_file_candidates"][:20]:
        print(f"  {c['path']}")
        if c.get("header"):
            print(f"      cols: {c['header'][:200]}")

    print("\n" + "=" * 72)
    print(f"Wrote {out_dir/'manifest.json'} and {out_dir/'summary.json'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
