#!/usr/bin/env python3
"""
run_archzoo_sweep.py -- driver for the stage-11 architecture generalisation sweep.

Writes into a SEPARATE tree, pipeline_out/results_arch/<tree>/<fold_dir>/, so the
existing 102-run headline tree in pipeline_out/results/ is untouched and stage 4
never merges the two experiments. One stage-4 invocation per <tree>.

Layout produced (fold dirs are what s04_stats.parse_fold_dir reads):

    pipeline_out/results_arch/<arch>/prostate_t2_cv{0..4}/*.json
    pipeline_out/results_arch/<arch>/confound_brain/*.json
    pipeline_out/results_arch/<arch>/prostate_dwi_cv{0..4}/*.json
    pipeline_out/results_arch/resnet18_scratch/prostate_t2_cv{0..4}/*.json

Ordering is deliberate: prostate_t2 (pre-registered primary null) -> brain
(the mechanism) -> prostate_dwi. Within a cohort, cheap+important architectures
first. If the sweep is cut short the most important cells are already on disk.

Resumable: a cell whose output JSON already exists is skipped. Every failure is
logged to the manifest and the sweep continues.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
import traceback
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common          # noqa: E402
import s03_train       # noqa: E402
import s11_archzoo     # noqa: E402

ROOT = common.OUT_ROOT / "results_arch"
SEED = 42

# resnet18 is the regression baseline and cheapest; complex_small is the
# phase-native model the objection is really about; vit_b_16 is last because it
# is ~10x the FLOPs of resnet18.
ARCH_ORDER = ["resnet18", "complex_small", "densenet121", "resnet50",
              "convnext_tiny", "vit_b_16"]

logger = logging.getLogger("archzoo_sweep")


def build_plan():
    """Every (tree, cohort, fold dir, split col, arch, condition, init) cell, in run order."""
    cells = []

    def add(tree, cohort, subdir, split_col, arch, conditions, scratch, group):
        for cond in conditions:
            cells.append({
                "group": group, "tree": tree, "cohort": cohort, "subdir": subdir,
                "split_col": split_col, "arch": arch, "condition": cond,
                "seed": SEED, "scratch": scratch,
            })

    # ---- 1. prostate_t2: pre-registered primary null, 5 folds x 3 conditions
    for arch in ARCH_ORDER:
        for k in range(5):
            add(arch, "prostate_t2", f"prostate_t2_cv{k}", f"cv{k}_split", arch,
                ["magnitude", "phase", "both"], False, "1_prostate_t2")

    # ---- 2. prostate_t2 from random init: is ImageNet pretraining doing the work?
    for k in range(5):
        add("resnet18_scratch", "prostate_t2", f"prostate_t2_cv{k}", f"cv{k}_split",
            "resnet18", ["magnitude", "phase", "both"], True, "2_prostate_t2_scratch")

    # ---- 3. brain: the mechanism (phase -> receive-coil count), official split
    for arch in ARCH_ORDER:
        add(arch, "brain", "confound_brain", "official_split", arch,
            ["magnitude", "phase"], False, "3_brain")

    # ---- 4. prostate_dwi
    for arch in ARCH_ORDER:
        for k in range(5):
            add(arch, "prostate_dwi", f"prostate_dwi_cv{k}", f"cv{k}_split", arch,
                ["magnitude", "phase", "both"], False, "4_prostate_dwi")

    return cells


def make_train_args(split_col: str, scratch: bool):
    """s03_train's own defaults + only what this sweep owns (see s11.build_train_args)."""
    base = s03_train.parse_args([])
    base.split_col = split_col
    base.region = "full"
    base.workers = 0
    base.dry_run = False
    base.no_pretrained = bool(scratch)
    return base


def expected_paths(cell) -> list[Path]:
    """Both possible output names: init is tagged from what the model ACTUALLY got."""
    d = ROOT / cell["tree"] / cell["subdir"]
    return [d / (s11_archzoo._stem(cell["cohort"], cell["arch"], cell["condition"],
                                   cell["seed"], init) + ".json")
            for init in ("imagenet", "scratch")]


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--groups", default="all", help="comma-separated group prefixes to run")
    p.add_argument("--archs", default="all")
    p.add_argument("--limit", type=int, default=None, help="stop after N cells (calibration)")
    p.add_argument("--device", default="auto")
    p.add_argument("--manifest", default=str(ROOT / "_manifest.jsonl"))
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")

    cells = build_plan()
    if args.groups != "all":
        want = tuple(g.strip() for g in args.groups.split(","))
        cells = [c for c in cells if c["group"] in want]
    if args.archs != "all":
        want = tuple(a.strip() for a in args.archs.split(","))
        cells = [c for c in cells if c["arch"] in want]

    device = s03_train.pick_device(args.device)
    ROOT.mkdir(parents=True, exist_ok=True)
    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    # index/h5 prep is per (cohort, split_col) and is reused across archs
    index_cache: dict = {}

    def get_index(cohort, split_col, train_args):
        key = (cohort, split_col)
        if key not in index_cache:
            cache_dir = Path(train_args.cache_dir)
            h5 = cache_dir / f"{cohort}.h5"
            raw = pd.read_csv(cache_dir / f"{cohort}_index.csv")
            idx = s03_train.prepare_index(raw, train_args.val_frac,
                                          train_args.val_split_seed, split_col=split_col)
            index_cache[key] = (idx, h5)
        return index_cache[key]

    n_run = n_skip = n_fail = 0
    t_start = time.time()
    for i, cell in enumerate(cells):
        if args.limit is not None and n_run >= args.limit:
            break
        paths = expected_paths(cell)
        if any(q.exists() for q in paths):
            n_skip += 1
            continue

        out_dir = ROOT / cell["tree"] / cell["subdir"]
        out_dir.mkdir(parents=True, exist_ok=True)
        label = (f"{cell['group']} {cell['tree']}/{cell['subdir']} "
                 f"{cell['arch']}/{cell['condition']}/seed{cell['seed']}")
        logger.info("#" * 78)
        logger.info("CELL %d/%d  %s   [%d run, %d skip, %d fail, %.1f min elapsed]",
                    i + 1, len(cells), label, n_run, n_skip, n_fail,
                    (time.time() - t_start) / 60.0)
        logger.info("#" * 78)

        rec = {"cell": cell, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
        t0 = time.time()
        try:
            train_args = make_train_args(cell["split_col"], cell["scratch"])
            index, h5 = get_index(cell["cohort"], cell["split_col"], train_args)
            res = s11_archzoo.run_tagged(
                cell["cohort"], cell["arch"], cell["condition"], cell["seed"],
                index, device, train_args,
                results_root=out_dir, h5_path=h5, arrays=None,
                scratch=cell["scratch"], allow_download=False,
                keep_checkpoints=False,
            )
            rec.update({"status": "ok", "test_auc": res["test_auc"],
                        "test_ap": res["test_ap"], "init": res["init"],
                        "pretrained": res["pretrained"],
                        "params_total": res["params_total"],
                        "best_epoch": res["best_epoch"], "path": res["path"]})
            n_run += 1
        except BaseException as exc:  # noqa: BLE001 - a dead arm must not kill the sweep
            rec.update({"status": "FAILED", "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc()[-2000:]})
            logger.error("CELL FAILED: %s -- %s: %s", label, type(exc).__name__, exc)
            n_fail += 1
            if isinstance(exc, KeyboardInterrupt):
                rec["seconds"] = time.time() - t0
                with open(manifest, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                raise
        rec["seconds"] = time.time() - t0
        with open(manifest, "a") as fh:
            fh.write(json.dumps(rec) + "\n")

        # a partial _scratch dir left by a crash would be picked up by s04's rglob
        shutil.rmtree(out_dir / "_scratch", ignore_errors=True)
        if device.type == "mps":
            torch.mps.empty_cache()

    logger.info("DONE: %d run, %d skipped (already on disk), %d failed, %.1f min",
                n_run, n_skip, n_fail, (time.time() - t_start) / 60.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
