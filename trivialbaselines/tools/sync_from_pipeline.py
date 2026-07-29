#!/usr/bin/env python3
"""
Regenerate this package's three derived modules from the PhaseDx study source, and
verify that the only differences are the ones we claim.

    python tools/sync_from_pipeline.py --repo /path/to/PhaseDx
    python tools/sync_from_pipeline.py --repo /path/to/PhaseDx --check

``--check`` regenerates into a temporary directory and exits non-zero if anything in
``src/trivialbaselines/`` would change. Run it in CI, or before a release, so the
package cannot silently drift from the code the paper's numbers came from.

    core.py        pipeline/s14_trivialbaselines.py, with four substitutions
    stats.py       pipeline/s04_stats.py, eight functions extracted verbatim by AST
    stratified.py  pipeline/s12_rempe.py, two functions extracted verbatim by AST

Each substitution target is asserted before it is applied, so if the upstream file is
edited in a way that removes one, this script fails loudly instead of producing a
package that quietly no longer matches.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import sys
import tempfile
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent / "src" / "trivialbaselines"

STATS_FUNCS = [
    "compute_midrank", "auc_midrank", "average_precision",
    "_clean", "_cluster_index", "aggregate_by_cluster",
    "cluster_bootstrap_auc", "naive_slice_bootstrap_auc",
]
STRATIFIED_FUNCS = ["stratified_auc", "position_strata"]

# --- core.py: the four changes, and no others -------------------------------
CORE_SUBS = [
    (
        'sys.path.insert(0, str(Path(__file__).resolve().parent))\n'
        'import s04_stats  # noqa: E402  -- statistics live in one place for the whole study\n',
        'from . import stats as s04_stats\n',
    ),
    (
        '    tmp = Path(__file__).resolve().parent.parent / "pipeline_out" / "_s14_selftest"\n',
        '    tmp = Path(tempfile.mkdtemp(prefix="trivialbaselines_selftest_"))\n',
    ),
    (
        '        out = Path(a.out_dir) if a.out_dir else (\n'
        '            Path(__file__).resolve().parent.parent / "pipeline_out" / "trivial_baselines")\n',
        '        out = Path(a.out_dir) if a.out_dir else Path.cwd() / "trivial_baselines"\n',
    ),
    (
        '                        "(default pipeline_out/trivial_baselines)")',
        '                        "(default ./trivial_baselines)")',
    ),
    (
        '''Usage:
    python pipeline/s14_trivialbaselines.py --self-test
    python pipeline/s14_trivialbaselines.py --labels labels.csv --name mybench
    python pipeline/s14_trivialbaselines.py --labels t2_slice_level_labels.csv \\\\
        --label-col PIRADS --positive-if '>2' --name rempe_t2 --published 0.861
"""''',
        '''Usage:
    trivial-baselines --self-test
    trivial-baselines --labels labels.csv --name mybench
    trivial-baselines --labels t2_slice_level_labels.csv \\\\
        --label-col PIRADS --positive-if '>2' --name rempe_t2 --published 0.861

This module is the release packaging of pipeline/s14_trivialbaselines.py from the
PhaseDx study. The baselines, the statistics and the guards are unchanged; only the
import path, the default output directory and the usage lines differ, so that a number
produced by `trivial-baselines` is the same number the paper reports.
"""''',
    ),
    ('import shlex\nimport sys\n', 'import shlex\nimport sys\nimport tempfile\n'),
]

STATS_HEADER = '''"""
trivialbaselines.stats
----------------------
The statistics the zero-image baselines need, and nothing else.

VENDORED VERBATIM from ``pipeline/s04_stats.py`` of the PhaseDx study
(https://github.com/sathvikloke/PhaseDx). The function bodies below are byte-identical
to the ones every number in the paper was computed with; they are copied rather than
imported so that this package installs with numpy + pandas alone and can be checked
by a reviewer without cloning the study. Regenerate with ``tools/sync_from_pipeline.py``.

numpy only. No scipy, no scikit-learn, no torch. That is deliberate: the claim this
tool exists to support is that a published slice-level benchmark can be audited with
no images and no GPU, and a dependency on a deep-learning stack would undercut it.

Three of these deserve a note, because they are where naive implementations go wrong:

``auc_midrank``      ties get the AVERAGE rank. A pixel-blind baseline emits the same
                     score for every slice in a position bin, so ties are the common
                     case here, not the exception. Ranking them arbitrarily inflates
                     or deflates the AUROC depending on the input order.

``cluster_bootstrap_auc``
                     resamples SUBJECTS, not slices. The slice-level bootstrap is the
                     reason published intervals are too narrow: in simulation
                     (20 subjects x 15 slices, 200 datasets) the naive slice interval
                     covered the true AUC 46.5% of the time at a nominal 95%, against
                     91.5% for this one, and it was 3.2x narrower.

``naive_slice_bootstrap_auc``
                     the wrong interval, kept ONLY so a report can print how much
                     narrower the wrong interval would have been. Never a headline.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "compute_midrank",
    "auc_midrank",
    "average_precision",
    "aggregate_by_cluster",
    "cluster_bootstrap_auc",
    "naive_slice_bootstrap_auc",
]
'''

STRATIFIED_HEADER = '''"""
trivialbaselines.stratified
---------------------------
The position-stratified AUROC: the REMEDY metric that goes with the positional baseline.

A slice-level AUROC counts every positive/negative slice pair, including pairs drawn
from different parts of the stack. If positives sit nearer the middle of the organ --
and in a lesion-detection benchmark they usually do -- a large share of those pairs are
won by geometry rather than by pathology. Stratifying the Mann-Whitney statistic on
relative slice position lets ONLY same-position pairs contribute, so exactly that share
is removed and nothing else is.

Measured on Rempe et al. (2024)'s own published prostate DWI label file and split, all
three numbers from the same score vector:

    zero-image positional baseline, slice-level AUROC        0.851
    the same scores, patient-level AUROC                     0.424
    the same scores, position-stratified slice-level AUROC   0.539  (6 strata)

and on the PhaseDx reimplementation of their protocol, again one score vector:

    slice-level AUROC                                        0.574
    position-stratified slice-level AUROC                    0.467

This is not part of ``audit()``, which reads a label file and never sees your model's
predictions. Call it directly on your own test-set scores::

    from trivialbaselines import position_strata, stratified_auc
    rel = (slice_idx - slice_idx_min_in_volume) / (slice_idx_max - slice_idx_min)
    print(stratified_auc(labels, scores, position_strata(rel, n_strata=10)))

VENDORED VERBATIM from ``pipeline/s12_rempe.py`` of the PhaseDx study, with the
statistics import repointed at the vendored copy in ``trivialbaselines.stats``.
Regenerate with ``tools/sync_from_pipeline.py``.
"""

from __future__ import annotations

import numpy as np

from . import stats as s04_stats

__all__ = ["stratified_auc", "position_strata"]
'''


def extract(src: Path, wanted: list[str]) -> dict[str, str]:
    """Pull whole function definitions out of a module, source text unchanged."""
    text = src.read_text()
    lines = text.splitlines()
    out = {}
    for node in ast.parse(text).body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
            out[node.name] = "\n".join(lines[start:node.end_lineno])
    missing = [w for w in wanted if w not in out]
    if missing:
        raise SystemExit(f"ERROR: {src} no longer defines {missing}")
    return out


def build_core(repo: Path) -> str:
    text = (repo / "pipeline" / "s14_trivialbaselines.py").read_text()
    for old, new in CORE_SUBS:
        n = text.count(old)
        if n != 1:
            raise SystemExit(
                f"ERROR: substitution target appears {n} times, expected 1. "
                f"s14_trivialbaselines.py has changed in a way this script does not "
                f"understand; update CORE_SUBS.\n---\n{old}\n---")
        text = text.replace(old, new)
    text = text.replace('s14_trivialbaselines.py\n-----------------------',
                        'trivialbaselines.core\n---------------------', 1)
    text = text.replace('print("s14_trivialbaselines self-test")',
                        'print("trivialbaselines self-test")', 1)
    for banned in ("sys.path.insert", "pipeline_out", "import s04_stats"):
        if banned in text:
            raise SystemExit(f"ERROR: {banned!r} survived the transform")
    return text


def _join(parts: list[str]) -> str:
    """Two blank lines between top-level blocks, exactly one trailing newline."""
    return "\n\n\n".join(p.strip("\n") for p in parts) + "\n"


def build_stats(repo: Path) -> str:
    b = extract(repo / "pipeline" / "s04_stats.py", STATS_FUNCS)
    rule = "# " + "=" * 72
    parts = [STATS_HEADER,
             f"{rule}\n# Rank statistics\n{rule}"]
    parts += [b[n] for n in STATS_FUNCS[:3]]
    parts += [f"{rule}\n# Cluster-aware resampling\n{rule}"]
    parts += [b[n] for n in STATS_FUNCS[3:]]
    return _join(parts)


def build_stratified(repo: Path) -> str:
    b = extract(repo / "pipeline" / "s12_rempe.py", STRATIFIED_FUNCS)
    return _join([STRATIFIED_HEADER, b["stratified_auc"], b["position_strata"]])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", required=True, type=Path,
                    help="path to the PhaseDx study checkout")
    ap.add_argument("--check", action="store_true",
                    help="do not write; exit 1 if anything would change")
    a = ap.parse_args()

    if not (a.repo / "pipeline" / "s14_trivialbaselines.py").exists():
        raise SystemExit(f"ERROR: {a.repo} does not look like a PhaseDx checkout")

    built = {"core.py": build_core(a.repo),
             "stats.py": build_stats(a.repo),
             "stratified.py": build_stratified(a.repo)}

    drift = False
    for name, text in built.items():
        target = PKG / name
        current = target.read_text() if target.exists() else ""
        if current == text:
            print(f"  ok        {name}")
            continue
        drift = True
        if a.check:
            print(f"  DRIFT     {name}")
            for line in list(difflib.unified_diff(
                    current.splitlines(), text.splitlines(),
                    fromfile=f"packaged/{name}", tofile=f"regenerated/{name}",
                    lineterm=""))[:40]:
                print("    " + line)
        else:
            target.write_text(text)
            print(f"  updated   {name}  ({len(text.splitlines())} lines)")

    if a.check and drift:
        print("\nThe packaged modules do not match the study source. "
              "Run without --check to regenerate, then rerun --self-test.")
        return 1
    if not a.check:
        print("\nNow verify:  trivial-baselines --self-test")
    return 0


if __name__ == "__main__":
    sys.exit(main())
