#!/bin/bash
# run_full.sh -- the full PhaseDx experiment, ordered by scientific priority so
# that a partial completion still supports the paper.
#
#   1. brain / knee confound cohorts   -- the MECHANISM (phase encodes acquisition
#      identity). Fast, well-powered, and the core claim of the revamped paper.
#   2. prostate_t2                     -- the PRE-REGISTERED PRIMARY clinical cohort.
#   3. breast, prostate_dwi            -- exploratory clinical cohorts.
#
# Clinical cohorts run over the stage-1 subject-level CV folds rather than the
# official split, because the official test folds hold 7 and 4 subjects
# respectively -- below the floor at which any confirmatory claim can be read.
#
# Usage:  bash pipeline/run_full.sh [EPOCHS] [SEEDS]
set -u

PY=./venv/bin/python
EPOCHS="${1:-20}"
SEEDS="${2:-42,123}"
OUT=pipeline_out/results
LOG=pipeline_out/run_full.log

mkdir -p "$OUT"
echo "=== PhaseDx full run: epochs=$EPOCHS seeds=$SEEDS  started $(date) ===" | tee -a "$LOG"

run () {  # run <results-subdir> <cohort> <extra args...>
  local sub="$1"; local cohort="$2"; shift 2
  local dest="$OUT/$sub"
  mkdir -p "$dest"
  echo "--- $(date +%H:%M:%S)  $cohort -> $sub $*" | tee -a "$LOG"
  # Each fold gets its own results subdirectory. s03 names its output
  # <cohort>_<condition>_seed<seed>.json with no fold component, so five folds
  # written to one directory would silently overwrite each other and leave a
  # single fold masquerading as the whole cross-validation.
  #
  # Never abort the sweep on one failed configuration: a cohort that dies (a
  # single-class fold, an OOM) must not cost us the ones that would have run
  # after it. Failures are recorded and reported at the end.
  if ! $PY pipeline/s03_train.py --cohort "$cohort" \
        --conditions all --seeds "$SEEDS" --epochs "$EPOCHS" \
        --results-dir "$dest" "$@" >> "$LOG" 2>&1; then
    echo "!!! FAILED: $cohort $sub $*" | tee -a "$LOG"
    echo "$cohort $sub $*" >> pipeline_out/run_full_failures.txt
  fi
}

# --- 1. confound cohorts: the mechanism -----------------------------------
# No CV needed: brain's official split already yields 136 independent test
# subjects (58 negative / 78 positive), comfortably above the reporting floor.
# The label is receive-coil count (brain) / pulse sequence (knee). NOT pathology.
for c in brain knee; do run "confound_$c" "$c"; done

# --- 2-4. clinical cohorts over the CV folds -------------------------------
# prostate_t2 first: it is the pre-registered primary and the only cohort whose
# reconstruction is validated at r=0.998 against the vendor's own images.
for cohort in prostate_t2 breast prostate_dwi; do
  for fold in 0 1 2 3 4; do
    run "${cohort}_cv${fold}" "$cohort" --split-col "cv${fold}_split"
  done
done

echo "=== finished $(date) ===" | tee -a "$LOG"
echo "runs written: $(ls -1 "$OUT"/*.json 2>/dev/null | wc -l)" | tee -a "$LOG"
if [ -s pipeline_out/run_full_failures.txt ]; then
  echo "FAILURES:" | tee -a "$LOG"; cat pipeline_out/run_full_failures.txt | tee -a "$LOG"
fi
