#!/bin/bash
# run_controls.sh -- the stage-5 falsification suite, for real, on every cohort.
#
# Structure of the run matrix, and why it is not simply "s05 --controls all":
#
#   * label_permutation / background_only / phase_scramble all evaluate on the
#     DIAGNOSTIC test split, so on the cross-validated clinical cohorts they
#     must be run once per fold with the SAME --split-col the headline used
#     (run_full.sh: --split-col cv<k>_split). A control measured on the
#     official 7-subject split cannot falsify a headline measured on pooled
#     out-of-fold predictions.
#
#   * acquisition_split and confound_predictability REBUILD their own split
#     from every row of the index (bipartition on the acquisition key /
#     subject-grouped split stratified on the confound). They never read
#     official_split, so running them once per fold would produce five
#     byte-identical answers. They are run once per cohort.
#
#   * s05 names its output {cohort}__{control}__{variant}__{condition}__seed{N}
#     with NO fold component, so the five folds get five results directories.
#     s06 rglobs the controls tree, so nesting is fine.
#
# ORDERING: every cohort gets its cheap destroy-the-signal controls and its
# confound-predictability ceiling BEFORE any cohort spends an hour on
# permutation replicates. A partial completion then still covers all five
# cohorts on the controls that decide the paper, instead of covering one
# cohort completely and four not at all.
#
# Completed invocations leave a marker in pipeline_out/controls/.done so the
# script can be killed and restarted without redoing hours of training.
#
# Usage: bash pipeline/run_controls.sh <stream>     (stream = 1 or 2)
set -u

PY=./venv/bin/python
ROOT=pipeline_out/controls/results
DONE=pipeline_out/controls/.done
STREAM="${1:-1}"
LOG=pipeline_out/controls_run_stream${STREAM}.log

mkdir -p "$ROOT" "$DONE"
echo "=== stage 5 falsification suite, stream $STREAM, started $(date) ===" | tee -a "$LOG"

step () {   # step <tag> <cohort> <controls> <results-subdir> <n_perm> [extra args...]
  local tag="$1"; local cohort="$2"; local controls="$3"; local sub="$4"; local nperm="$5"
  shift 5
  if [ -f "$DONE/$tag" ]; then
    echo "--- $(date +%H:%M:%S)  SKIP (already done) $tag" | tee -a "$LOG"
    return 0
  fi
  echo "--- $(date +%H:%M:%S)  $tag" | tee -a "$LOG"
  if $PY pipeline/s05_controls.py --cohort "$cohort" --controls "$controls" \
        --epochs 20 --n-permutations "$nperm" \
        --results-dir "$ROOT/$sub" \
        --ckpt-dir pipeline_out/controls/checkpoints "$@" >> "$LOG" 2>&1; then
    touch "$DONE/$tag"
  else
    echo "!!! FAILED: $tag" | tee -a "$LOG"
  fi
}

folded () { # folded <cohort> <controls> <n_perm> <tagsuffix>
  local cohort="$1"; local controls="$2"; local nperm="$3"; local suf="$4"
  for fold in 0 1 2 3 4; do
    step "${cohort}_cv${fold}_${suf}" "$cohort" "$controls" "${cohort}_cv${fold}" \
         "$nperm" --split-col "cv${fold}_split"
  done
}

# brain/knee are the confound cohorts: their LABEL is already an acquisition
# property (receive-coil count / pulse sequence), so the confound targets must
# be OTHER hardware descriptors -- never receiver_channels (that IS the brain
# headline) and never te/flip_angle (stage 1: those separate the knee classes
# at p~1e-40, so predicting them is close to predicting the label itself).
BK_CONFOUNDS=device_id,institution,field_strength,scanner_model

case "$STREAM" in
1)
  # ---------- phase A: the controls that decide the paper, all CV cohorts ----
  for cohort in prostate_t2 breast prostate_dwi; do
    folded "$cohort" background_only,phase_scramble 1 destroy
    step "${cohort}_acqconf" "$cohort" acquisition_split,confound_predictability \
         "$cohort" 1
  done
  # ---------- phase B: permutation nulls, priority order --------------------
  folded prostate_t2 label_permutation 10 perm
  folded breast      label_permutation 10 perm
  ;;
2)
  # ---------- the confound cohorts: brain first, it carries the mechanism ----
  step brain_acqconf brain confound_predictability,acquisition_split brain 1 \
       --confounds "$BK_CONFOUNDS"
  step brain_destroy brain background_only,phase_scramble brain 1
  step knee_acqconf  knee  confound_predictability,acquisition_split knee 1 \
       --confounds "$BK_CONFOUNDS"
  step knee_destroy  knee  background_only,phase_scramble knee 1
  step brain_perm    brain label_permutation brain 20
  step knee_perm     knee  label_permutation knee 20
  # ---------- then take prostate_dwi's permutation load off stream 1 --------
  folded prostate_dwi label_permutation 10 perm
  ;;
esac

echo "=== stream $STREAM finished $(date) ===" | tee -a "$LOG"
