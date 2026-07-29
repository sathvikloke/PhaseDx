#!/bin/bash
# Helper stream: breast permutation folds from the far end (4, 3, then 2) while
# the main stream walks 0..4 forward. Same done-markers, so whichever arrives
# second skips; the reverse order plus the main stream's prostate_t2 head start
# keeps them out of the same fold.
set -u
PY=./venv/bin/python
ROOT=pipeline_out/controls/results
DONE=pipeline_out/controls/.done
LOG=pipeline_out/controls_run_helper.log
mkdir -p "$ROOT" "$DONE"
for fold in 4 3 2; do
  tag="breast_cv${fold}_perm"
  [ -f "$DONE/$tag" ] && { echo "SKIP $tag" | tee -a "$LOG"; continue; }
  echo "--- $(date +%H:%M:%S) $tag" | tee -a "$LOG"
  if $PY pipeline/s05_controls.py --cohort breast --controls label_permutation \
       --epochs 20 --n-permutations 10 --split-col "cv${fold}_split" \
       --results-dir "$ROOT/breast_cv${fold}" \
       --ckpt-dir pipeline_out/controls/checkpoints >> "$LOG" 2>&1; then
    touch "$DONE/$tag"
  else
    echo "!!! FAILED $tag" | tee -a "$LOG"
  fi
done
echo "=== helper finished $(date) ===" | tee -a "$LOG"
