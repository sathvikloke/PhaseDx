#!/bin/zsh
# One stage-4 invocation PER ARCHITECTURE TREE. Running s04 over the whole
# results_arch/ at once would pool different architectures into one estimate,
# because s04's unit key is (cohort, region, split family, condition, seed) and
# carries no architecture dimension.
cd /Users/sathvikloke/Downloads/PhaseDx
for t in pipeline_out/results_arch/*/; do
  name=$(basename $t)
  [[ -d "$t" ]] || continue
  n=$(find "$t" -name '*.json' ! -name 'statistics.json' | wc -l | tr -d ' ')
  [[ "$n" == "0" ]] && continue
  echo "=== stage 4: $name  ($n run JSONs) ==="
  venv/bin/python pipeline/s04_stats.py --results-dir "$t" --quiet \
      > pipeline_out/results_arch/s04_${name}.log 2>&1 \
      && echo "    ok -> ${t}statistics.json" || echo "    FAILED (see s04_${name}.log)"
done
