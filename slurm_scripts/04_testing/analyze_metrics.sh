#!/bin/bash
set -euo pipefail

# Usage: ./summarize_best_tests.sh /path/to/surgvu_results
RESULTS_ROOT="${1:-/net/tscratch/people/plgjmachali/surgvu_results}"

# Filter date (modify as needed)
FILTER_DATE="2025-10-11"

command -v jq >/dev/null 2>&1 || {
  echo "ERROR: 'jq' command not found. Please install jq." >&2
  exit 1
}

results=()

while IFS= read -r METRICS; do
  DIR="$(dirname "$METRICS")"

  JOB_ID=$(echo "$DIR" | sed -nE 's/.*job_([0-9]+)_.*/\1/p')
  [ -z "$JOB_ID" ] && JOB_ID="UNKNOWN"

  BEST_LINE=$(
    jq -r '
      select(.test_accuracy_list_meter? != null)
      | [ (.train_phase_idx // 0)
        , (.test_accuracy_list_meter.top_1["0"] // .test_accuracy_list_meter.top_1[0] // 0)
        ] | @tsv
    ' "$METRICS" 2>/dev/null \
    | awk 'BEGIN{best=-1} {if($2>best){best=$2; tp=$1}} END{if(best>=0) printf "%s\t%f\n", tp, best}'
  )

  [ -z "$BEST_LINE" ] && continue

  BEST_PHASE=$(echo "$BEST_LINE" | awk -F'\t' '{print $1}')
  BEST_ACC=$(echo "$BEST_LINE" | awk -F'\t' '{print $2}')
  BEST_ACC_ROUND=$(awk -v x="$BEST_ACC" 'BEGIN{printf "%.4f", x}')

  BEST_MODEL_PATH="${DIR}/model_phase${BEST_PHASE}.torch"
  [ -f "$BEST_MODEL_PATH" ] || BEST_MODEL_PATH="(checkpoint not found)"

  # ✅ Basename of checkpoint
  BEST_MODEL_BASE=$(basename "$BEST_MODEL_PATH")

  results+=("$BEST_ACC|$BEST_ACC_ROUND|$JOB_ID|$DIR|$BEST_MODEL_BASE")
done < <(find "$RESULTS_ROOT" -type f -name metrics.json -newermt "$FILTER_DATE" 2>/dev/null | sort)

# Print only if results exist
if ((${#results[@]})); then
  printf "%s\n" "${results[@]}" \
    | sort -t'|' -nrk1,1 \
    | awk -F'|' '{printf "%-8s  %-10s  %s  [%s]\n", $2, $3, $4, $5}'
fi