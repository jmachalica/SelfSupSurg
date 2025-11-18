#!/bin/bash
set -euo pipefail

# Usage: script JOB_ID
if [ "$#" -ne 1 ]; then
  echo -e "\e[31mUsage: $0 JOB_ID\e[0m"
  exit 1
fi

JOB_ID=$1

# ---- Roots (generalized) ----
RESULTS_ROOT="/net/tscratch/people/plgjmachali/surgvu_results"
LOGS_ROOT="${RESULTS_ROOT}/logs"

# ---- Find .out / .err logs (any prefix), select newest ----
get_newest_log() {
  local pattern="$1"   # e.g. "*.out" or "*.err"
  # search for files matching "*-${JOB_ID}.out" or "*-${JOB_ID}.err"
  find "${LOGS_ROOT}" -type f -name "*-${JOB_ID}.${pattern}" -printf "%T@ %p\n" 2>/dev/null \
    | sort -nr | awk 'NR==1{print $2}'
}

OUT_LOG=$(get_newest_log "out" || true)
ERR_LOG=$(get_newest_log "err" || true)

if [ -z "${OUT_LOG:-}" ] || [ ! -f "$OUT_LOG" ]; then
  echo -e "\e[31mOutput log not found for JOB_ID=${JOB_ID} under ${LOGS_ROOT}.\e[0m"
  echo "Tried pattern: *-${JOB_ID}.out"
  exit 1
fi
if [ -z "${ERR_LOG:-}" ] || [ ! -f "$ERR_LOG" ]; then
  echo -e "\e[31mError log not found for JOB_ID=${JOB_ID} under ${LOGS_ROOT}.\e[0m"
  echo "Tried pattern: *-${JOB_ID}.err"
  exit 1
fi

echo -e "\e[34m=== Logs detected ===\e[0m"
echo "OUT_LOG: $OUT_LOG"
echo "ERR_LOG: $ERR_LOG"

echo "=== Last 10 lines of the output log ==="
tail -n 10 "$OUT_LOG" || true
echo "=== Last 10 lines of the error log ==="
tail -n 10 "$ERR_LOG" || true

echo "=== Checking for 'All done' message in the output log ==="
if grep -qi "All done" "$OUT_LOG"; then
  echo "Job $JOB_ID completed successfully."
else
  echo "Job $JOB_ID may not have completed successfully or the 'All done' message is missing.\e[0m"
fi

# ---- Find results directory (any branch), select newest ----
RESULTS_FOLDER=$(find "${RESULTS_ROOT}" -type d -name "job_${JOB_ID}_*" -printf "%T@ %p\n" 2>/dev/null \
  | sort -nr | awk 'NR==1{for (i=2;i<=NF;i++){printf "%s%s", $i, (i<NF?" ":"")}}')

if [ -z "$RESULTS_FOLDER" ]; then
  echo -e "\e[31mResults folder not found for job: $JOB_ID under ${RESULTS_ROOT}\e[0m"
  exit 1
fi

echo "=== Results folder: $RESULTS_FOLDER ==="

# ---- metrics.json (optional) ----
METRICS_FILE="${RESULTS_FOLDER}/metrics.json"
if [ -f "$METRICS_FILE" ]; then
  echo "=== Extracting key metrics from metrics.json ==="
  echo "Path to metrics.json: $METRICS_FILE"

  echo "=== Train vs Test Comparison ==="
  grep -a "train_accuracy_list_meter" "$METRICS_FILE" | tail -n 5 || true
  echo -e "\e[32m---Test---\e[0m"
  grep -a "test_accuracy_list_meter" "$METRICS_FILE" | tail -n 5 || true

 echo -e "\e[32m---Best model (prefer: test top-1 -> val top-1 -> latest checkpoint)---\e[0m"

    if ! command -v jq >/dev/null 2>&1; then
    echo -e "\e[31m'jq' not found. Please install jq to extract best model from metrics.json.\e[0m"
    else
    # 1) Try test top-1
    TEST_COUNT=$(jq -c 'select(.test_accuracy_list_meter? != null)' "$METRICS_FILE" | wc -l | tr -d ' ')
    if [ "${TEST_COUNT}" -gt 0 ]; then
        BEST_LINE=$(jq -r '
        select(.test_accuracy_list_meter? != null)
        | [ .phase_idx
            , .train_phase_idx
            , (.test_accuracy_list_meter.top_1["0"] // .test_accuracy_list_meter.top_1[0] // 0)
            ] | @tsv
        ' "$METRICS_FILE" \
        | awk 'BEGIN{best=-1}{if($3>best){best=$3; tp=$2}}END{if(best>=0)printf("best=%f\ttrain_phase=%d\n",best,tp)}')

        if [ -n "$BEST_LINE" ]; then
        BEST_ACC=$(echo "$BEST_LINE" | awk -F'\t' '{print $1}' | cut -d'=' -f2)
        BEST_TRAIN_PHASE=$(echo "$BEST_LINE" | awk -F'\t' '{print $2}' | cut -d'=' -f2)
        BEST_MODEL="model_phase${BEST_TRAIN_PHASE}.torch"
        BEST_MODEL_PATH="${RESULTS_FOLDER}/${BEST_MODEL}"
        printf "Best test top-1: %.4f%% => %s\n" "$BEST_ACC" "$BEST_MODEL"
        [ -f "$BEST_MODEL_PATH" ] && echo "Best model path: $BEST_MODEL_PATH" || echo -e "\e[33mCheckpoint not found: $BEST_MODEL_PATH\e[0m"
        FOUND_BEST=1
        fi
    fi

    # 2) If no test — try validation
    if [ -z "${FOUND_BEST:-}" ]; then
        VAL_COUNT=$(jq -c 'select(.val_accuracy_list_meter? != null)' "$METRICS_FILE" | wc -l | tr -d ' ')
        if [ "${VAL_COUNT}" -gt 0 ]; then
        BEST_LINE=$(jq -r '
            select(.val_accuracy_list_meter? != null)
            | [ .phase_idx
            , .train_phase_idx
            , (.val_accuracy_list_meter.top_1["0"] // .val_accuracy_list_meter.top_1[0] // 0)
            ] | @tsv
        ' "$METRICS_FILE" \
            | awk 'BEGIN{best=-1}{if($3>best){best=$3; tp=$2}}END{if(best>=0)printf("best=%f\ttrain_phase=%d\n",best,tp)}')

        if [ -n "$BEST_LINE" ]; then
            BEST_ACC=$(echo "$BEST_LINE" | awk -F'\t' '{print $1}' | cut -d'=' -f2)
            BEST_TRAIN_PHASE=$(echo "$BEST_LINE" | awk -F'\t' '{print $2}' | cut -d'=' -f2)
            BEST_MODEL="model_phase${BEST_TRAIN_PHASE}.torch"
            BEST_MODEL_PATH="${RESULTS_FOLDER}/${BEST_MODEL}"
            printf "Best val top-1: %.4f%% => %s\n" "$BEST_ACC" "$BEST_MODEL"
            [ -f "$BEST_MODEL_PATH" ] && echo "Best model path: $BEST_MODEL_PATH" || echo -e "\e[33mCheckpoint not found: $BEST_MODEL_PATH\e[0m"
            FOUND_BEST=1
        fi
        fi
    fi

    # 3) If no test/val in metrics.json — select newest checkpoint
    if [ -z "${FOUND_BEST:-}" ]; then
        LATEST_MODEL=$(ls -1 "${RESULTS_FOLDER}"/model_phase*.torch 2>/dev/null \
        | sed -E 's/.*model_phase([0-9]+)\.torch/\1 \0/' \
        | sort -nr | head -1 | awk '{print $2}')

        if [ -n "$LATEST_MODEL" ]; then
        echo -e "\e[33mNo test/val accuracy in metrics.json. Using latest checkpoint:\e[0m"
        echo "Latest model: $LATEST_MODEL"
        else
        echo -e "\e[33mNo checkpoints model_phase*.torch found in ${RESULTS_FOLDER}\e[0m"
        fi
    fi
    fi
else
  echo -e "\e[33mmetrics.json not found (this can be normal for some pretraining runs).\e[0m"
fi

echo -e "\e[34m=== RAM usage from monitoring/top.log ===\e[0m"

declare -a TOP_CANDIDATES
TOP_CANDIDATES+=("${RESULTS_FOLDER}/monitoring/top.log")

# SCRATCH variant
if [ -n "${SCRATCH:-}" ]; then
  while IFS= read -r p; do TOP_CANDIDATES+=("$p"); done < <(
    find "${SCRATCH}/surgvu_results" -type f -path "*/job_${JOB_ID}_*/monitoring/top.log" 2>/dev/null | head -n 3
  )
fi

# Fallback: entire results tree
while IFS= read -r p; do TOP_CANDIDATES+=("$p"); done < <(
  find "${RESULTS_ROOT}" -type f -path "*/job_${JOB_ID}_*/monitoring/top.log" 2>/dev/null | head -n 3
)

FOUND_TOP=""
for cand in "${TOP_CANDIDATES[@]:-}"; do
  if [ -f "$cand" ]; then
    FOUND_TOP="$cand"
    break
  fi
done

if [ -z "$FOUND_TOP" ]; then
  echo -e "\e[33mNo monitoring/top.log found for job ${JOB_ID}.\e[0m"
  echo "Tried candidates:"
  for c in "${TOP_CANDIDATES[@]:-}"; do echo "  - $c"; done
else
  echo "Using: $FOUND_TOP"

  # Max 'used' (MiB) z linii "MiB Mem"
  MAX_USED=$(
  grep -a "MiB Mem" "$FOUND_TOP" \
  | sed -E 's/.*,\s*([0-9.]+)\s+used,.*/\1/' \
  | awk 'BEGIN{m=0} {v=$1+0; if(v>m)m=v} END{if(m>0) printf "%.1f\n", m}'
)

# Max 'avail Mem' (MiB) – number before 'avail Mem' phrase (in MiB Swap lines)
MAX_AVAIL=$(
  grep -a "avail Mem" "$FOUND_TOP" \
  | sed -E 's/.*\s([0-9.]+)\s+avail Mem.*/\1/' \
  | awk 'BEGIN{m=0} {v=$1+0; if(v>m)m=v} END{if(m>0) printf "%.1f\n", m}'
)

  if [ -n "$MAX_USED" ]; then
    USED_GB=$(awk -v x="$MAX_USED" 'BEGIN{printf "%.1f", x/1024}')
    echo -e "Max used RAM: \e[32m${MAX_USED} MiB (~${USED_GB} GB)\e[0m"
  else
    echo -e "\e[33mCould not parse max 'used' from ${FOUND_TOP}\e[0m"
  fi

  if [ -n "$MAX_AVAIL" ]; then
    AVAIL_GB=$(awk -v x="$MAX_AVAIL" 'BEGIN{printf "%.1f", x/1024}')
    echo -e "Max avail Mem: ${MAX_AVAIL} MiB (~${AVAIL_GB} GB)"
  else
    echo -e "\e[33mCould not parse max 'avail Mem' from ${FOUND_TOP}\e[0m"
  fi
fi
# 1945257
# 1942741