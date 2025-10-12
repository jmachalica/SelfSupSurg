#!/bin/bash

# Check if the job ID is provided
if [ "$#" -ne 1 ]; then
  echo -e "\e[31mUsage: $0 JOB_ID\e[0m"
  exit 1
fi

JOB_ID=$1
LOG_DIR="/net/tscratch/people/plgjmachali/surgvu_results/logs/finetuning"
RESULTS_DIR="/net/tscratch/people/plgjmachali/surgvu_results/finetuning/imagenet_to_surgvu"

# Find the logs
OUT_LOG="${LOG_DIR}/finetuning-${JOB_ID}.out"
ERR_LOG="${LOG_DIR}/finetuning-${JOB_ID}.err"

if [ ! -f "$OUT_LOG" ]; then
  echo -e "\e[31mOutput log not found: $OUT_LOG\e[0m"
  exit 1
fi

if [ ! -f "$ERR_LOG" ]; then
  echo -e "\e[31mError log not found: $ERR_LOG\e[0m"
  exit 1
fi

# Display the last 10 lines of the logs
echo "=== Last 10 lines of the output log ==="
tail -n 10 "$OUT_LOG"

echo "=== Last 10 lines of the error log ==="
tail -n 10 "$ERR_LOG"

# Check if the output log contains the "All done" message
echo "=== Checking for 'All done' message in the output log ==="
if grep -q -i "All done" "$OUT_LOG"; then
  echo "Job $JOB_ID completed successfully."
else
  echo -e "\e[31mJob $JOB_ID did not complete successfully or the 'All done' message is missing.\e[0m"
fi

# Find the results folder
RESULTS_FOLDER=$(find "$RESULTS_DIR" -type d -name "job_${JOB_ID}_*" | head -n 1)

if [ -z "$RESULTS_FOLDER" ]; then
  echo -e "\e[31mResults folder not found for job: $JOB_ID\e[0m"
  exit 1
fi

echo "=== Results folder: $RESULTS_FOLDER ==="

# Extract metrics from metrics.json
METRICS_FILE="${RESULTS_FOLDER}/metrics.json"

# Check if the metrics file exists
if [ ! -f "$METRICS_FILE" ]; then
  echo -e "\e[31mmetrics.json file not found in the results folder.\e[0m"
  exit 1
else
  echo "=== Extracting key metrics from metrics.json ==="
  echo "Path to metrics.json: $METRICS_FILE"
fi


# Compare train vs test
echo "=== Train vs Test Comparison ==="
grep train_accuracy_list_meter "$METRICS_FILE" | tail -n 5
echo -e "\e[32m---Test---\e[0m"
grep test_accuracy_list_meter "$METRICS_FILE" | tail -n 5

echo -e "\e[32m---Best model---\e[0m"
grep '"test_accuracy_list_meter"' $METRICS_FILE \
| jq -r '[.phase_idx, .train_phase_idx, .test_accuracy_list_meter.top_1["0"]] | @tsv' \
| awk 'BEGIN{best=-1;model=""} {if ($3>best){best=$3; model=sprintf("model_phase%d.torch", $2)}} END{print "Best test accuracy:", best"%", "=>", model}'
