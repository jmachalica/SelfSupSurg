#!/bin/bash
# filepath: /net/pr2/projects/plgrid/plgg_13/sources/SelfSupSurg/check_job.sh
# Usage: ./check_job.sh <JOB_ID>
# Example: ./check_job.sh 1942563

if [ $# -eq 0 ]; then
    echo "Usage: $0 <JOB_ID>"
    echo "Example: $0 1942563"
    exit 1
fi

JOB_ID="$1"
USER="plgjmachali"

echo "=========================================="
echo "🔍 CHECKING JOB: $JOB_ID"
echo "=========================================="

############################
# 1. JOB STATUS
############################
echo
echo "📊 JOB STATUS:"
echo "----------------------------------------"
sacct -j "$JOB_ID" --format=JobID,JobName,State,ExitCode,Start,End,Elapsed,MaxRSS,MaxVMSize,ReqMem,ReqCPUS,ReqGRES,NodeList -P | column -t -s '|'

############################
# 2. JOB DETAILS
############################
echo
echo "📋 JOB DETAILS:"
echo "----------------------------------------"
scontrol show job "$JOB_ID" 2>/dev/null || echo "Job $JOB_ID not found in current queue (might be completed)"

############################
# 3. SLURM LOGS
############################
echo
echo "📄 SLURM LOGS:"
echo "----------------------------------------"
OUTPUT_LOG="/net/tscratch/people/$USER/surgvu_results/logs/finetuning/output-${JOB_ID}.out"
ERROR_LOG="/net/tscratch/people/$USER/surgvu_results/logs/finetuning/error-${JOB_ID}.err"

echo "Output log: $OUTPUT_LOG"
if [ -f "$OUTPUT_LOG" ]; then
    echo "✅ Output log exists ($(wc -l < "$OUTPUT_LOG") lines)"
    echo "📄 Last 10 lines:"
    tail -10 "$OUTPUT_LOG"
else
    echo "❌ Output log not found"
fi

echo
echo "Error log: $ERROR_LOG"
if [ -f "$ERROR_LOG" ]; then
    if [ -s "$ERROR_LOG" ]; then
        echo "⚠️ Error log has content ($(wc -l < "$ERROR_LOG") lines):"
        tail -10 "$ERROR_LOG"
    else
        echo "✅ Error log is empty (no errors)"
    fi
else
    echo "❌ Error log not found"
fi

############################
# 4. FIND RESULTS DIRECTORY
############################
echo
echo "📁 RESULTS DIRECTORY:"
echo "----------------------------------------"
RESULTS_BASE="/net/tscratch/people/$USER/surgvu_results/finetuning/imagenet_to_surgvu"

# Find job directory
JOB_DIRS=$(find "$RESULTS_BASE" -type d -name "*job_${JOB_ID}_*" 2>/dev/null)
if [ -n "$JOB_DIRS" ]; then
    echo "✅ Found results directory:"
    for dir in $JOB_DIRS; do
        echo "   $dir"
        
        echo
        echo "📊 Directory contents:"
        ls -lah "$dir" 2>/dev/null || echo "   Cannot access directory"
        
        # Check key files
        echo
        echo "🔍 Key files:"
        [ -f "$dir/log.txt" ] && echo "   ✅ Training log: log.txt" || echo "   ❌ Training log missing"
        [ -f "$dir/metrics.json" ] && echo "   ✅ Metrics: metrics.json" || echo "   ❌ Metrics missing"
        [ -f "$dir/model_final_checkpoint_phase0.torch" ] && echo "   ✅ Final checkpoint available" || echo "   ❌ Final checkpoint missing"
        
        # Check monitoring
        if [ -d "$dir/monitoring" ]; then
            echo "   ✅ Monitoring directory exists:"
            ls -la "$dir/monitoring/" 2>/dev/null | sed 's/^/     /'
        else
            echo "   ❌ Monitoring directory missing"
        fi
        
        # Show final metrics if available
        if [ -f "$dir/metrics.json" ]; then
            echo
            echo "📈 FINAL METRICS:"
            echo "   Test accuracy: $(grep -o '"accuracy_list_meter":\[[^]]*\]' "$dir/metrics.json" | tail -1 || echo "Not found")"
            echo "   Train accuracy: $(grep -o '"top1":[0-9.]*' "$dir/metrics.json" | tail -1 || echo "Not found")"
        fi
        
        # GPU monitoring summary
        if [ -f "$dir/monitoring/gpu.csv" ]; then
            echo
            echo "🖥️ GPU USAGE SUMMARY:"
            if [ -s "$dir/monitoring/gpu.csv" ]; then
                echo "   Max GPU utilization: $(tail -n +2 "$dir/monitoring/gpu.csv" | cut -d, -f3 | sort -n | tail -1 || echo "N/A")%"
                echo "   Max memory used: $(tail -n +2 "$dir/monitoring/gpu.csv" | cut -d, -f5 | sort -n | tail -1 || echo "N/A") MB"
            else
                echo "   ❌ GPU monitoring file is empty"
            fi
        fi
    done
else
    echo "❌ No results directory found for job $JOB_ID"
    echo "   Searched in: $RESULTS_BASE"
fi

echo
echo "=========================================="
echo "✅ JOB CHECK COMPLETE"
echo "=========================================="