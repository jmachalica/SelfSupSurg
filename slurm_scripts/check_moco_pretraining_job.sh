#!/bin/bash
# ========================================================
# Check MoCo pre-training job status
# Usage: ./check_moco_pretraining_job.sh <JOB_ID>
# Example: ./check_moco_pretraining_job.sh 1942563
# ========================================================

if [ $# -eq 0 ]; then
    echo "Usage: $0 <JOB_ID>"
    echo "Example: $0 1942563"
    exit 1
fi

JOB_ID="$1"
USER="plgjmachali"

echo "=========================================="
echo "🔍 CHECKING MOCO PRE-TRAINING JOB: $JOB_ID"
echo "=========================================="

############################
# 1. JOB STATUS
############################
echo
echo "📊 JOB STATUS:"
echo "----------------------------------------"
sacct -j "$JOB_ID" --format=JobID,JobName,State,ExitCode,Start,End,Elapsed,MaxRSS,MaxVMSize,ReqMem,ReqCPUS,ReqTRES,NodeList -P | column -t -s '|'

############################
# 2. SLURM LOGS
############################
echo
echo "📄 SLURM LOGS:"
echo "----------------------------------------"
OUTPUT_LOG="/net/tscratch/people/$USER/surgvu_results/logs/pretraining/moco_output-${JOB_ID}.out"
ERROR_LOG="/net/tscratch/people/$USER/surgvu_results/logs/pretraining/moco_error-${JOB_ID}.err"

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
# 3. FIND RESULTS DIRECTORY
############################
echo
echo "📁 RESULTS DIRECTORY:"
echo "----------------------------------------"
RESULTS_BASE="/net/tscratch/people/$USER/surgvu_results/pretraining/moco"

# Find job directory
JOB_DIRS=$(find "$RESULTS_BASE" -type d -name "*job_${JOB_ID}_*" 2>/dev/null)
if [ -n "$JOB_DIRS" ]; then
    echo "✅ Found MoCo pre-training results directory:"
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
        
        # Look for MoCo checkpoints
        CHECKPOINTS=$(find "$dir" -name "*.torch" -o -name "checkpoint.torch" 2>/dev/null)
        if [ -n "$CHECKPOINTS" ]; then
            echo "   ✅ Checkpoints available:"
            echo "$CHECKPOINTS" | sed 's/^/     /'
        else
            echo "   ❌ No checkpoints found"
        fi
        
        # Check monitoring
        if [ -d "$dir/monitoring" ]; then
            echo "   ✅ Monitoring directory exists:"
            ls -la "$dir/monitoring/" 2>/dev/null | sed 's/^/     /'
        else
            echo "   ❌ Monitoring directory missing"
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
        
        # Show training progress
        if [ -f "$dir/log.txt" ]; then
            echo
            echo "📈 TRAINING PROGRESS:"
            echo "   Total log lines: $(wc -l < "$dir/log.txt")"
            
            # Look for epoch information
            EPOCHS=$(grep -i "epoch" "$dir/log.txt" | tail -5 | sed 's/^/   /')
            if [ -n "$EPOCHS" ]; then
                echo "   Recent epoch info:"
                echo "$EPOCHS"
            fi
        fi
    done
else
    echo "❌ No MoCo pre-training results directory found for job $JOB_ID"
    echo "   Searched in: $RESULTS_BASE"
fi

echo
echo "=========================================="
echo "✅ MOCO PRE-TRAINING JOB CHECK COMPLETE"
echo "=========================================="