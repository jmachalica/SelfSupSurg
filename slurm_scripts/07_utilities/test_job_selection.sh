#!/bin/bash

# Test script to show which job would be selected as "latest" based on name sorting

SOURCE_DIR="/net/tscratch/people/plgjmachali/surgvu_results/finetuning"
MODELS=("imagenet_to_surgvu" "moco_to_surgvu" "simclr_to_surgvu")
DATA_SUBSETS=("12" "25" "100")

echo "🔍 Testing job selection logic..."
echo ""

for model in "${MODELS[@]}"; do
    for subset in "${DATA_SUBSETS[@]}"; do
        subset_dir="${SOURCE_DIR}/${model}/${subset}"
        
        if [ ! -d "${subset_dir}" ]; then
            continue
        fi
        
        echo "📂 ${model}/${subset}:"
        
        # Get all job directories
        job_dirs=()
        while IFS= read -r -d '' dir; do
            job_dirs+=("$(basename "$dir")")
        done < <(find "${subset_dir}" -maxdepth 1 -type d -name "job_*" -print0 2>/dev/null)
        
        if [ ${#job_dirs[@]} -eq 0 ]; then
            echo "   ❌ No jobs found"
            continue
        fi

        # Filter out backup directories
        filtered_jobs=()
        for job in "${job_dirs[@]}"; do
            if [[ "$job" != *_backup ]]; then
                filtered_jobs+=("$job")
            fi
        done

        if [ ${#filtered_jobs[@]} -eq 0 ]; then
            echo "   ❌ No non-backup jobs found"
            continue
        fi
        
        # Sort jobs
        IFS=$'\n' sorted_jobs=($(sort <<<"${filtered_jobs[*]}"))
        unset IFS
        
        echo "   Total jobs: ${#job_dirs[@]} (${#filtered_jobs[@]} non-backup)"
        
        # Show all jobs with indicators
        for job in "${sorted_jobs[@]}"; do
            if [ "$job" = "${sorted_jobs[-1]}" ]; then
                echo "   ✅ $job (WOULD BE SELECTED)"
            else
                echo "   → $job"
            fi
        done
        echo ""
    done
done

echo "✨ Test completed. Run copy_latest_results.sh to proceed with actual copying."