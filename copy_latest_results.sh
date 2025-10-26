#!/bin/bash

# Script to copy the latest job directories from each model/subset combination
# Usage: ./copy_latest_results.sh

set -euo pipefail

# Source and destination directories
SOURCE_DIR="/net/tscratch/people/plgjmachali/surgvu_results/finetuning"
DEST_DIR="/net/pr2/projects/plgrid/plgg_13/results"

# Models and data subsets
MODELS=("imagenet_to_surgvu" "moco_to_surgvu" "simclr_to_surgvu")
DATA_SUBSETS=("12" "25" "100")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Starting copy of latest job results...${NC}"
echo -e "${BLUE}Source: ${SOURCE_DIR}${NC}"
echo -e "${BLUE}Destination: ${DEST_DIR}${NC}"

# Create destination directory if it doesn't exist
mkdir -p "${DEST_DIR}"

# Counter for processed jobs
processed_count=0
skipped_count=0

# Iterate through all model/subset combinations
for model in "${MODELS[@]}"; do
    for subset in "${DATA_SUBSETS[@]}"; do
        echo ""
        echo -e "${YELLOW}🔍 Processing: ${model}/${subset}${NC}"
        
        subset_dir="${SOURCE_DIR}/${model}/${subset}"
        
        # Check if source directory exists
        if [ ! -d "${subset_dir}" ]; then
            echo -e "${RED}❌ Directory does not exist: ${subset_dir}${NC}"
            ((skipped_count++))
            continue
        fi
        
        # Find all job directories (sorted by name to get the latest)
        job_dirs=()
        while IFS= read -r -d '' dir; do
            job_dirs+=("$(basename "$dir")")
        done < <(find "${subset_dir}" -maxdepth 1 -type d -name "job_*" -print0 2>/dev/null)
        
        if [ ${#job_dirs[@]} -eq 0 ]; then
            echo -e "${RED}❌ No job directories found in ${subset_dir}${NC}"
            ((skipped_count++))
            continue
        fi
        
        # Sort job directories by name (latest will be last)
        # This works because:
        # - job_ID format comes first
        # - job_ID_YYYYMMDD_HHMM format comes after, sorted by date/time
        # - backup jobs are excluded
        
        filtered_jobs=()
        for job in "${job_dirs[@]}"; do
            # Skip backup jobs
            if [[ "$job" == *"_backup"* ]]; then
                continue
            fi
            filtered_jobs+=("$job")
        done
        
        if [ ${#filtered_jobs[@]} -eq 0 ]; then
            echo -e "${RED}❌ No valid job directories found (all are backups)${NC}"
            ((skipped_count++))
            continue
        fi
        
        # Sort and get the latest (last in sorted order)
        IFS=$'\n' sorted_jobs=($(sort <<<"${filtered_jobs[*]}"))
        unset IFS
        
        newest_job_name="${sorted_jobs[-1]}"
        newest_job_dir="${subset_dir}/${newest_job_name}"
        
        job_name="$newest_job_name"
        echo -e "${GREEN}📁 Found newest job: ${job_name}${NC}"
        echo -e "${BLUE}📋 Available jobs: ${#filtered_jobs[@]} total (showing last 3)${NC}"
        
        # Show last 3 jobs for context
        start_idx=$((${#sorted_jobs[@]} - 3))
        if [ $start_idx -lt 0 ]; then start_idx=0; fi
        
        for ((i=start_idx; i<${#sorted_jobs[@]}; i++)); do
            if [ "$i" -eq $((${#sorted_jobs[@]} - 1)) ]; then
                echo -e "   ${GREEN}→ ${sorted_jobs[i]} (SELECTED)${NC}"
            else
                echo -e "   → ${sorted_jobs[i]}"
            fi
        done
        
        # Create destination structure
        dest_model_dir="${DEST_DIR}/${model}/${subset}"
        mkdir -p "${dest_model_dir}"
        
        dest_job_dir="${dest_model_dir}/${job_name}"
        
        # Check if destination already exists
        if [ -d "$dest_job_dir" ]; then
            echo -e "${YELLOW}⚠️  Destination already exists: ${dest_job_dir}${NC}"
            exit 1
            # read -p "Overwrite? [y/N]: " -n 1 -r
            # echo ""
            # if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            #     echo -e "${YELLOW}⏭️  Skipping${NC}"
            #     ((skipped_count++))
            #     continue
            # fi
            # echo -e "${YELLOW}🗑️  Removing existing directory${NC}"
            # rm -rf "$dest_job_dir"
        fi
        
        # Copy the directory
        echo -e "${BLUE}📦 Copying ${newest_job_dir} -> ${dest_job_dir}${NC}"
        
        # Show source directory size
        source_size=$(du -sh "$newest_job_dir" 2>/dev/null | cut -f1 || echo "unknown")
        echo -e "${BLUE}📊 Source size: ${source_size}${NC}"
        
        # Copy with progress
        if command -v rsync >/dev/null 2>&1; then
            # Use rsync if available (shows progress)
            rsync -ah --progress "$newest_job_dir/" "$dest_job_dir/"
        else
            # Fallback to cp
            cp -r "$newest_job_dir" "$dest_model_dir/"
        fi
        
        # Verify copy
        if [ -d "$dest_job_dir" ]; then
            dest_size=$(du -sh "$dest_job_dir" 2>/dev/null | cut -f1 || echo "unknown")
            echo -e "${GREEN}✅ Successfully copied (${dest_size})${NC}"
            ((processed_count++))
            
            # List key files
            echo -e "${BLUE}📋 Key files in destination:${NC}"
            if [ -f "${dest_job_dir}/metrics.json" ]; then
                lines=$(wc -l < "${dest_job_dir}/metrics.json" 2>/dev/null || echo "0")
                echo -e "   📊 metrics.json (${lines} lines)"
            fi
            
            checkpoint_count=$(find "${dest_job_dir}" -name "*.torch" -o -name "*.pth" | wc -l)
            if [ "$checkpoint_count" -gt 0 ]; then
                echo -e "   🎯 ${checkpoint_count} checkpoint file(s)"
            fi
            
            if [ -d "${dest_job_dir}/logs" ] || [ -d "${dest_job_dir}/tb_logs" ]; then
                echo -e "   📝 Log directories found"
            fi
        else
            echo -e "${RED}❌ Copy failed${NC}"
            ((skipped_count++))
        fi
    done
done

echo ""
echo -e "${GREEN}🎉 Copy operation completed!${NC}"
echo -e "${GREEN}📊 Processed: ${processed_count} jobs${NC}"
echo -e "${YELLOW}⏭️  Skipped: ${skipped_count} jobs${NC}"
echo -e "${BLUE}📁 Results location: ${DEST_DIR}${NC}"

# Show final directory structure
echo ""
echo -e "${BLUE}📂 Final directory structure:${NC}"
if command -v tree >/dev/null 2>&1; then
    tree -L 3 "$DEST_DIR" 2>/dev/null || find "$DEST_DIR" -type d | head -20
else
    find "$DEST_DIR" -type d | head -20
fi

echo ""
echo -e "${GREEN}✨ You can now run the analysis script on the copied data:${NC}"
echo -e "${BLUE}   cd $(dirname "$DEST_DIR")${NC}"
echo -e "${BLUE}   # Update analyze_results.py to use: RESULTS_DIR = \"${DEST_DIR}\"${NC}"