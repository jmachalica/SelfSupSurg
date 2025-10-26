#!/bin/bash -l
# ===================== SLURM HEADER =====================
#SBATCH -J surgvu_test_all
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH -p plgrid-gpu-a100
#SBATCH --output=/net/tscratch/people/plgjmachali/surgvu_results/logs/testing/test_all-%j.out
#SBATCH --error=/net/tscratch/people/plgjmachali/surgvu_results/logs/testing/test_all-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jmachalica@student.agh.edu.pl
# ========================================================

set -euo pipefail

############################
# CONFIGURATION
############################
export CONDA_ENV="/net/pr2/projects/plgrid/plgg_13/conda/envs/vissl_3_8"

############################
# PATHS & PREP
############################
cd "$SLURM_SUBMIT_DIR"

DATE="$(date +'%Y%m%d_%H%M')"
: "${SCRATCH:?SCRATCH must be set (e.g., /net/tscratch/people/$USER)}"

# Base results directory with job structure
RESULTS_BASE="${SCRATCH}/surgvu_results/testing/job_${SLURM_JOB_ID}_${DATE}"
mkdir -p "${RESULTS_BASE}"

# Create logs directory
mkdir -p "${SCRATCH}/surgvu_results/logs/testing"

echo "[INFO] Job: $SLURM_JOB_NAME ($SLURM_JOB_ID)"
echo "[INFO] Node(s): $SLURM_JOB_NODELIST"
echo "[INFO] Results will be saved to: ${RESULTS_BASE}"
echo "[INFO] Starting test evaluations for all models..."

############################
# ENVIRONMENT
############################
module load Miniconda3
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

export DATE="${DATE}"

############################
# DEFINE MODEL CONFIGURATIONS
############################
# Based on detailed_metrics.csv

declare -A MODELS=(
    ["imagenet_to_surgvu,12"]="/net/tscratch/people/plgjmachali/surgvu_results/finetuning/imagenet_to_surgvu/12/job_1952602_20251015_1437/model_phase1.torch,hparams/surgvu/finetuning/imagenet_to_surgvu/12/imagenet_test_only.yaml"
    ["imagenet_to_surgvu,25"]="/net/tscratch/people/plgjmachali/surgvu_results/finetuning/imagenet_to_surgvu/25/job_1952603_20251015_1437/model_phase21.torch,hparams/surgvu/finetuning/imagenet_to_surgvu/25/imagenet_test_only.yaml"
    ["imagenet_to_surgvu,100"]="/net/tscratch/people/plgjmachali/surgvu_results/finetuning/imagenet_to_surgvu/100/job_1952604_20251015_1437/model_phase3.torch,hparams/surgvu/finetuning/imagenet_to_surgvu/100/imagenet_test_only.yaml"

    ["moco_to_surgvu,12"]="/net/tscratch/people/plgjmachali/surgvu_results/finetuning/moco_to_surgvu/12/job_1953621_20251016_1141/model_phase2.torch,hparams/surgvu/finetuning/moco_to_surgvu/12/moco_finetuning_test_only.yaml"
    ["moco_to_surgvu,25"]="/net/tscratch/people/plgjmachali/surgvu_results/finetuning/moco_to_surgvu/25/job_1953622_20251016_1141/model_phase10.torch,hparams/surgvu/finetuning/moco_to_surgvu/25/moco_finetuning_test_only.yaml"
    ["moco_to_surgvu,100"]="/net/tscratch/people/plgjmachali/surgvu_results/finetuning/moco_to_surgvu/100/job_1953627_20251016_1145/model_phase8.torch,hparams/surgvu/finetuning/moco_to_surgvu/100/moco_finetuning_test_only.yaml"

    ["simclr_to_surgvu,12"]="/net/tscratch/people/plgjmachali/surgvu_results/finetuning/simclr_to_surgvu/12/job_1952103_20251015_0322/model_phase2.torch,hparams/surgvu/finetuning/simclr_to_surgvu/12/simclr_finetuning_test_only.yaml"
    ["simclr_to_surgvu,25"]="/net/tscratch/people/plgjmachali/surgvu_results/finetuning/simclr_to_surgvu/25/job_1952104_20251015_0322/model_phase10.torch,hparams/surgvu/finetuning/simclr_to_surgvu/25/simclr_finetuning_test_only.yaml"
    ["simclr_to_surgvu,100"]="/net/tscratch/people/plgjmachali/surgvu_results/finetuning/simclr_to_surgvu/100/job_1952105_20251015_0322/model_phase2.torch,hparams/surgvu/finetuning/simclr_to_surgvu/100/simclr_finetuning_test_only.yaml"
)

############################
# HELPER FUNCTIONS
############################
run_test_evaluation() {
    local model="$1"
    local subset="$2" 
    local checkpoint_path="$3"
    local config_path="$4"
    
    echo ""
    echo "🧪 Testing ${model}/${subset}%"
    echo "   Checkpoint: ${checkpoint_path}"
    echo "   Config: ${config_path}"
    
    # Create output directory: testing/job_ID/model/subset
    local output_dir="${RESULTS_BASE}/${model}/${subset}"
    mkdir -p "${output_dir}"
    
    # Verify checkpoint exists
    if [ ! -f "${checkpoint_path}" ]; then
        echo "   ❌ Checkpoint not found: ${checkpoint_path}"
        echo "   Skipping ${model}/${subset}"
        return 1
    fi
    
    # Verify config exists
    if [ ! -f "${SLURM_SUBMIT_DIR}/configs/config/${config_path}" ]; then
        echo "   ❌ Config not found: ${config_path}"
        echo "   Skipping ${model}/${subset}"
        return 1
    fi
    
    # Set Python warnings
    export PYTHONWARNINGS="default,\
ignore::UserWarning:torchvision.transforms._functional_video,\
ignore::UserWarning:torchvision.transforms._transforms_video,\
ignore:.*fvcore version of PathManager.*:UserWarning,\
ignore:.*Please migrate to the version in iopath repo.*:UserWarning"
    
    # Run test evaluation
    echo "   🚀 Starting evaluation..."
    export RUN_DIR="${output_dir}" # for f1 metrics

    if python main.py -hp "${config_path}" -m supervised \
        config.SLURM.USE_SLURM=false \
        hydra.verbose=true \
        hydra.job_logging.root.level=INFO \
        config.DISTRIBUTED.NUM_PROC_PER_NODE=1 \
        config.DATA.TEST.BATCHSIZE_PER_REPLICA=256 \
        config.DATA.NUM_DATALOADER_WORKERS=8 \
        config.LOG_FREQUENCY=10 \
        "config.MODEL.WEIGHTS_INIT.PARAMS_FILE=${checkpoint_path}" \
        "config.RUN_DIR=${output_dir}" \
        "config.CHECKPOINT.DIR=${output_dir}" \
        config.OPTIMIZER.num_epochs=1 \
        "config.HOOKS.TENSORBOARD_SETUP.EXPERIMENT_LOG_DIR=${output_dir}/tb_logs"; then
        
        echo "   ✅ ${model}/${subset} completed successfully"
        
        # Create summary file
        cat > "${output_dir}/test_summary.txt" << EOF
Test Evaluation Summary
======================
Model: ${model}
Subset: ${subset}%
Checkpoint: ${checkpoint_path}
Config: ${config_path}
Job ID: ${SLURM_JOB_ID}
Date: $(date)
Output Directory: ${output_dir}

Metrics file: ${output_dir}/metrics.json
Tensorboard logs: ${output_dir}/tb_logs
EOF
        
        # Log successful completion
        echo "${model}/${subset}: SUCCESS" >> "${RESULTS_BASE}/completion_log.txt"
        
    else
        echo "   ❌ ${model}/${subset} failed"
        echo "${model}/${subset}: FAILED" >> "${RESULTS_BASE}/completion_log.txt"
        return 1
    fi
}

############################
# MAIN EXECUTION
############################

echo ""
echo "📊 Testing configuration loaded for ${#MODELS[@]} model/subset combinations"

# Initialize completion log
echo "Test Evaluation Completion Log - Job ${SLURM_JOB_ID}" > "${RESULTS_BASE}/completion_log.txt"
echo "Started: $(date)" >> "${RESULTS_BASE}/completion_log.txt"
echo "Results Directory: ${RESULTS_BASE}" >> "${RESULTS_BASE}/completion_log.txt"
echo "" >> "${RESULTS_BASE}/completion_log.txt"

# Counter for progress
total_tests=${#MODELS[@]}
current_test=0

# Run tests for each model/subset
for key in "${!MODELS[@]}"; do
    current_test=$((current_test + 1))
    
    # Parse key and values
    IFS=',' read -r model subset <<< "$key"
    IFS=',' read -r checkpoint_path config_path <<< "${MODELS[$key]}"
    
    echo ""
    echo "==============================================="
    echo "Progress: ${current_test}/${total_tests}"
    
    # Run the test
    run_test_evaluation "$model" "$subset" "$checkpoint_path" "$config_path" || true
done

############################
# FINAL SUMMARY
############################

echo ""
echo "==============================================="
echo "🏁 All test evaluations completed!"
echo ""

# Count successes and failures
success_count=$(grep -c "SUCCESS" "${RESULTS_BASE}/completion_log.txt" || true)
failure_count=$(grep -c "FAILED" "${RESULTS_BASE}/completion_log.txt" || true)

echo "📊 Summary:"
echo "   Total tests: ${total_tests}"
echo "   Successful: ${success_count}"
echo "   Failed: ${failure_count}"

if [ "$failure_count" -gt 0 ]; then
    echo ""
    echo "❌ Failed tests:"
    grep "FAILED" "${RESULTS_BASE}/completion_log.txt" || true
fi

echo ""
echo "📁 Results directory: ${RESULTS_BASE}"
echo "📄 Completion log: ${RESULTS_BASE}/completion_log.txt"

# Add completion timestamp to log
echo "" >> "${RESULTS_BASE}/completion_log.txt"
echo "Completed: $(date)" >> "${RESULTS_BASE}/completion_log.txt"
echo "Summary: ${success_count}/${total_tests} successful" >> "${RESULTS_BASE}/completion_log.txt"

echo ""
echo "✨ Job completed!"

# Exit with appropriate code
if [ "$failure_count" -eq 0 ]; then
    exit 0
else
    exit 1
fi