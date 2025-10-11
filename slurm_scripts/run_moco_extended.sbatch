#!/bin/bash -l
# ===================== SLURM HEADER =====================
#SBATCH -J surgvu_moco_extended
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH -p plgrid-gpu-a100
#SBATCH --output=/net/tscratch/people/plgjmachali/surgvu_results/logs/pretraining/moco_extended_output-%j.out
#SBATCH --error=/net/tscratch/people/plgjmachali/surgvu_results/logs/pretraining/moco_extended_error-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jmachalica@student.agh.edu.pl
# ========================================================

set -euo pipefail

############################
# CONFIGURATION - Extended MoCo Pre-training
############################
export CONDA_ENV="/net/pr2/projects/plgrid/plgg_13/conda/envs/vissl_3_8"
export CFG="hparams/surgvu/pre_training/moco_extended.yaml"

export EPOCHS=200  # Extended from 100 to 200 total epochs
export BATCH_PER_GPU=256  # 4×256=1024 (same global batch as 8×128)
export WORKERS=8  # Keep per GPU
export GPUS=4  # Better resource availability
export TRAIN_LIMIT=-1

export DATA_SOURCE_ROOT="/net/tscratch/people/plgjmachali/surgvu_data_sampled_03"

# ⚠️ CHECKPOINT PATH - Update this to your actual checkpoint!
export MOCO_CHECKPOINT_PATH="/net/tscratch/people/plgjmachali/surgvu_results/pretraining/moco/job_1942741_20251010_0629/model_final_checkpoint_phase99.torch"

# Verify checkpoint exists
if [ ! -f "${MOCO_CHECKPOINT_PATH}" ]; then
    echo "ERROR: Checkpoint not found: ${MOCO_CHECKPOINT_PATH}"
    echo "Available checkpoints:"
    find /net/tscratch/people/plgjmachali/surgvu_results/pretraining/moco/ -name "*.torch" -type f | head -10
    exit 1
fi

echo "Will resume from checkpoint: ${MOCO_CHECKPOINT_PATH}"

# Source the main extended MoCo script functions
source "${SLURM_SUBMIT_DIR}/slurm_scripts/run_moco_extended_main.sh"