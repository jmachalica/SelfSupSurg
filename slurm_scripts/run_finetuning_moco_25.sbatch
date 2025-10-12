#!/bin/bash -l
# ===================== SLURM HEADER =====================
#SBATCH -J surgvu_moco_finetune_25
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH -p plgrid-gpu-a100
#SBATCH --output=/net/tscratch/people/plgjmachali/surgvu_results/logs/finetuning/moco_finetuning-%j.out
#SBATCH --error=/net/tscratch/people/plgjmachali/surgvu_results/logs/finetuning/moco_finetuning-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jmachalica@student.agh.edu.pl
# ========================================================

set -euo pipefail

############################
# CONFIGURATION - MoCo Fine-tuning 25%
############################
export CONDA_ENV="/net/pr2/projects/plgrid/plgg_13/conda/envs/vissl_3_8"
export CFG="hparams/surgvu/finetuning/moco_to_surgvu/25/moco_finetuning.yaml"
export DATA_PERCENTAGE="25"

export EPOCHS=50
export BATCH_PER_GPU=256
export WORKERS=8
export GPUS=4
export TRAIN_LIMIT=-1
export VAL_LIMIT=-1
export TEST_LIMIT=-1

export DATA_SOURCE_ROOT="/net/tscratch/people/plgjmachali/surgvu_data_sampled_03"

# ⚠️ MOCO CHECKPOINT PATH - Verify this exists!
export MOCO_CHECKPOINT_PATH="/net/tscratch/people/plgjmachali/surgvu_results/pretraining/moco/job_1942741_20251010_0629/model_final_checkpoint_phase99.torch"

# Verify checkpoint exists
if [ ! -f "${MOCO_CHECKPOINT_PATH}" ]; then
    echo "ERROR: MoCo checkpoint not found: ${MOCO_CHECKPOINT_PATH}"
    echo "Available checkpoints:"
    find /net/tscratch/people/plgjmachali/surgvu_results/pretraining/moco/ -name "*.torch" -type f | head -10
    exit 1
fi

echo "Using MoCo checkpoint: ${MOCO_CHECKPOINT_PATH}"

# Source the main finetuning script functions
source "${SLURM_SUBMIT_DIR}/slurm_scripts/run_finetuning_moco_main.sh"