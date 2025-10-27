#!/bin/bash -l
# ===================== SLURM HEADER =====================
#SBATCH -J surgvu_simclr_finetune_25
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH -p plgrid-gpu-a100
#SBATCH --output=/net/tscratch/people/plgjmachali/surgvu_results/logs/finetuning/simclr_finetuning-%j.out
#SBATCH --error=/net/tscratch/people/plgjmachali/surgvu_results/logs/finetuning/simclr_finetuning-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jmachalica@student.agh.edu.pl
# ========================================================

set -euo pipefail

############################
# CONFIGURATION - SimCLR Fine-tuning 25%
############################
export CONDA_ENV="/net/pr2/projects/plgrid/plgg_13/conda/envs/vissl_3_8"
export CFG="hparams/surgvu/finetuning/simclr_to_surgvu/25/simclr_finetuning.yaml"
export DATA_PERCENTAGE="25"

export EPOCHS=50
export BATCH_PER_GPU=256
export WORKERS=8
export GPUS=4
export TRAIN_LIMIT=-1
export TEST_LIMIT=-1

export DATA_SOURCE_ROOT="/net/tscratch/people/plgjmachali/surgvu_data_sampled_03"

# ⚠️ SIMCLR CHECKPOINT PATH - Verify this exists!
export SIMCLR_CHECKPOINT_PATH="/net/tscratch/people/plgjmachali/surgvu_results/pretraining/simclr/job_1948589_20251014_0023/model_phase150.torch"

# Verify checkpoint exists
if [ ! -f "${SIMCLR_CHECKPOINT_PATH}" ]; then
    echo "ERROR: SimCLR checkpoint not found: ${SIMCLR_CHECKPOINT_PATH}"
    echo "Available checkpoints:"
    find /net/tscratch/people/plgjmachali/surgvu_results/pretraining/simclr/ -name "*.torch" -type f | head -10
    exit 1
fi

echo "Using SimCLR checkpoint: ${SIMCLR_CHECKPOINT_PATH}"

# Source the main finetuning script functions
source "${SLURM_SUBMIT_DIR}/slurm_scripts/run_finetuning_simclr_main.sh"