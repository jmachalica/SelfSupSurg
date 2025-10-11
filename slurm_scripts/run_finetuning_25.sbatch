#!/bin/bash -l
# ===================== SLURM HEADER =====================
#SBATCH -J surgvu_finetune_25
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH -p plgrid-gpu-a100
#SBATCH --output=/net/tscratch/people/plgjmachali/surgvu_results/logs/finetuning/finetuning-%j.out
#SBATCH --error=/net/tscratch/people/plgjmachali/surgvu_results/logs/finetuning/finetuning-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jmachalica@student.agh.edu.pl
# ========================================================

set -euo pipefail

############################
# CONFIGURATION - 25%
############################
export CONDA_ENV="/net/pr2/projects/plgrid/plgg_13/conda/envs/vissl_3_8"
export CFG="hparams/surgvu/finetuning/imagenet_to_surgvu/25/imagenet_fully_supervised.yaml"
export DATA_PERCENTAGE="25"

export EPOCHS=100
export BATCH_PER_GPU=256
export WORKERS=8  # Test: więcej workers per GPU (4×12=48 total)
export GPUS=4
export TRAIN_LIMIT=-1
export VAL_LIMIT=-1
export TEST_LIMIT=-1

export DATA_SOURCE_ROOT="/net/tscratch/people/plgjmachali/surgvu_data_sampled_03"


# Source the main finetuning script functions
source "${SLURM_SUBMIT_DIR}/slurm_scripts/run_finetuning_main.sh"