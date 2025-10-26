#!/bin/bash -l
# ===================== SLURM HEADER =====================
#SBATCH -J surgvu_simclr_pretrain
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH -p plgrid-gpu-a100
#SBATCH --output=/net/tscratch/people/plgjmachali/surgvu_results/logs/pretraining/simclr_output-%j.out
#SBATCH --error=/net/tscratch/people/plgjmachali/surgvu_results/logs/pretraining/simclr_error-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jmachalica@student.agh.edu.pl
# ========================================================

set -euo pipefail

############################
# CONFIGURATION - SimCLR Pre-training
############################
export CONDA_ENV="/net/pr2/projects/plgrid/plgg_13/conda/envs/vissl_3_8"
export CFG="hparams/surgvu/pre_training/simclr.yaml"

export EPOCHS=150
export BATCH_PER_GPU=256
export WORKERS=8
export GPUS=4
export TRAIN_LIMIT=-1

export DATA_SOURCE_ROOT="/net/tscratch/people/plgjmachali/surgvu_data_sampled_03"

# Source the main SimCLR pre-training script functions
source "${SLURM_SUBMIT_DIR}/slurm_scripts/run_simclr_pretraining_main.sh"