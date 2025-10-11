#!/bin/bash -l
# ===================== SLURM HEADER =====================
#SBATCH -J surgvu_moco_pretrain
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:8
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH -p plgrid-gpu-a100
#SBATCH --output=/net/tscratch/people/plgjmachali/surgvu_results/logs/pretraining/moco_output-%j.out
#SBATCH --error=/net/tscratch/people/plgjmachali/surgvu_results/logs/pretraining/moco_error-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jmachalica@student.agh.edu.pl
# ========================================================

set -euo pipefail

############################
# CONFIGURATION - MoCo Pre-training
############################
export CONDA_ENV="/net/pr2/projects/plgrid/plgg_13/conda/envs/vissl_3_8"
export CFG="hparams/surgvu/pre_training/moco.yaml"

export EPOCHS=100
export BATCH_PER_GPU=128
export WORKERS=8
export GPUS=8  # MoCo config has NUM_PROC_PER_NODE: 2
export TRAIN_LIMIT=-1

export DATA_SOURCE_ROOT="/net/tscratch/people/plgjmachali/surgvu_data_sampled_03"

# Source the main MoCo pre-training script functions
source "${SLURM_SUBMIT_DIR}/slurm_scripts/run_moco_pretraining_main.sh"