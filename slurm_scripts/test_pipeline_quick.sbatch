#!/bin/bash -l
# ===================== QUICK PIPELINE TEST =====================
#SBATCH -J test_pipeline_quick
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH -p plgrid-gpu-a100
#SBATCH --output=/net/tscratch/people/plgjmachali/temp_test/pipeline_test-%j.out
#SBATCH --error=/net/tscratch/people/plgjmachali/temp_test/pipeline_test-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jmachalica@student.agh.edu.pl
# ========================================================

set -euo pipefail

echo "🧪 [PIPELINE TEST] Starting quick validation..."

############################
# MINIMAL CONFIG - TESTING
############################
export CONDA_ENV="/net/pr2/projects/plgrid/plgg_13/conda/envs/vissl_3_8"
export CFG="hparams/surgvu/finetuning/imagenet_to_surgvu/100/imagenet_fully_supervised.yaml"

# MINIMAL SETTINGS FOR QUICK TEST
export EPOCHS=2              # Only 3 epochs for quick test
export BATCH_PER_GPU=16       # Small batch to save memory
export WORKERS=4              # Fewer workers
export GPUS=2               # Two GPUs
export TRAIN_LIMIT=1000        # Only 100 samples for training
export VAL_LIMIT=1000           # Only 50 samples for validation
export TEST_LIMIT=1000          # Only 20 samples for testing

export DATA_SOURCE_ROOT="/net/tscratch/people/plgjmachali/surgvu_data_sampled_03"

############################
# TEMP DIRECTORIES
############################
cd "$SLURM_SUBMIT_DIR"

DATE="$(date +'%Y%m%d_%H%M')"
: "${SCRATCH:?SCRATCH must be set}"

# Test directories in temp folder
TEMP_BASE="/net/tscratch/people/plgjmachali/temp_test"
BASE="${TEMP_BASE}/pipeline_test_job_${SLURM_JOB_ID}_${DATE}"
RUN_DIR="${BASE}"
CKPT_DIR="${BASE}/checkpoints"
TB_DIR="${BASE}/tb_logs"
mkdir -p "${RUN_DIR}" "${CKPT_DIR}" "${TB_DIR}" "${TEMP_BASE}"

echo "🗂️  [PIPELINE TEST] Test directories:"
echo "   Base: ${BASE}"
echo "   Checkpoints: ${CKPT_DIR}"
echo "   TensorBoard: ${TB_DIR}"

############################
# ENVIRONMENT
############################
echo "🐍 [PIPELINE TEST] Loading conda environment..."
module load Miniconda3
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

export DATE="${DATE}"

############################
# DATA VERIFICATION
############################
echo "📂 [PIPELINE TEST] Verifying data paths..."
export DATA_ROOT="${DATA_SOURCE_ROOT}"

for split in train val test; do
  if [ ! -d "${DATA_ROOT}/${split}" ]; then
    echo "❌ [ERROR] Missing data directory: ${DATA_ROOT}/${split}"
    exit 1
  fi
done
echo "✅ [PIPELINE TEST] All data directories verified"

############################
# QUICK TRAINING TEST
############################
echo "🚀 [PIPELINE TEST] Starting minimal training (${EPOCHS} epochs, ${TRAIN_LIMIT} samples)..."

python main.py -hp "${CFG}" -m supervised \
  config.SLURM.USE_SLURM=false \
  hydra.verbose=true \
  hydra.job_logging.root.level=INFO \
  config.DISTRIBUTED.NUM_PROC_PER_NODE="${GPUS}" \
  config.DATA.TRAIN.BATCHSIZE_PER_REPLICA="${BATCH_PER_GPU}" \
  config.DATA.VAL.BATCHSIZE_PER_REPLICA="${BATCH_PER_GPU}" \
  config.DATA.TEST.BATCHSIZE_PER_REPLICA="${BATCH_PER_GPU}" \
  config.DATA.NUM_DATALOADER_WORKERS="${WORKERS}" \
  config.DATA.TRAIN.DATA_LIMIT="${TRAIN_LIMIT}" \
  config.DATA.VAL.DATA_LIMIT="${VAL_LIMIT}" \
  config.DATA.TEST.DATA_LIMIT="${TEST_LIMIT}" \
  "config.DATA.TRAIN.DATA_PATHS=[${DATA_ROOT}/train]" \
  "config.DATA.VAL.DATA_PATHS=[${DATA_ROOT}/val]" \
  "config.DATA.TEST.DATA_PATHS=[${DATA_ROOT}/test]" \
  config.OPTIMIZER.num_epochs="${EPOCHS}" \
  config.LOG_FREQUENCY=5 \
  config.TEST_EVERY_NUM_EPOCH=2 \
  "config.CHECKPOINT.DIR=${CKPT_DIR}" \
  "config.RUN_DIR=${RUN_DIR}" \
  "config.HOOKS.TENSORBOARD_SETUP.EXPERIMENT_LOG_DIR=${TB_DIR}"

echo "✅ [PIPELINE TEST] Training completed successfully!"
echo "📊 [PIPELINE TEST] Results saved to: ${BASE}"
echo "🎉 [PIPELINE TEST] Pipeline validation PASSED!"