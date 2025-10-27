#!/bin/bash -l
# ===================== SLURM HEADER =====================
#SBATCH -J surgvu_finetune
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
#SBATCH -C localfs
# ========================================================

set -euo pipefail

############################
# USER CONFIG
############################
CONDA_ENV="/net/pr2/projects/plgrid/plgg_13/conda/envs/vissl_3_8"
CFG="hparams/surgvu/finetuning/imagenet_to_surgvu/100/imagenet_fully_supervised.yaml"

EPOCHS=100
BATCH_PER_GPU=256
WORKERS=8
GPUS=4
TRAIN_LIMIT=-1
VAL_LIMIT=-1
TEST_LIMIT=-1
############################
# PATHS & PREP
############################
cd "$SLURM_SUBMIT_DIR"

DATE="$(date +'%Y%m%d_%H%M')"
: "${SCRATCH:?SCRATCH must be set (e.g., /net/tscratch/people/$USER)}"

BASE="${SCRATCH}/surgvu_results/finetuning/imagenet_to_surgvu/100/job_${SLURM_JOB_ID}_${DATE}"
RUN_DIR="${BASE}"
CKPT_DIR="${BASE}"
TB_DIR="${BASE}/logs"
mkdir -p "${RUN_DIR}" "${CKPT_DIR}" "${TB_DIR}"

echo "[INFO] Job: $SLURM_JOB_NAME ($SLURM_JOB_ID)"
echo "[INFO] Node(s): $SLURM_JOB_NODELIST"

############################
# ENVIRONMENT
############################
module load Miniconda3
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

export DATE="${DATE}"

############################
# FAST STAGING (tar-stream)
############################
start_time=$(date +%s)
set +e
set +o pipefail

SRC_ROOT="/net/pr2/projects/plgrid/plgg_13/surgvu_data"
DST_LOCAL="${SCRATCH_LOCAL:-}"
DST_GLOBAL="${SCRATCH}/surgvu_staged"

need_dir() { mkdir -p "$1"; }

copy_tarstream_one() {
  local src_root="$1"; local dst_root="$2"; local name="$3"
  echo "[STAGE] ${name}: ${src_root}/${name} -> ${dst_root}/${name}"
  need_dir "${dst_root}"
  # kopiowanie strumieniowe tar → tar (bez kompresji)
  ( cd "${src_root}" && tar -cf - "${name}" ) | ( cd "${dst_root}" && tar -xf - )
  return $?
}

copy_split_to_target() {
  local src_root="$1"; local dst_root="$2"
  for split in train val test; do
    copy_tarstream_one "${src_root}" "${dst_root}" "${split}" || return 1
  done
  return 0
}

DST="${DST_GLOBAL}"
if [ -n "${DST_LOCAL}" ] && [ -d "${DST_LOCAL}" ]; then
  CAND="${DST_LOCAL}/surgvu_split"
  echo "[STAGE] Trying SCRATCH_LOCAL: ${CAND}"
  need_dir "${CAND}"
  copy_split_to_target "${SRC_ROOT}" "${CAND}"
  STATUS=$?
  if [ ${STATUS} -eq 0 ]; then
    DST="${CAND}"
    echo "[STAGE] OK on SCRATCH_LOCAL: ${DST}"
  else
    echo "[STAGE][WARN] localfs copy failed, falling back to SCRATCH"
    DST="${DST_GLOBAL}"
  fi
else
  echo "[STAGE] SCRATCH_LOCAL not available, using SCRATCH"
fi

if [ "${DST}" = "${DST_GLOBAL}" ]; then
  need_dir "${DST_GLOBAL}"
  echo "[STAGE] Copying to SCRATCH: ${DST_GLOBAL}"
  copy_split_to_target "${SRC_ROOT}" "${DST_GLOBAL}"
fi

export DATA_ROOT="${DST}"
echo "[STAGE] DATA_ROOT=${DATA_ROOT}"

end_time=$(date +%s)
total=$((end_time - start_time))
printf "[STAGE] ✅ Total staging time: %dm %ds\n" $((total / 60)) $((total % 60))

set -e
set -o pipefail

############################
# MONITORING (safe)
############################
MON_DIR="${BASE}/monitoring"
mkdir -p "${MON_DIR}"

start_monitoring() {
  set +e
  set +o pipefail
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total \
               --format=csv -l 30 > "${MON_DIR}/gpu.csv" 2>> "${MON_DIR}/gpu.err" &
    GPU_PID=$!
  fi
  if command -v dstat >/dev/null 2>&1; then
    dstat --cpu --mem --io --net --output "${MON_DIR}/sys.csv" 30 >/dev/null 2>> "${MON_DIR}/sys.err" &
    SYS_PID=$!
  elif command -v pidstat >/dev/null 2>&1; then
    pidstat -r -u -d -h -p ALL 30 > "${MON_DIR}/pidstat.log" 2>> "${MON_DIR}/pidstat.err" &
    SYS_PID=$!
  else
    top -b -d 60 -n 2000 > "${MON_DIR}/top.log" 2>> "${MON_DIR}/top.err" &
    SYS_PID=$!
  fi
  set -e
  set -o pipefail
}

stop_monitoring() {
  set +e
  [ -n "${GPU_PID:-}" ] && kill "${GPU_PID}" >/dev/null 2>&1 || true
  [ -n "${SYS_PID:-}" ] && kill "${SYS_PID}" >/dev/null 2>&1 || true
  set -e
}

trap stop_monitoring EXIT
trap stop_monitoring INT TERM
start_monitoring

############################
# TRAIN
############################
echo "[INFO] Launching training ..."

srun python main.py -hp "${CFG}" -m supervised \
  config.SLURM.USE_SLURM=false \
  hydra.verbose=true \
  hydra.job_logging.root.level=DEBUG \
  config.DISTRIBUTED.NUM_PROC_PER_NODE="${GPUS}" \
  config.DATA.TRAIN.BATCHSIZE_PER_REPLICA="${BATCH_PER_GPU}" \
  config.DATA.VAL.BATCHSIZE_PER_REPLICA="${BATCH_PER_GPU}" \
  config.DATA.TEST.BATCHSIZE_PER_REPLICA="${BATCH_PER_GPU}" \
  config.DATA.NUM_DATALOADER_WORKERS="${WORKERS}" \
  config.DATA.TRAIN.DATA_LIMIT="${TRAIN_LIMIT}" \
  config.DATA.VAL.DATA_LIMIT="${VAL_LIMIT}" \
  config.DATA.TEST.DATA_LIMIT="${TEST_LIMIT}" \
  "config.DATA.TRAIN.DATA_PATH=${DATA_ROOT}/train" \
  "config.DATA.VAL.DATA_PATH=${DATA_ROOT}/val" \
  "config.DATA.TEST.DATA_PATH=${DATA_ROOT}/test" \
  config.OPTIMIZER.num_epochs="${EPOCHS}" \
  config.LOG_FREQUENCY=10 \
  config.TEST_EVERY_NUM_EPOCH=10 \
  "config.CHECKPOINT.DIR=${CKPT_DIR}" \
  "config.RUN_DIR=${RUN_DIR}" \
  "config.HOOKS.TENSORBOARD_SETUP.EXPERIMENT_LOG_DIR=${TB_DIR}"

echo "[INFO] Done."