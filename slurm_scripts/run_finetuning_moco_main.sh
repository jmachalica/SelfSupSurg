#!/bin/bash
# ========================================================
# Shared MoCo fine-tuning script logic
# This file is sourced by specific MoCo fine-tuning sbatch scripts
# ========================================================

############################
# PATHS & PREP
############################
cd "$SLURM_SUBMIT_DIR"

DATE="$(date +'%Y%m%d_%H%M')"
: "${SCRATCH:?SCRATCH must be set (e.g., /net/tscratch/people/$USER)}"

BASE="${SCRATCH}/surgvu_results/finetuning/moco_to_surgvu/${DATA_PERCENTAGE}/job_${SLURM_JOB_ID}_${DATE}"
RUN_DIR="${BASE}"
CKPT_DIR="${BASE}"
TB_DIR="${BASE}/logs"
MON_DIR="${BASE}/monitoring"
mkdir -p "${RUN_DIR}" "${CKPT_DIR}" "${TB_DIR}" "${MON_DIR}"

echo "[INFO] Job: $SLURM_JOB_NAME ($SLURM_JOB_ID)"
echo "[INFO] Node(s): $SLURM_JOB_NODELIST"
echo "[INFO] Config: ${CFG}"
echo "[INFO] Data Percentage: ${DATA_PERCENTAGE}%"
echo "[INFO] MoCo Fine-tuning from checkpoint: ${MOCO_CHECKPOINT_PATH}"

############################
# ENVIRONMENT
############################
module load Miniconda3
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

export DATE="${DATE}"

############################
# DATA SETUP
############################
# Data already available at DATA_SOURCE_ROOT, no staging needed
export DATA_ROOT="${DATA_SOURCE_ROOT}"
echo "[INFO] Using data from: ${DATA_ROOT}"

# Verify data directories exist
for split in train val test; do
  if [ ! -d "${DATA_ROOT}/${split}" ]; then
    echo "[ERROR] Missing data directory: ${DATA_ROOT}/${split}"
    exit 1
  fi
done
echo "[INFO] ✅ All data directories verified"

############################
# MONITORING (enhanced from testing script)
############################
start_monitoring() {
  set +e
  set +o pipefail
  
  echo "[MONITOR] Starting monitoring processes..."
  
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total \
               --format=csv -l 30 > "${MON_DIR}/gpu.csv" 2>> "${MON_DIR}/gpu.err" &
    GPU_PID=$!
    echo "[MONITOR] GPU monitoring started (PID: ${GPU_PID})"
  fi

  if command -v dstat >/dev/null 2>&1; then
    dstat --cpu --mem --io --net --output "${MON_DIR}/sys.csv" 30 >/dev/null 2>> "${MON_DIR}/sys.err" &
    SYS_PID=$!
    echo "[MONITOR] System monitoring (dstat) started (PID: ${SYS_PID})"
  elif command -v pidstat >/dev/null 2>&1; then
    pidstat -r -u -d -h -p ALL 30 > "${MON_DIR}/pidstat.log" 2>> "${MON_DIR}/pidstat.err" &
    SYS_PID=$!
    echo "[MONITOR] System monitoring (pidstat) started (PID: ${SYS_PID})"
  else
    top -b -d 60 -n 2000 > "${MON_DIR}/top.log" 2>> "${MON_DIR}/top.err" &
    SYS_PID=$!
    echo "[MONITOR] System monitoring (top) started (PID: ${SYS_PID})"
  fi
  
  # Log job info
  cat > "${MON_DIR}/job_info.txt" << EOF
Job ID: ${SLURM_JOB_ID}
Job Name: ${SLURM_JOB_NAME}
Node(s): ${SLURM_JOB_NODELIST}
Config: ${CFG}
Data Percentage: ${DATA_PERCENTAGE}%
Start Time: $(date)
Data Source: ${DATA_SOURCE_ROOT}
Staged Data: ${DATA_ROOT}
MoCo Checkpoint: ${MOCO_CHECKPOINT_PATH}
Experiment Type: MoCo Fine-tuning
Epochs: ${EPOCHS}
GPUs: ${GPUS}
Batch per GPU: ${BATCH_PER_GPU}
EOF
  
  set -e
  set -o pipefail
}

stop_monitoring() {
  set +e
  echo "[MONITOR] Stopping monitoring processes..."
  [ -n "${GPU_PID:-}" ] && kill "${GPU_PID}" >/dev/null 2>&1 && echo "[MONITOR] Stopped GPU monitoring" || true
  [ -n "${SYS_PID:-}" ] && kill "${SYS_PID}" >/dev/null 2>&1 && echo "[MONITOR] Stopped system monitoring" || true
  
  # Log completion
  echo "End Time: $(date)" >> "${MON_DIR}/job_info.txt"
  echo "Final monitoring directory: ${MON_DIR}" >> "${MON_DIR}/job_info.txt"
  set -e
}

trap stop_monitoring EXIT
trap stop_monitoring INT TERM
start_monitoring

############################
# TRAIN
############################
echo "[INFO] Launching MoCo fine-tuning ..."

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
  "config.DATA.TRAIN.DATA_PATHS=[${DATA_ROOT}/train]" \
  "config.DATA.VAL.DATA_PATHS=[${DATA_ROOT}/val]" \
  "config.DATA.TEST.DATA_PATHS=[${DATA_ROOT}/test]" \
  config.OPTIMIZER.num_epochs="${EPOCHS}" \
  config.LOG_FREQUENCY=10 \
  config.TEST_EVERY_NUM_EPOCH=1 \
  "config.CHECKPOINT.DIR=${CKPT_DIR}" \
  "config.RUN_DIR=${RUN_DIR}" \
  "config.HOOKS.TENSORBOARD_SETUP.EXPERIMENT_LOG_DIR=${TB_DIR}" \
  "config.MODEL.WEIGHTS_INIT.PARAMS_FILE=${MOCO_CHECKPOINT_PATH}"

echo "[INFO] MoCo fine-tuning done."