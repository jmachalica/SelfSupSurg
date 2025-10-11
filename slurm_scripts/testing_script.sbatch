#!/bin/bash -l
# ===================== SLURM HEADER =====================
#SBATCH -J pack_surgvu_to_scratch
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH -p plgrid-gpu-a100
#SBATCH --output=/net/tscratch/people/%u/surgvu_pack/logs/out-%j.log
#SBATCH --error=/net/tscratch/people/%u/surgvu_pack/logs/err-%j.log
#SBATCH -C localfs
# ========================================================

set -euo pipefail

DATE="$(date +'%Y%m%d_%H%M')"
SRC="/net/pr2/projects/plgrid/plgg_13/surgvu_data"
TMP="/localfs/${SLURM_JOB_ID}/surgvu_pack_tmp"
DST="/net/tscratch/people/plgjmachali/surgvu_data_archives"
MON_DIR="/net/tscratch/people/${USER}/surgvu_pack/job_${SLURM_JOB_ID}_${DATE}/monitoring"

mkdir -p "${TMP}" "${DST}" "${MON_DIR}"
echo "[INFO] SRC: ${SRC}"
echo "[INFO] DST: ${DST}"
echo "[INFO] TMP: ${TMP}"
echo "[INFO] MONITORING: ${MON_DIR}"

############################
# MONITORING (non-fatal)
############################
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
  [ -n "${SYS_PID:-}" ] && kill "${SYS_PID}" >/dev/null 2>&1 || true
  set -e
}
trap stop_monitoring EXIT
start_monitoring

############################
# PACKING (per split)
############################
pack_split() {
  local split="$1"
  local src_dir="${SRC}/${split}"
  local tar_tmp="${TMP}/${split}.tar"
  local tar_dst="${DST}/${split}.tar"

  echo "[PACK] → ${split} (start: $(date '+%H:%M:%S'))"
  t0=$(date +%s)

  ( cd "${SRC}" && tar -cf "${tar_tmp}" "${split}" )
  echo "[PACK]   Created tar: ${tar_tmp}"
  echo "[PACK]   Copying to DST: ${tar_dst}"
  cp "${tar_tmp}" "${tar_dst}"

  t1=$(date +%s)
  dt=$((t1 - t0))
  size=$(du -h "${tar_dst}" | awk '{print $1}')
  printf "[PACK] ← %s done: %s (%dm %ds)\n" "${split}" "${size}" $((dt/60)) $((dt%60))
}

start_all=$(date +%s)
for split in train val test; do
  pack_split "${split}"
done
end_all=$(date +%s)
dt_all=$((end_all - start_all))
printf "[PACK] ✅ TOTAL: %dm %ds\n" $((dt_all/60)) $((dt_all%60))

############################
# VERIFY
############################
echo "[VERIFY] Final files in ${DST}:"
ls -lh "${DST}"

echo "[INFO] Done at $(date)"