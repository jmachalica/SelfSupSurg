#!/bin/bash -l
# ===================== SLURM HEADER =====================
#SBATCH -J surgvu_pack_to_scratch
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH -p plgrid-gpu-a100
#SBATCH --output=/net/tscratch/people/%u/surgvu_pack/logs/out-%j.log
#SBATCH --error=/net/tscratch/people/%u/surgvu_pack/logs/err-%j.log
#SBATCH -C localfs
# ========================================================

set -euo pipefail

DATE="$(date +'%Y%m%d_%H%M')"
SRC="/net/pr2/projects/plgrid/plgg_13/surgvu_data"
DST_TARS="/net/tscratch/people/plgjmachali/surgvu_data_tars"
TMP_TARS="${SCRATCH_LOCAL}/surgvu_tars"
BASE="/net/tscratch/people/${USER}/surgvu_pack/job_${SLURM_JOB_ID}_${DATE}"
MON_DIR="${BASE}/monitoring"
REPORT="${BASE}/report.txt"

mkdir -p "${TMP_TARS}" "${DST_TARS}" "${MON_DIR}" "${BASE}"

echo "[INFO] SRC      = ${SRC}"
echo "[INFO] TMP_TARS = ${TMP_TARS}"
echo "[INFO] DST_TARS = ${DST_TARS}"
echo "[INFO] LOGS     = ${BASE}"

# -------- monitoring (non-fatal) --------
start_monitoring() {
  set +e; set +o pipefail
  if command -v dstat >/dev/null 2>&1; then
    dstat --cpu --mem --io --net --output "${MON_DIR}/sys.csv" 30 >/dev/null 2>> "${MON_DIR}/sys.err" &
    SYS_PID=$!
  else
    top -b -d 60 -n 2000 > "${MON_DIR}/top.log" 2>> "${MON_DIR}/top.err" &
    SYS_PID=$!
  fi
  set -e; set -o pipefail
}
stop_monitoring(){ set +e; [ -n "${SYS_PID:-}" ] && kill "${SYS_PID}" >/dev/null 2>&1 || true; set -e; }
trap stop_monitoring EXIT
start_monitoring
# ----------------------------------------

# funkcja pakująca jeden katalog w tar i kopiująca na SCRATCH
pack_one_dir_then_copy() {
  local split="$1"   # train | val | test
  local dname="$2"   # np. case_000_video_part_001
  local src_split="${SRC}/${split}"
  local dst_split="${DST_TARS}/${split}_tars"
  local tar_tmp="${TMP_TARS}/${split}_${dname}.tar"
  local tar_dst="${dst_split}/${dname}.tar"

  mkdir -p "${dst_split}"
  echo "[PACK][${split}] → ${dname} (start: $(date '+%H:%M:%S'))"

  t0=$(date +%s)
  ( cd "${src_split}" && tar --exclude='._*' --exclude='.DS_Store' -cf "${tar_tmp}" "${dname}" )
  cp "${tar_tmp}" "${tar_dst}"
  rm -f "${tar_tmp}" || true
  t1=$(date +%s)
  dt=$((t1 - t0))
  printf "[PACK][%s] ← %s.tar done in %dm %ds\n" "${split}" "${dname}" $((dt/60)) $((dt%60))
}

export SRC DST_TARS TMP_TARS
export -f pack_one_dir_then_copy

# kopiowanie równoległe
pack_split_parallel() {
  local split="$1"
  local par="$2"
  local src_split="${SRC}/${split}"

  echo "[PACK] === ${split} === start $(date '+%H:%M:%S') (parallel=${par})"
  t0=$(date +%s)
  find "${src_split}" -mindepth 1 -maxdepth 1 -type d -printf '%f\0' \
  | xargs -0 -n1 -P "${par}" -I{} bash -lc 'pack_one_dir_then_copy "'"${split}"'" "$@"' _ {}
  t1=$(date +%s); dt=$((t1 - t0))
  printf "[PACK] === %s DONE (in %dm %ds)\n" "${split}" $((dt/60)) $((dt%60))
}

# ustal równoległość
CPUS="${SLURM_CPUS_PER_TASK:-32}"
PAR_TRAIN=$((CPUS*3/4))
PAR_VAL=$((PAR_TRAIN/3))
PAR_TEST=$((PAR_TRAIN/3))
echo "[PARALLELISM] train=${PAR_TRAIN}, val=${PAR_VAL}, test=${PAR_TEST}"

ALL0=$(date +%s)
pack_split_parallel train "${PAR_TRAIN}"
pack_split_parallel val   "${PAR_VAL}"
pack_split_parallel test  "${PAR_TEST}"
ALL1=$(date +%s)
ALLDT=$((ALL1 - ALL0))
printf "[TOTAL] All splits done in %dm %ds\n" $((ALLDT/60)) $((ALLDT%60))

echo "[VERIFY] Listing ${DST_TARS}:"
ls -lh "${DST_TARS}" || true
echo "[INFO] Done at $(date)"