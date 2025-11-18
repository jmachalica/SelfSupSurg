#!/bin/bash -l
# ===================== SLURM HEADER =====================
#SBATCH -J surgvu_copy_to_scratch
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH -p plgrid-gpu-a100         # if you have CPU queue, use it instead of GPU
#SBATCH --output=/net/tscratch/people/%u/surgvu_copy/logs/out-%j.log
#SBATCH --error=/net/tscratch/people/%u/surgvu_copy/logs/err-%j.log
# ========================================================

set -euo pipefail

DATE="$(date +'%Y%m%d_%H%M')"
SRC="/net/pr2/projects/plgrid/plgg_13/surgvu_data"         # source: has train/ val/ test
DST="/net/tscratch/people/plgjmachali/surgvu_data"         # destination on SCRATCH
BASE="/net/tscratch/people/${USER}/surgvu_copy/job_${SLURM_JOB_ID}_${DATE}"
MON_DIR="${BASE}/monitoring"
mkdir -p "${DST}" "${MON_DIR}" "${BASE}"

echo "[INFO] SRC=${SRC}"
echo "[INFO] DST=${DST}"
echo "[INFO] LOGS=${BASE}"

# --------- monitoring (nie przerywa joba) ----------
start_monitoring() {
  set +e
  set +o pipefail
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
# ------------------------------------------------------

# tar-stream jednego katalogu top-level (case_*)
copy_one_dir_tarpipe() {
  local src_root="$1"   # np. /.../train
  local dst_root="$2"   # np. /.../train
  local name="$3"       # np. case_000_video_part_001
  mkdir -p "${dst_root}"
  # strumień bez kompresji: szybki i oszczędza metadane
  ( cd "${src_root}" && tar -cf - "${name}" ) | ( cd "${dst_root}" && tar -xf - )
}

# kopiuj split równolegle per katalog top-level
copy_split_parallel() {
  local split="$1"          # train | val | test
  local par="$2"            # stopień równoległości (np. 16/8)
  local src_split="${SRC}/${split}"
  local dst_split="${DST}/${split}"

  echo "[COPY] === ${split} === start $(date '+%H:%M:%S') (parallel=${par})"
  local t0=$(date +%s)

  mkdir -p "${dst_split}"

  # znajdź katalogi top-level (case_*), kopiuj równolegle
  find "${src_split}" -mindepth 1 -maxdepth 1 -type d -print0 \
  | xargs -0 -n1 -P "${par}" -I{} bash -lc '
      d="{}"; base=$(basename "$d")
      echo "[COPY][part] → '"${split}"'/$base (start $(date "+%H:%M:%S"))"
      tps=$(date +%s)
      copy_one_dir_tarpipe "'"${src_split}"'" "'"${dst_split}"'" "$base"
      tpe=$(date +%s); dte=$((tpe - tps))
      printf "[COPY][part] ← %s/%s done in %dm %ds\n" "'"${split}"'" "$base" $((dte/60)) $((dte%60))
  '

  local t1=$(date +%s); local dt=$((t1 - t0))
  printf "[COPY] === %s DONE (in %dm %ds)\n" "${split}" $((dt/60)) $((dt%60))
}

# eksport funkcji do subshelli xargs
export -f copy_one_dir_tarpipe

ALL_START=$(date +%s)
# Ustal równoległość: train zwykle największy
copy_split_parallel train 16
copy_split_parallel val   8
copy_split_parallel test  8
ALL_END=$(date +%s); ALL_DT=$((ALL_END - ALL_START))
printf "[COPY] ✅ TOTAL: %dm %ds\n" $((ALL_DT/60)) $((ALL_DT%60))

echo "[VERIFY] Listing ${DST}:"
ls -lh "${DST}" || echo "[WARN] cannot list ${DST}"

echo "[INFO] Done at $(date)"