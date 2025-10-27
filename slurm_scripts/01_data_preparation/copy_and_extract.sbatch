#!/bin/bash -l
# ===================== SLURM HEADER =====================
#SBATCH -J stage_and_extract_localfs_test
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64          # podnieś jeśli chcesz większą równoległość
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH -p plgrid-gpu-a100
#SBATCH --output=/net/tscratch/people/%u/surgvu_stage/logs/out-%j.log
#SBATCH --error=/net/tscratch/people/%u/surgvu_stage/logs/err-%j.log
#SBATCH -C localfs                   # WYMAGANE, żeby mieć $SCRATCH_LOCAL
# ========================================================

set -euo pipefail

DATE="$(date +'%Y%m%d_%H%M')"

# --- skąd bierzemy tar-y (na SCRATCH) ---
SRC_TRAIN="/net/tscratch/people/plgjmachali/surgvu_data_tars/train_tars"
SRC_VAL="/net/tscratch/people/plgjmachali/surgvu_data_tars/val_tars"
SRC_TEST="/net/tscratch/people/plgjmachali/surgvu_data_tars/test_tars"

# --- dokąd kopiujemy i rozpakowujemy na localfs ---
STAGE_ROOT="${SCRATCH_LOCAL}/surgvu_stage_test"
DST_TARS="${STAGE_ROOT}/tars"             # tu kopiujemy tar-y
DST_EXTRACT="${STAGE_ROOT}/extracted"     # tu rozpakowujemy

BASE="/net/tscratch/people/${USER}/surgvu_stage/job_${SLURM_JOB_ID}_${DATE}"
MON_DIR="${BASE}/monitoring"
REPORT="${BASE}/report.txt"

mkdir -p "${DST_TARS}" "${DST_EXTRACT}" "${MON_DIR}" "$(dirname "${REPORT}")"

echo "[INFO] DATE       = ${DATE}"
echo "[INFO] LOCALFS    = ${SCRATCH_LOCAL}"
echo "[INFO] STAGE_ROOT = ${STAGE_ROOT}"
echo "[INFO] SRC_TRAIN  = ${SRC_TRAIN}"
echo "[INFO] SRC_VAL    = ${SRC_VAL}"
echo "[INFO] SRC_TEST   = ${SRC_TEST}"
echo "[INFO] REPORT     = ${REPORT}"

# ---------- monitoring (bezpieczny) ----------
start_monitoring() {
  set +e; set +o pipefail
  if command -v dstat >/dev/null 2>&1; then
    dstat --time --cpu --mem --io --net --output "${MON_DIR}/sys.csv" 30 >/dev/null 2>> "${MON_DIR}/sys.err" &
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
# --------------------------------------------

# ===== ustawienia równoległości =====
CPUS="${SLURM_CPUS_PER_TASK:-32}"
PAR_COPY=$(( CPUS ))          # ile równoległych kopii .tar
PAR_EXTRACT=$(( CPUS / 2 ))   # ile równoległych rozpakowań
[ "${PAR_COPY}"    -lt 4 ] && PAR_COPY=4
[ "${PAR_EXTRACT}" -lt 2 ] && PAR_EXTRACT=2
echo "[PAR] copy=${PAR_COPY}  extract=${PAR_EXTRACT}"

# ===== funkcje pomocnicze =====
bytes_dir() { find "$1" -maxdepth 1 -type f -name '*.tar' -printf '%s\n' 2>/dev/null | awk '{s+=$1} END{print s+0}'; }
human() { awk -v b="${1:-0}" 'function h(x){s="BKMGTPEZY";while(x>=1024&&length(s)>1){x/=1024;s=substr(s,2)};printf "%.2f %s",x,substr(s,1,1)} BEGIN{h(b)}'; }

gather_tar_list() {
  # wypisuje listę plików .tar (pełne ścieżki) z przekazanych katalogów
  for d in "$@"; do
    [ -d "$d" ] || continue
    find "$d" -maxdepth 1 -type f -name '*.tar' -print
  done
}

# ===== przygotowanie list tarów i sprawdzenie miejsca =====
echo "[LIST] Scanning source tar files..."
mapfile -t TARS < <(gather_tar_list "${SRC_TRAIN}" "${SRC_VAL}" "${SRC_TEST}")
N_TARS="${#TARS[@]}"
if [ "${N_TARS}" -eq 0 ]; then
  echo "[ERROR] No .tar files found in sources."
  exit 1
fi
echo "[LIST] Found ${N_TARS} tar files."
TOTAL_BYTES_SRC=0
for f in "${TARS[@]}"; do
  sz=$(stat -c%s "$f" 2>/dev/null || echo 0)
  TOTAL_BYTES_SRC=$((TOTAL_BYTES_SRC + sz))
done
FREE_LOCAL=$(df -B1 "${SCRATCH_LOCAL}" | awk 'NR==2{print $4}')
echo "[SPACE] Total tar size: $(human "${TOTAL_BYTES_SRC}")"
echo "[SPACE] Free on SCRATCH_LOCAL: $(human "${FREE_LOCAL}")"

if [ "${TOTAL_BYTES_SRC}" -ge "${FREE_LOCAL}" ]; then
  echo "[WARN] Not enough space to stage ALL tars to localfs."
  echo "       Will copy sequentially per split with space checks."
fi

echo -e "Stage+Extract benchmark\nDate: ${DATE}\nJob: ${SLURM_JOB_ID}\nNode: ${SLURM_JOB_NODELIST}\n" | tee "${REPORT}"

# ===== kopiowanie równoległe =====
stage_split() {
  local src_dir="$1"   # np. SRC_TRAIN
  local dst_dir="$2"   # DST_TARS/train
  local par="$3"
  local name="$4"      # train|val|test

  [ -d "$src_dir" ] || { echo "[STAGE][$name] skip: $src_dir missing"; return 0; }
  mkdir -p "$dst_dir"

  echo "[STAGE][$name] → ${dst_dir}  (P=${par})"
  local need=$(bytes_dir "$src_dir")
  local free=$(df -B1 "${dst_dir}" | awk 'NR==2{print $4}')

  if [ "${need}" -gt "${free}" ]; then
    echo "[STAGE][$name][WARN] Not enough space for all ${name} tars (need=$(human "$need"), free=$(human "$free")). Copying as much as fits..."
  fi

  local t0=$(date +%s)
  # kopiuj równolegle, ale z guardem na wolną przestrzeń – prosto kopiujemy, a jeśli zabraknie, cp zwróci błąd i przejdziemy dalej
  find "$src_dir" -maxdepth 1 -type f -name '*.tar' -print0 \
  | xargs -0 -n1 -P "${par}" -I{} bash -lc '
      src="{}"; base=$(basename "$src"); dst="'"$dst_dir"'/$base";
      echo "[STAGE]['"$name"'] cp $base"
      cp "$src" "$dst"
    '
  local t1=$(date +%s)
  local dt=$((t1 - t0))
  printf "[STAGE][%s] DONE in %dm %ds (dst now: %s)\n" "$name" $((dt/60)) $((dt%60)) "$(du -sh "$dst_dir" 2>/dev/null | awk "{print \$1}")" | tee -a "${REPORT}"
}

# ===== rozpakowanie równoległe =====
extract_split() {
  local src_dir="$1"   # np. DST_TARS/train
  local dst_dir="$2"   # DST_EXTRACT/train
  local par="$3"
  local name="$4"

  [ -d "$src_dir" ] || { echo "[EXTRACT][$name] skip: $src_dir missing"; return 0; }
  mkdir -p "$dst_dir"

  echo "[EXTRACT][$name] → ${dst_dir}  (P=${par})"
  local t0=$(date +%s)
  # rozpakowuj równolegle: każdy tar do własnego podkatalogu (nazwa bez .tar)
  find "$src_dir" -maxdepth 1 -type f -name '*.tar' -print0 \
  | xargs -0 -n1 -P "${par}" -I{} bash -lc '
      tarfile="{}"; base=$(basename "$tarfile" .tar)
      outdir="'"$dst_dir"'/$base"
      mkdir -p "$outdir"
      echo "[EXTRACT]['"$name"'] tar -xf $base.tar"
      ( cd "$outdir" && tar -xf "$tarfile" )
    '
  local t1=$(date +%s)
  local dt=$((t1 - t0))
  printf "[EXTRACT][%s] DONE in %dm %ds (dst now: %s)\n" "$name" $((dt/60)) $((dt%60)) "$(du -sh "$dst_dir" 2>/dev/null | awk "{print \$1}")" | tee -a "${REPORT}"
}

ALL0=$(date +%s)

# 1) STAGE (SCRATCH -> SCRATCH_LOCAL)
stage_split "${SRC_TRAIN}" "${DST_TARS}/train" "${PAR_COPY}" "train"
stage_split "${SRC_VAL}"   "${DST_TARS}/val"   "${PAR_COPY}" "val"
stage_split "${SRC_TEST}"  "${DST_TARS}/test"  "${PAR_COPY}" "test"

# 2) EXTRACT (na SCRATCH_LOCAL)
extract_split "${DST_TARS}/train" "${DST_EXTRACT}/train" "${PAR_EXTRACT}" "train"
extract_split "${DST_TARS}/val"   "${DST_EXTRACT}/val"   "${PAR_EXTRACT}" "val"
extract_split "${DST_TARS}/test"  "${DST_EXTRACT}/test"  "${PAR_EXTRACT}" "test"

ALL1=$(date +%s); ALLDT=$((ALL1 - ALL0))
printf "\n[TOTAL] Stage + Extract all: %dm %ds\n" $((ALLDT/60)) $((ALLDT%60)) | tee -a "${REPORT}"

echo -e "\n[VERIFY] LocalFS layout:\n${STAGE_ROOT}\n"
tree -L 2 "${STAGE_ROOT}" 2>/dev/null || find "${STAGE_ROOT}" -maxdepth 2 -print

echo "[INFO] Report saved: ${REPORT}"
echo "[INFO] Done at $(date)"