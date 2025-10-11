#!/bin/bash -l
#SBATCH -J copy_subsample_surgvu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH -p plgrid-gpu-a100
#SBATCH --output=/net/tscratch/people/%u/copy_subsample/out-%j.log
#SBATCH --error=/net/tscratch/people/%u/copy_subsample/err-%j.log

set -euo pipefail

########################
# KONFIGURACJA (zmień)
########################
SRC="/net/pr2/projects/plgrid/plgg_13/surgvu_data"         # ma: train/ val/ test/
DST="/net/tscratch/people/$USER/surgvu_data_sampled_03"    # wyjście z subsamplingiem
N=3                                   # co N-ta klatka (3 => ~33%)
FRAME_GLOB="frame_*.jpg"              # wzorzec nazw klatek
PARALLEL="${SLURM_CPUS_PER_TASK:-16}" # równoległość per case

echo "[INFO] SRC=$SRC"
echo "[INFO] DST=$DST  (stride N=$N)"
mkdir -p "$DST"

# --- funkcja: skopiuj co N-tą klatkę z jednego case'u ---
copy_case_stride() {
  local src_case="$1" dst_case="$2" n="$3" glob="$4"
  mkdir -p "$dst_case"

  # sort alfabetyczny (spójny z Pythonem, który sortuje po stringu file_name)
  mapfile -t frames < <(ls "$src_case"/$glob 2>/dev/null | sort)
  local total="${#frames[@]}"
  if [ "$total" -eq 0 ]; then
    echo "[WARN] empty: $src_case"; return 0
  fi

  local kept=0
  for ((i=0; i<total; i+=n)); do
    f="${frames[$i]}"; base="$(basename "$f")"
    cp "$f" "$dst_case/$base"
    ((kept++))
  done
  echo "[COPY] $(basename "$src_case"): kept=$kept / total=$total"
}

export -f copy_case_stride

process_split() {
  local split="$1"
  local src_split="$SRC/$split"
  local dst_split="$DST/$split"
  echo "[SPLIT] $split → $dst_split"
  mkdir -p "$dst_split"

  find "$src_split" -mindepth 1 -maxdepth 1 -type d -print0 \
  | xargs -0 -I{} -P "$PARALLEL" bash -lc "
      src_case=\"{}\"
      case_name=\"\$(basename \"\$src_case\")\"
      dst_case=\"$dst_split/\$case_name\"
      copy_case_stride \"\$src_case\" \"\$dst_case\" \"$N\" \"$FRAME_GLOB\"
    "
}

t0=$(date +%s)
for split in train val test; do
  if [ -d "$SRC/$split" ]; then
    process_split "$split"
  else
    echo "[SPLIT] $split: brak w SRC, pomijam."
  fi
done
t1=$(date +%s)
printf "[TIME] total: %dm %ds\n" $(((t1-t0)/60)) $(((t1-t0)%60))
echo "[DONE] $(date)"