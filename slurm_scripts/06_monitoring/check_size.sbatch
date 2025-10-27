#!/bin/bash -l
#SBATCH -J check_free_localfs
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH -p plgrid-gpu-a100
#SBATCH --output=/net/tscratch/people/%u/localfs_check/out-%j.log
#SBATCH --error=/net/tscratch/people/%u/localfs_check/err-%j.log
#SBATCH -C localfs            # aby mieć dostęp do $SCRATCH_LOCAL

set -euo pipefail

echo "[INFO] Job: $SLURM_JOB_ID on node(s): $SLURM_JOB_NODELIST"

if [ -z "${SCRATCH_LOCAL:-}" ]; then
  echo "ERROR: \$SCRATCH_LOCAL not defined"
  exit 1
fi

echo "[CHECK] SCRATCH_LOCAL = $SCRATCH_LOCAL"
echo "[CHECK] Filesystem usage (human):"
df -h "$SCRATCH_LOCAL"

echo "[CHECK] Free bytes exactly:"
df -B1 "$SCRATCH_LOCAL" | awk 'NR==2 { print "Free bytes:", $4 }'

echo "[CHECK] Size of existing data in scratch_local (if any):"
du -sh "$SCRATCH_LOCAL" || echo "[WARN] du failed"

echo "[DONE] check complete."