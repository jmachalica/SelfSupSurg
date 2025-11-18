#!/usr/bin/env bash
# Works folder-by-folder, no parallelism.

set -euo pipefail

## ======= SET THESE =======
SRC=""                 # source root folder
DST_USER=""                           # SSH username
DST_HOST="athena.cyfronet.pl"             # SSH host (or IP)
DST_BASE=""     # destination root folder on the remote machine
TAR_DIR=""                      # temporary tar directory
LOGFILE="./rsync_sync.log"                # where to store the run log
## ==========================

DIRS=(
  case_000_video_part_001
  case_001_video_part_001
  case_002_video_part_001
  case_002_video_part_002
  case_003_video_part_001
  case_003_video_part_002
  case_004_video_part_001
  case_004_video_part_002
  case_005_video_part_001
  case_006_video_part_001
  case_006_video_part_002
  case_007_video_part_001
  case_007_video_part_002
  case_008_video_part_001
  case_009_video_part_001
  case_009_video_part_002
  case_010_video_part_001
  case_010_video_part_002
  case_011_video_part_001
  case_011_video_part_002
  case_012_video_part_001
  case_012_video_part_002
  case_013_video_part_001
  case_013_video_part_002
  case_014_video_part_001
  case_014_video_part_002
  case_015_video_part_001
  case_016_video_part_001
  case_016_video_part_002
  case_017_video_part_001
  case_017_video_part_002
  case_018_video_part_001
  case_018_video_part_002
  case_019_video_part_001
  case_020_video_part_001
  case_020_video_part_002
  case_021_video_part_001
  case_021_video_part_002
  case_022_video_part_001
  case_022_video_part_002
  case_023_video_part_001
  case_024_video_part_001
  case_024_video_part_002
  case_025_video_part_001
  case_025_video_part_002
  case_026_video_part_001
  case_026_video_part_002
  case_027_video_part_001
  case_027_video_part_002
  case_028_video_part_001
  case_028_video_part_002
  case_029_video_part_001
  case_030_video_part_001
  case_030_video_part_002
  case_031_video_part_001
  case_031_video_part_002
  case_032_video_part_001
  case_032_video_part_002
  case_033_video_part_001
  case_033_video_part_002
  case_034_video_part_001
  case_035_video_part_001
  case_036_video_part_001
  case_036_video_part_002
  case_037_video_part_001
  case_037_video_part_002
  case_038_video_part_001
  case_038_video_part_002
  case_039_video_part_001
  case_039_video_part_002
  case_040_video_part_001
  case_040_video_part_002
  case_041_video_part_001
  case_041_video_part_002
  case_042_video_part_001
  case_042_video_part_002
  case_043_video_part_001
  case_043_video_part_002
  case_044_video_part_001
  case_045_video_part_001
  case_046_video_part_001
  case_047_video_part_001
  case_047_video_part_002
  case_048_video_part_001
  case_048_video_part_002
  case_049_video_part_001
  case_049_video_part_002
  case_050_video_part_001
  case_050_video_part_002
  case_051_video_part_001
  case_051_video_part_002
  case_052_video_part_001
  case_053_video_part_001
  case_053_video_part_002
  case_054_video_part_001
  case_054_video_part_002
  case_055_video_part_001
  case_055_video_part_002
  case_056_video_part_001
  case_056_video_part_002
  case_057_video_part_001
  case_057_video_part_002
  case_058_video_part_001
  case_058_video_part_002
  case_059_video_part_001
  case_060_video_part_001
  case_060_video_part_002
  case_061_video_part_001
  case_061_video_part_002
  case_062_video_part_001
  case_063_video_part_001
  case_063_video_part_002
  case_064_video_part_001
  case_064_video_part_002
  case_065_video_part_001
  case_065_video_part_002
  case_066_video_part_001
  case_066_video_part_002
  case_067_video_part_001
  case_067_video_part_002
  case_068_video_part_001
  case_068_video_part_002
  case_069_video_part_001
  case_069_video_part_002
  case_070_video_part_001
  case_070_video_part_002
  case_071_video_part_002
  case_072_video_part_001
  case_072_video_part_002
  case_073_video_part_001
  case_073_video_part_002
  case_074_video_part_001
  case_074_video_part_002
  case_075_video_part_001
  case_075_video_part_002
  case_076_video_part_001
  case_076_video_part_002
  case_077_video_part_001
  case_077_video_part_002
  case_078_video_part_001
  case_079_video_part_001
  case_079_video_part_002
  case_080_video_part_001
  case_080_video_part_002
  case_081_video_part_001
  case_082_video_part_001
  case_082_video_part_002
  case_083_video_part_001
  case_083_video_part_002
  case_084_video_part_001
  case_084_video_part_002
  case_085_video_part_001
  case_085_video_part_002
  case_086_video_part_001
  case_086_video_part_002
  case_087_video_part_001
  case_087_video_part_002
  case_088_video_part_001
  case_088_video_part_002
  case_089_video_part_001
  case_089_video_part_002
  case_090_video_part_001
  case_090_video_part_002
  case_091_video_part_001
  case_091_video_part_002
  case_092_video_part_001
  case_092_video_part_002
  case_093_video_part_001
  case_093_video_part_002
  case_094_video_part_001
  case_094_video_part_002
  case_095_video_part_001
  case_095_video_part_002
  case_096_video_part_001
  case_096_video_part_002
  case_097_video_part_001
  case_097_video_part_002
  case_098_video_part_001
  case_098_video_part_002
  case_099_video_part_001
  case_100_video_part_001
  case_100_video_part_002
  case_101_video_part_001
  case_101_video_part_002
  case_102_video_part_001
  case_102_video_part_002
  case_103_video_part_001
  case_103_video_part_002
  case_104_video_part_001
  case_104_video_part_002
  case_105_video_part_001
  case_106_video_part_001
  case_106_video_part_002
  case_107_video_part_001
  case_108_video_part_001
  case_108_video_part_002
  case_109_video_part_001
  case_109_video_part_002
  case_110_video_part_001
  case_110_video_part_002
  case_111_video_part_001
  case_111_video_part_002
  case_112_video_part_001
  case_112_video_part_002
  case_113_video_part_001
  case_113_video_part_002
  case_114_video_part_001
  case_114_video_part_002
  case_115_video_part_001
  case_115_video_part_002
  case_116_video_part_001
  case_116_video_part_002
  case_117_video_part_001
  case_117_video_part_002
  case_118_video_part_001
  case_118_video_part_002
  case_119_video_part_001
  case_119_video_part_002
  case_120_video_part_001
  case_120_video_part_002
  case_121_video_part_001
  case_121_video_part_002
  case_122_video_part_001
  case_123_video_part_001
  case_124_video_part_001
  case_124_video_part_002
  case_125_video_part_001
  case_126_video_part_001
  case_127_video_part_001
  case_128_video_part_001
  case_128_video_part_002
  case_129_video_part_001
  case_129_video_part_002
  case_130_video_part_001
  case_130_video_part_002
  case_131_video_part_001
  case_131_video_part_002
  case_132_video_part_001
  case_132_video_part_002
  case_133_video_part_001
  case_133_video_part_002
  case_134_video_part_001
  case_134_video_part_002
  case_135_video_part_001
  case_135_video_part_002
  case_136_video_part_001
  case_136_video_part_002
  case_137_video_part_001
  case_137_video_part_002
  case_138_video_part_001
  case_138_video_part_002
  case_139_video_part_001
  case_139_video_part_002
  case_140_video_part_001
  case_140_video_part_002
  case_141_video_part_001
  case_141_video_part_002
  case_142_video_part_001
  case_142_video_part_002
  case_143_video_part_001
  case_144_video_part_001
  case_145_video_part_001
  case_145_video_part_002
  case_146_video_part_001
  case_146_video_part_002
  case_147_video_part_001
  case_148_video_part_001
  case_148_video_part_002
  case_149_video_part_001
  case_149_video_part_002
  case_150_video_part_001
  case_150_video_part_002
  case_151_video_part_001
  case_151_video_part_002
  case_152_video_part_001
  case_152_video_part_002
  case_153_video_part_001
  case_153_video_part_002
  case_154_video_part_001
  case_154_video_part_002
)

mkdir -p "$TAR_DIR"

# SSH without compression (JPGs are already compressed) + keepalive
export RSYNC_RSH="ssh -T -o Compression=no -o ServerAliveInterval=30 -o IPQoS=throughput"

# Common rsync options
RSYNC_OPTS=(-aH --info=progress2 --human-readable \
            --partial --inplace \
            --mkpath)

# Nice logging: write to both screen and file
exec >> >(tee -a "$LOGFILE") 2>&1

echo ">>> Start: $(date)"
echo "SRC=$SRC"
echo "DST=${DST_USER}@${DST_HOST}:${DST_BASE}"
echo

# check if each entry exists under $SRC
for d in "${DIRS[@]}"; do test -d "$SRC/$d" || echo "Missing: $SRC/$d"; done

# Iterate over user-provided subfolder names (relative to $SRC)
for NAME in "${DIRS[@]}"; do
  SUBDIR="$SRC/$NAME"
  if [[ ! -d "$SUBDIR" ]]; then
    echo "Skipping (not found under $SRC): $NAME" >&2
    continue
  fi

  TARFILE="$TAR_DIR/${NAME}.tar"
  echo "===> Archiving: $NAME"
  tar -cf "$TARFILE" -C "$SRC" "$NAME"

  echo "===> Sending tar: $TARFILE"
  ATTEMPTS=0
  while ! rsync "${RSYNC_OPTS[@]}" "$TARFILE" "${DST_USER}@${DST_HOST}:${DST_BASE}/"; do
    ATTEMPTS=$((ATTEMPTS+1))
    SLEEP_TIME=$((20 * 2**(ATTEMPTS-1)))
    echo "Rsync failed for $NAME (attempt $ATTEMPTS). Retrying in ${SLEEP_TIME}s..."
    sleep "$SLEEP_TIME"
  done

  echo "===> Removing tar: $TARFILE"
  rm -f "$TARFILE"

  echo "<=== Done: $NAME"
  echo
done

echo ">>> End: $(date)"
echo "Log saved to: $LOGFILE"