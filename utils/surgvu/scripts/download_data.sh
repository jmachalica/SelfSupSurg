#!/bin/bash

set -e

DATA_URL="https://storage.googleapis.com/isi-surgvu/surgvu24_videos_only.zip"
OUTPUT_DIR="${1:-../data}"
FILENAME="${2:-surgvu24_videos_only.zip}"
LOG_FILE="$HOME/download_data.log"

if [ -f "$OUTPUT_DIR/$FILENAME" ]; then
  echo "✅ File already exists at $OUTPUT_DIR/$FILENAME. Skipping download."
  exit 0
fi

mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

exec > >(tee -i "$LOG_FILE")
exec 2>&1

echo "📦 Starting download: $DATA_URL"
echo "📁 Output directory: $OUTPUT_DIR"
echo "📅 Start time: $(date)"

aria2c --file-allocation=none -x 16 -s 16 \
  --allow-overwrite=false \
  --summary-interval=0 --console-log-level=notice \
  -o "$FILENAME" "$DATA_URL"

if [ $? -ne 0 ]; then
  echo "❌ aria2c failed to download the file properly. Aborting."
  exit 1
fi


echo "✅ Download complete. File saved as $OUTPUT_DIR/$FILENAME"
echo "📅 End time: $(date)"
