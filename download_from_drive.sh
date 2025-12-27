
#!/bin/bash
set -e

# -------- CONFIG --------
DRIVE_FILE_ID="$1"
ARCHIVE_NAME="current_chunk.tar.gz"

SCRATCH_BASE="/scratch/chuk303/nih_chestxray"
ARCHIVES_DIR="$SCRATCH_BASE/data/archives"
TEMP_IMAGES_DIR="$SCRATCH_BASE/data/temp/images"
# ------------------------

if [ -z "$DRIVE_FILE_ID" ]; then
    echo "[ERROR] No Drive file ID provided"
    exit 1
fi

mkdir -p "$ARCHIVES_DIR" "$TEMP_IMAGES_DIR"

echo "[CLEANUP] Removing old temp files..."
rm -f "$ARCHIVES_DIR"/*.tar.gz*
rm -rf "$TEMP_IMAGES_DIR"/*

echo "[INFO] Downloading dataset chunk from Google Drive..."
gdown "https://drive.google.com/uc?id=$DRIVE_FILE_ID" \
      -O "$ARCHIVES_DIR/$ARCHIVE_NAME"

echo "[INFO] Extracting chunk..."
tar -xzf "$ARCHIVES_DIR/$ARCHIVE_NAME" \
    -C "$TEMP_IMAGES_DIR" \
    --strip-components=1

rm -f "$ARCHIVES_DIR/$ARCHIVE_NAME"

echo "[INFO] Chunk ready."

