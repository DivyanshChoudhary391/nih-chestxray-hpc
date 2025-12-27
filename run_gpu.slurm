#!/bin/bash
#SBATCH --job-name=nih_xray_gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/gpu_%j.out
#SBATCH --error=logs/gpu_%j.err

module purge
module load python/conda-python/3.7
module load cuda

cd ~/nih-chestxray-hpc
source venv/bin/activate

echo "Downloading dataset chunk from Google Drive..."

set -a
source .env
set +a


for FILE_ID in $NIH_DRIVE_FILE_IDS; do
    echo "Processing Drive file: $FILE_ID"
    bash download_from_drive.sh "$FILE_ID"
    python train_chunked.py
done



python train_chunked.py

