#!/bin/bash
#SBATCH --job-name=nih_xray_ddp
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00

#SBATCH --output=logs/ddp_%j.out
#SBATCH --error=logs/ddp_%j.err

# ---------------- MODULES ----------------
module purge
module load python/conda-python/3.7
module load cuda

# ---------------- ENV ----------------
cd ~/nih-chestxray-hpc
source venv/bin/activate

export OMP_NUM_THREADS=2
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN

# ---------------- LOAD DATASET IDS ----------------
set -a
source .env
set +a

echo "Using NIH_DRIVE_FILE_IDS=$NIH_DRIVE_FILE_IDS"

# ---------------- PIPELINE LOOP ----------------
for FILE_ID in $NIH_DRIVE_FILE_IDS; do
    echo "=============================================="
    echo "Processing Drive file: $FILE_ID"
    echo "=============================================="

    # Download + extract (your existing script)
    bash download_from_drive.sh "$FILE_ID"

    # Multi-GPU training using SAME train_chunked.py
    torchrun --nproc_per_node=2 train_chunked.py
done

echo "ALL CHUNKS COMPLETED"
