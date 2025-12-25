import os
import tarfile
import shutil
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from dataset import ChestXrayDataset
from model import create_model
from utils_checkpoint import save_checkpoint, load_checkpoint
from performance_logger import PerformanceLogger
from evaluation import evaluate



# ---------------- CONFIG ----------------
ARCHIVE_DIR = "data/archives"
TEMP_DIR = "data/temp/images"
LABELS_CSV = "data/labels.csv"

BATCH_SIZE = 8
EPOCHS_PER_CHUNK = 1
# CHECKPOINT_PATH = "checkpoint.pt"
perf_logger = PerformanceLogger("performance_cpu.csv")


# Windows-safe
NUM_WORKERS = 0
# ----------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


os.makedirs(TEMP_DIR, exist_ok=True)

# List archives
archives = sorted([
    f for f in os.listdir(ARCHIVE_DIR)
    if f.endswith(".tar.gz")
])

if len(archives) == 0:
    raise RuntimeError("No archives found in data/archives")

print(f"Found {len(archives)} archives")

# Model & optimizer
model = create_model().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Resume if checkpoint exists
start_chunk = load_checkpoint(model, optimizer)
print(f"Resuming from chunk index {start_chunk}")


# ---------------- TRAIN LOOP ----------------


for idx in range(start_chunk, len(archives)):
    archive = archives[idx]
    archive_path = os.path.join(ARCHIVE_DIR, archive)

    print(f"\n=== Processing {archive} (chunk {idx+1}/{len(archives)}) ===")

    # 1. Extract
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall("data/temp")

    image_dir = TEMP_DIR

    # 2. Dataset + loader
    train_ds = ChestXrayDataset(
        csv_file=LABELS_CSV,
        image_dir=image_dir,
        split_file="data/split.csv",
        split="train"
    )

    val_ds = ChestXrayDataset(
        csv_file=LABELS_CSV,
        image_dir=image_dir,
        split_file="data/split.csv",
        split="val"
    )

    print(f"Train samples in chunk: {len(train_ds)}")
    print(f"Val samples in chunk: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )




    # 3. Train
    model.train()
    perf_logger.start()

    for epoch in range(EPOCHS_PER_CHUNK):
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(
            f"Chunk {idx+1} | Epoch {epoch+1} | "
            f"Avg Loss: {running_loss/len(train_loader):.4f}"
        )

    elapsed, throughput = perf_logger.end_and_log(
        chunk_idx=idx + 1,
        num_images=len(train_ds)
    )

    print(
        f"[PERF] Chunk {idx+1}: "
        f"{elapsed:.2f}s, "
        f"{throughput:.2f} images/sec"
    )

    # Evaluation on val set
    val_loss, val_auc = evaluate(model, val_loader, criterion)

    print(
        f"[VAL] Chunk {idx+1}: "
        f"Loss = {val_loss:.4f}, "
        f"ROC-AUC = {val_auc:.4f}"
    )




    # 4. Save checkpoint
    save_checkpoint(model, optimizer, idx + 1)
    print("Checkpoint saved")

    # 5. Cleanup
    shutil.rmtree("data/temp")
    os.makedirs(TEMP_DIR, exist_ok=True)
    print("Temporary data deleted")

print("\nAll chunks processed successfully.")
