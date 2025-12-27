import os
import shutil
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
TEMP_IMAGES_DIR = "/scratch/chuk303/nih_chestxray/data/temp/images"
LABELS_CSV = "data/labels.csv"
SPLIT_FILE = "data/split.csv"

BATCH_SIZE = 8
EPOCHS_PER_CHUNK = 1
NUM_WORKERS = 2

CHECKPOINT_PATH = "checkpoint.pt"
PERF_LOG_PATH = "performance_gpu.csv"
# --------------------------------------


def main():
    # --------- Sanity Check ----------
    if not os.path.isdir(TEMP_IMAGES_DIR):
      print("[INFO] No images staged. Skipping run.")
      return

    num_images = len(os.listdir(TEMP_IMAGES_DIR))
    if num_images == 0:
      print("[INFO] No images staged. Skipping run.")
      return

    print(f"[INFO] Images staged for training: {num_images}")

    # --------- Device ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --------- Model ----------
    model = create_model()
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # --------- Resume ----------
    start_chunk = load_checkpoint(model, optimizer)
    print(f"Resuming from chunk index {start_chunk}")

    # --------- Datasets ----------
    train_ds = ChestXrayDataset(
        csv_file=LABELS_CSV,
        image_dir=TEMP_IMAGES_DIR,
        split_file=SPLIT_FILE,
        split="train"
    )

    val_ds = ChestXrayDataset(
        csv_file=LABELS_CSV,
        image_dir=TEMP_IMAGES_DIR,
        split_file=SPLIT_FILE,
        split="val"
    )

    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # --------- Training ----------
    perf_logger = PerformanceLogger(PERF_LOG_PATH)
    perf_logger.start()

    model.train()

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

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1} | Avg Loss: {avg_loss:.4f}")

    elapsed, throughput = perf_logger.end_and_log(
        chunk_idx=start_chunk + 1,
        num_images=len(train_ds)
    )

    print(f"[PERF] {elapsed:.2f}s | {throughput:.2f} images/sec")

    # --------- Validation ----------
    val_loss, val_auc = evaluate(model, val_loader, criterion)
    print(f"[VAL] Loss = {val_loss:.4f} | ROC-AUC = {val_auc:.4f}")

    # --------- Save ----------
    save_checkpoint(model, optimizer, start_chunk + 1)
    print("Checkpoint saved")

    # --------- Cleanup ----------
    print("[CLEANUP] Removing temp images...")
    shutil.rmtree(TEMP_IMAGES_DIR, ignore_errors=True)
    os.makedirs(TEMP_IMAGES_DIR, exist_ok=True)

    print("\nAll chunks processed successfully.")


if __name__ == "__main__":
    main()

