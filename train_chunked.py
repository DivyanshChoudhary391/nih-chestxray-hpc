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


# ---------------- CONFIG ----------------
ARCHIVE_DIR = "data/archives"
TEMP_DIR = "data/temp/images"
LABELS_CSV = "data/labels.csv"

BATCH_SIZE = 8
EPOCHS_PER_CHUNK = 1
CHECKPOINT_PATH = "checkpoint.pt"

# Windows-safe
NUM_WORKERS = 0
# ----------------------------------------

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
model = create_model()
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
    dataset = ChestXrayDataset(
        csv_file=LABELS_CSV,
        image_dir=image_dir
    )

    print(f"Images in this chunk: {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    # 3. Train
    model.train()
    for epoch in range(EPOCHS_PER_CHUNK):
        running_loss = 0.0
        for images, labels in dataloader:
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Chunk {idx+1} | Epoch {epoch+1} | Avg Loss: {running_loss/len(dataloader):.4f}")

    # 4. Save checkpoint
    save_checkpoint(model, optimizer, idx + 1)
    print("Checkpoint saved")

    # 5. Cleanup
    shutil.rmtree("data/temp")
    os.makedirs(TEMP_DIR, exist_ok=True)
    print("Temporary data deleted")

print("\nAll chunks processed successfully.")
