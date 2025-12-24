import os
import tarfile
import shutil
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from dataset import ChunkChestXrayDataset
from model import create_model
from downloader import download_file

# CONFIG
ARCHIVE_URLS = [
    # Example URLs – replace with actual NIH .gz URLs
    "https://nihcc.app.box.com/shared/static/<FILE_ID>.gz"

]

ARCHIVE_DIR = "data/archives"
EXTRACT_DIR = "data/temp"
LABELS_CSV = "data/labels.csv"

os.makedirs(ARCHIVE_DIR, exist_ok=True)
os.makedirs(EXTRACT_DIR, exist_ok=True)

# Load full labels CSV (small file)
labels_df = pd.read_csv(LABELS_CSV)

# Model
model = create_model()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for archive_url in ARCHIVE_URLS:
    archive_name = archive_url.split("/")[-1]
    archive_path = os.path.join(ARCHIVE_DIR, archive_name)

    print(f"\nDownloading {archive_name}")
    download_file(archive_url, archive_path)

    print("Extracting archive...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(EXTRACT_DIR)

    extracted_images = os.listdir(EXTRACT_DIR)

    # Filter labels for current chunk
    chunk_labels = labels_df[labels_df["Image"].isin(extracted_images)]

    dataset = ChunkChestXrayDataset(EXTRACT_DIR, chunk_labels)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    print("Training on current chunk...")
    for epoch in range(1):  # 1 epoch per chunk
        for images, labels in dataloader:
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Chunk loss: {loss.item():.4f}")

    print("Cleaning up disk...")
    shutil.rmtree(EXTRACT_DIR)
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    os.remove(archive_path)

print("\nTraining complete.")
