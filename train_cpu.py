import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from dataset import ChestXrayDataset
from model import create_model

# Load dataset
dataset = ChestXrayDataset(
    csv_file="data/labels.csv",
    image_dir="data/images"
)

dataloader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True
)

# Model
model = create_model()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
for epoch in range(2):
    for images, labels in dataloader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")
