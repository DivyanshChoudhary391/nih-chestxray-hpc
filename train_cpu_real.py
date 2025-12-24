import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from dataset import ChestXrayDataset
from model import create_model

dataset = ChestXrayDataset(
    csv_file="data/labels.csv",
    image_dir="data/temp/images"
)

print("Training samples:", len(dataset))

dataloader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0
)

model = create_model()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(2):
    running_loss = 0.0

    for images, labels in dataloader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()


    print(f"Epoch {epoch+1} | Avg Loss: {running_loss / len(dataloader):.4f}")
