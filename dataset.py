import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2
import os
import numpy as np

class ChestXrayDataset(Dataset):
    def __init__(self, csv_file, image_dir):
        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir

        # Keep only images that exist in image_dir
        image_files = set(os.listdir(image_dir))
        self.df = self.df[self.df["Image"].isin(image_files)]

        self.label_cols = self.df.columns[1:]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = os.path.join(self.image_dir, row["Image"])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0

        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)

        label = row[self.label_cols].astype("float32").values
        label = torch.tensor(label)

        return img, label

