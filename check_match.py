import os
import pandas as pd

labels = pd.read_csv("data/labels.csv")
image_dir = "data/temp/images"

image_files = set(os.listdir(image_dir))
csv_images = set(labels["Image"])

common = image_files.intersection(csv_images)

print("Images in folder:", len(image_files))
print("Images in CSV:", len(csv_images))
print("Matching images:", len(common))
print("Sample matches:", list(common)[:5])
