import pandas as pd
import numpy as np

# Load original NIH CSV (NOT labels.csv)
df = pd.read_csv("data/Data_Entry_2017_v2020.csv")

# Keep only what we need
df = df[["Image Index", "Patient ID"]]
df = df.rename(columns={"Image Index": "Image"})

# Unique patients
patients = df["Patient ID"].unique()
np.random.seed(42)
np.random.shuffle(patients)

n = len(patients)
train_patients = set(patients[:int(0.7 * n)])
val_patients   = set(patients[int(0.7 * n):int(0.85 * n)])
test_patients  = set(patients[int(0.85 * n):])

def assign_split(pid):
    if pid in train_patients:
        return "train"
    elif pid in val_patients:
        return "val"
    else:
        return "test"

df["split"] = df["Patient ID"].apply(assign_split)

# Save mapping
df[["Image", "split"]].to_csv("data/split.csv", index=False)

print("Split created:")
print(df["split"].value_counts())
