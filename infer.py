import torch
import cv2
import numpy as np
from torchvision import transforms

from model import create_model

# ---------------- CONFIG ----------------
CHECKPOINT_PATH = "/scratch/chuk303/nih_chestxray/checkpoints/checkpoint/latest.pt"
LABELS_FILE = "labels.txt"
IMAGE_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ---------------------------------------


def load_labels():
    with open(LABELS_FILE, "r") as f:
        return [line.strip() for line in f.readlines()]


def load_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image not found or unreadable")

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    img = transform(img)
    img = img.unsqueeze(0)  # batch dimension
    return img


def load_model():
    model = create_model()
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

    # Handles DDP / non-DDP checkpoints
    state_dict = checkpoint["model_state"]
    new_state = {}

    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state[k.replace("module.", "")] = v
        else:
            new_state[k] = v

    model.load_state_dict(new_state)
    model.to(DEVICE)
    model.eval()
    return model


def main(image_path):
    labels = load_labels()
    model = load_model()
    image = load_image(image_path).to(DEVICE)

    with torch.no_grad():
        logits = model(image)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    print("\n===== Prediction Results =====")
    for label, prob in zip(labels, probs):
        print(f"{label:20s}: {prob:.4f}")

    print("\n===== Most Likely Findings =====")
    for label, prob in zip(labels, probs):
        if prob > 0.5:
            print(f"⚠ {label} ({prob:.2f})")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python infer.py <path_to_xray_image>")
        exit(1)

    main(sys.argv[1])

