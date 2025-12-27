import torch
import os
import json

CHECKPOINT_DIR = "/scratch/chuk303/nih_chestxray/checkpoints/checkpoint"
LATEST_PATH = os.path.join(CHECKPOINT_DIR, "latest.pt")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def save_checkpoint(model, optimizer, chunk_idx):
    path = os.path.join(CHECKPOINT_DIR, f"chunk_{chunk_idx:03d}.pt")

    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "chunk_idx": chunk_idx
    }, path)

    # Update latest pointer
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "chunk_idx": chunk_idx
    }, LATEST_PATH)

    # Save metadata (optional but good for report)
    meta = {
        "last_completed_chunk": chunk_idx
    }
    with open(os.path.join(CHECKPOINT_DIR, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

def load_checkpoint(model, optimizer):
    if not os.path.exists(LATEST_PATH):
        return 0

    ckpt = torch.load(LATEST_PATH, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt["chunk_idx"]


