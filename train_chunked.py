import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
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

BATCH_SIZE = 32
EPOCHS_PER_CHUNK = 3
NUM_WORKERS = 2

CHECKPOINT_PATH = "checkpoint.pt"
PERF_LOG_PATH = "performance_gpu.csv"
# --------------------------------------


def main():
    # --------- DDP INIT ----------
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    rank = dist.get_rank()
    is_main_process = rank == 0

    # --------- SANITY CHECK ----------
    if not os.path.isdir(TEMP_IMAGES_DIR):
        if is_main_process:
            print("[INFO] No images staged. Skipping run.")
        dist.destroy_process_group()
        return

    num_images = len(os.listdir(TEMP_IMAGES_DIR))
    if num_images == 0:
        if is_main_process:
            print("[INFO] No images staged. Skipping run.")
        dist.destroy_process_group()
        return

    if is_main_process:
        print(f"[INFO] Images staged for training: {num_images}")

    # --------- MODEL ----------
    model = create_model().to(device)
    model = DDP(model, device_ids=[local_rank])

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # --------- RESUME ----------
    start_chunk = load_checkpoint(model.module, optimizer)
    if is_main_process:
        print(f"Resuming from chunk index {start_chunk}")

    # --------- DATASETS ----------
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

    if is_main_process:
        print(f"Train samples: {len(train_ds)}")
        print(f"Val samples: {len(val_ds)}")

    train_sampler = DistributedSampler(train_ds)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
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

    # --------- PERFORMANCE LOGGER ----------
    if is_main_process:
        perf_logger = PerformanceLogger(PERF_LOG_PATH)
        perf_logger.start()

    # --------- TRAINING ----------
    model.train()

    for epoch in range(EPOCHS_PER_CHUNK):
        train_sampler.set_epoch(epoch)
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        if is_main_process:
            print(f"Epoch {epoch + 1} | Avg Loss: {avg_loss:.4f}")

    # --------- PERFORMANCE ----------
    if is_main_process:
        elapsed, throughput = perf_logger.end_and_log(
            chunk_idx=start_chunk + 1,
            num_images=len(train_ds)
        )
        print(f"[PERF] {elapsed:.2f}s | {throughput:.2f} images/sec")

    # --------- VALIDATION (RANK 0 ONLY) ----------
    if is_main_process:
        val_loss, val_auc = evaluate(model.module, val_loader, criterion)
        print(f"[VAL] Loss = {val_loss:.4f} | ROC-AUC = {val_auc:.4f}")

    # --------- SAVE CHECKPOINT ----------
    if is_main_process:
        save_checkpoint(model.module, optimizer, start_chunk + 1)
        print("Checkpoint saved")

    # --------- CLEANUP ----------
    dist.barrier()

    if is_main_process:
        print("[CLEANUP] Removing temp images...")
        shutil.rmtree(TEMP_IMAGES_DIR, ignore_errors=True)
        os.makedirs(TEMP_IMAGES_DIR, exist_ok=True)

        print("\nAll chunks processed successfully.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
