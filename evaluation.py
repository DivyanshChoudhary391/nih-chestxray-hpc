import torch
import numpy as np
from sklearn.metrics import roc_auc_score

@torch.no_grad()
def evaluate(model, dataloader, criterion):
    model.eval()

    total_loss = 0.0
    all_targets = []
    all_outputs = []

    for images, labels in dataloader:
        images = images.to(next(model.parameters()).device)
        labels = labels.to(next(model.parameters()).device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        total_loss += loss.item()

        all_targets.append(labels.cpu().numpy())
        all_outputs.append(torch.sigmoid(outputs).cpu().numpy())

    avg_loss = total_loss / len(dataloader)

    y_true = np.vstack(all_targets)
    y_pred = np.vstack(all_outputs)

    # Some classes may have only one label in a chunk → handle safely
    try:
        auc = roc_auc_score(y_true, y_pred, average="macro")
    except ValueError:
        auc = float("nan")

    return avg_loss, auc
