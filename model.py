import os
import torch
import torch.nn as nn
import torchvision.models as models


NUM_CLASSES = 14

# OPTIONAL: path where pretrained weights MAY exist
PRETRAINED_PATH = "/scratch/chuk303/torch_weights/densenet121-a639ec97.pth"


def create_model():
    """
    Creates DenseNet-121 model.
    - Does NOT attempt internet download
    - Loads pretrained weights only if file exists
    """

    # Always initialize without pretrained weights
    model = models.densenet121(weights=None)

    # Try loading local pretrained weights (if available)
    if os.path.isfile(PRETRAINED_PATH):
        print("[INFO] Loading pretrained DenseNet weights from local file")
        state = torch.load(PRETRAINED_PATH, map_location="cpu")

        # Remove classifier weights if present
        filtered_state = {
            k: v for k, v in state.items()
            if not k.startswith("classifier")
        }

        model.load_state_dict(filtered_state, strict=False)
    else:
        print("[INFO] Training DenseNet from scratch (no pretrained weights)")

    # Replace classifier for NIH dataset
    model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)

    return model
