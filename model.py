import torch.nn as nn
import torchvision.models as models

def create_model(num_classes=14):
    model = models.densenet121(pretrained=None)
    model.classifier = nn.Linear(
        model.classifier.in_features,
        num_classes
    )
    return model
