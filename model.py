import torchvision.models as models
import torch.nn as nn
import torch

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.network = nn.Sequential(
            models.densenet121(pretrained=True),
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 17),
            nn.Softmax()
        )

    def forward(self, images):
        return self.network(images).squeeze(1)

