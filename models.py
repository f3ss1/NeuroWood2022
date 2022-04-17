import torchvision
import torch.nn as nn


class Effnetb3(nn.Module):
    def __init__(self):
        super(Effnetb3, self).__init__()
        self.main_model = torchvision.models.efficientnet_b3(pretrained=True)
        fc = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 3),
            nn.Softmax(dim=1)
        )
        self.main_model.fc = fc

    def forward(self, x):
        return self.main_model.forward(x)
