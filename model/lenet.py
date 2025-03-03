import torch
from torch import nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    def __init__(self, num_classes):
        super(LeNet5, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
        )
        self.subsample1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.subsample2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.L1 = nn.Linear(16 * 4 * 4, 120)
        self.relu = nn.ReLU()
        self.L2 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.L3 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.subsample1(out)
        out = self.layer2(out)
        out = self.subsample2(out)
        out = out.view(out.size(0), -1)

        out = self.L1(out)
        out = self.relu(out)
        out = self.L2(out)
        out = self.relu1(out)
        out = self.L3(out)

        return out
