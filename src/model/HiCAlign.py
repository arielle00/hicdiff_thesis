import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleHiCAlign(nn.Module):
    def __init__(self):
        super(SimpleHiCAlign, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.tanh(self.conv3(x))  # Ensure output is in range [-1, 1]
        return x
