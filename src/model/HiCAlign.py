import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleHiCAlign(nn.Module):
    def __init__(self):
        super(SimpleHiCAlign, self).__init__()

        # Increase network capacity
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)  # More filters
        self.bn1 = nn.BatchNorm2d(32)  
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)  
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 1, kernel_size=1)  # Reduce to 1 channel

    def forward(self, x):
        residual = x  # Store original input for residual learning

        x = F.relu(self.bn1(self.conv1(x)))  
        x = F.relu(self.bn2(self.conv2(x)))  
        x = F.relu(self.bn3(self.conv3(x)))  
        x = torch.tanh(self.conv4(x))  # Ensure output is in [-1,1]

        return residual + x  # Residual learning (helps stabilize training)

if __name__ == "__main__":
    # Quick test
    model = SimpleHiCAlign()
    sample_input = torch.randn(1, 1, 64, 64)  # Simulated Hi-C matrix
    output = model(sample_input)

    # print(f"Input  shape: {sample_input.shape}, min={sample_input.min()}, max={sample_input.max()}")
    # print(f"Output shape: {output.shape}, min={output.min()}, max={output.max()}")
