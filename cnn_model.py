import torch
import torch.nn as nn


class ActivityCNN(nn.Module):
    def __init__(self):
        super(ActivityCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.AvgPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.AvgPool1d(kernel_size=2)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(64, 50)
        self.fc2 = nn.Linear(50, 6)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.global_pool(x).squeeze(-1)
        x = torch.relu(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=1)
