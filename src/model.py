import torch
import torch.nn as nn
import torch.optim as optim


class QLearningDLModel(nn.Module):
    def __init__(self, img_channels, num_actions, img_cols, img_rows):
        super(QLearningDLModel, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=img_channels, out_channels=32, kernel_size=8, stride=4, padding=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1536, 512)  # Adjusted for the output size
        self.fc2 = nn.Linear(512, num_actions)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
