import torch
import torch.nn as nn


class DuelingDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        conv_out = self.conv(torch.zeros(1, *input_shape)).size(-1)

        self.value = nn.Sequential(
            nn.Linear(conv_out, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

        self.advantage = nn.Sequential(
            nn.Linear(conv_out, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        x = x / 255.0
        features = self.conv(x)

        value = self.value(features)
        advantage = self.advantage(features)

        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q