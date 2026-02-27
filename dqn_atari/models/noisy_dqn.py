import torch
import torch.nn as nn

from dqn_atari.models.noisy_layer import NoisyLinear


class NoisyDQN(nn.Module):
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

        size = self.conv(torch.zeros(1, *input_shape)).size()[-1]

        self.fc = nn.Sequential(
            NoisyLinear(size, 512),
            nn.ReLU(),
            NoisyLinear(512, n_actions),
        )

    def forward(self, x):
        x = x / 255.0
        return self.fc(self.conv(x))

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()