import torch
import torch.nn as nn
import torch.nn.functional as F


class C51DQN(nn.Module):
    def __init__(self, input_shape, n_actions, n_atoms=51, vmin=-10, vmax=10):
        super().__init__()

        self.n_actions = n_actions
        self.n_atoms = n_atoms
        self.vmin = vmin
        self.vmax = vmax

        self.register_buffer(
            "support", torch.linspace(vmin, vmax, n_atoms)
        )

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 2, 1),
            nn.ReLU(),
            nn.Flatten(),
        )

        conv_out = self.conv(torch.zeros(1, *input_shape)).size(-1)

        self.fc = nn.Sequential(
            nn.Linear(conv_out, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions * n_atoms),
        )

    def forward(self, x):
        x = x / 255.0
        batch = x.size(0)

        logits = self.fc(self.conv(x))
        logits = logits.view(batch, self.n_actions, self.n_atoms)

        probs = F.softmax(logits, dim=2)
        return probs

    def q_values(self, x):
        probs = self.forward(x)
        return torch.sum(probs * self.support, dim=2)