import typing as tt
import numpy as np
import torch

from dqn_atari.replay.buffer import Experience


class RainbowAgent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self.state = None
        self._reset()

    def _reset(self):
        self.state, _ = self.env.reset()
        self.total_reward = 0.0

    @torch.no_grad()
    def play_step(self, net, device):
        net.reset_noise()

        state_v = torch.as_tensor(self.state).to(device).unsqueeze(0)
        q_vals = net.q_values(state_v)
        action = int(torch.argmax(q_vals, dim=1).item())

        next_state, reward, done, trunc, _ = self.env.step(action)
        self.total_reward += reward

        exp = Experience(
            state=self.state,
            action=action,
            reward=float(reward),
            done_trunc=done or trunc,
            next_state=next_state,
        )

        self.exp_buffer.append(exp)
        self.state = next_state

        if done or trunc:
            done_reward = self.total_reward
            self._reset()
            return done_reward

        return None