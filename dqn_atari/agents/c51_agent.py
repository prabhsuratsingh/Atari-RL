import torch
import numpy as np
import typing as tt

from dqn_atari.models.c51_dqn import C51DQN
from dqn_atari.replay.buffer import Experience, ExperienceBuffer


class C51Agent:
    def __init__(self, env, exp_buffer: ExperienceBuffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self.state: tt.Optional[np.ndarray] = None
        self._reset()

    def _reset(self):
        self.state, _ = self.env.reset()
        self.total_reward = 0.0

    @torch.no_grad()
    def play_step(
        self,
        net: C51DQN,
        device: torch.device,
        epsilon: float = 0.0,
    ) -> tt.Optional[float]:

        done_reward = None

        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state_v = torch.as_tensor(self.state).to(device).unsqueeze(0)

            # use expected Q from distribution
            q_vals_v = net.q_values(state_v)

            action = int(torch.argmax(q_vals_v, dim=1).item())

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


def batch_to_tensors(batch, device):
    states, actions, rewards, dones, next_states = [], [], [], [], []

    for e in batch:
        states.append(e.state)
        actions.append(e.action)
        rewards.append(e.reward)
        dones.append(e.done_trunc)
        next_states.append(e.next_state)

    states = torch.as_tensor(np.asarray(states)).to(device)
    actions = torch.LongTensor(actions).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    dones = torch.BoolTensor(dones).to(device)
    next_states = torch.as_tensor(np.asarray(next_states)).to(device)

    return states, actions, rewards, dones, next_states


def projection(next_dist, rewards, dones, gamma, support, vmin, vmax, n_atoms):
    batch_size = rewards.size(0)
    delta_z = (vmax - vmin) / (n_atoms - 1)

    projected = torch.zeros_like(next_dist)

    for j in range(n_atoms):
        tz = rewards + (1 - dones.float()) * gamma * support[j]
        tz = tz.clamp(vmin, vmax)

        bj = (tz - vmin) / delta_z
        l = bj.floor().long()
        u = bj.ceil().long()

        eq_mask = (u == l)
        projected[eq_mask, l[eq_mask]] += next_dist[eq_mask, j]

        ne_mask = (u != l)
        projected[ne_mask, l[ne_mask]] += next_dist[ne_mask, j] * (
            u.float()[ne_mask] - bj[ne_mask]
        )
        projected[ne_mask, u[ne_mask]] += next_dist[ne_mask, j] * (
            bj[ne_mask] - l.float()[ne_mask]
        )

    return projected


def calc_c51_loss(batch, net: C51DQN, target_net: C51DQN, gamma, device):
    states, actions, rewards, dones, next_states = batch_to_tensors(
        batch, device
    )

    dist = net(states)
    dist = dist[range(len(actions)), actions]

    with torch.no_grad():
        next_q = target_net.q_values(next_states)
        next_actions = next_q.argmax(1)

        next_dist = target_net(next_states)
        next_dist = next_dist[range(len(actions)), next_actions]

        proj = projection(
            next_dist,
            rewards,
            dones,
            gamma,
            net.support,
            net.vmin,
            net.vmax,
            net.n_atoms,
        )

    loss = -(proj * dist.log()).sum(1).mean()
    return loss