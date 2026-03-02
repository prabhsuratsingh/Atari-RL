import torch
import numpy as np

from dqn_atari.models.rainbow_dqn import RainbowDQN


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

        eq = (u == l)
        projected[eq, l[eq]] += next_dist[eq, j]

        ne = (u != l)
        projected[ne, l[ne]] += next_dist[ne, j] * (u.float()[ne] - bj[ne])
        projected[ne, u[ne]] += next_dist[ne, j] * (bj[ne] - l.float()[ne])

    return projected


def calc_rainbow_loss(
    batch,
    weights,
    net: RainbowDQN,
    target_net: RainbowDQN,
    gamma,
    device,
):
    states, actions, rewards, dones, next_states = batch_to_tensors(
        batch, device
    )

    dist = net(states)
    dist = dist[range(len(actions)), actions]

    with torch.no_grad():
        next_q = net.q_values(next_states)
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

    loss = -(proj * dist.log()).sum(1)

    if weights is not None:
        loss = loss * weights

    return loss.mean(), loss.detach()