import torch
import torch.nn as nn
import numpy as np

from dqn_atari.models.dqn import DQN


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


def calc_double_dqn_loss(batch, net: DQN, target_net: DQN, gamma: float, device):
    states, actions, rewards, dones, next_states = batch_to_tensors(batch, device)

    q_values = net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

    with torch.no_grad():
        next_actions = net(next_states).argmax(1)

        next_q = target_net(next_states).gather(
            1, next_actions.unsqueeze(-1)
        ).squeeze(-1)

        next_q[dones] = 0.0

    target = rewards + gamma * next_q
    return nn.MSELoss()(q_values, target)