import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import typing as tt

from dqn_atari.models.dqn import DQN
from dqn_atari.replay.buffer import Experience, ExperienceBuffer

GAMMA = 0.99

State = np.ndarray
Action = int
BatchTensors = tt.Tuple[
    torch.ByteTensor,
    torch.LongTensor,
    torch.Tensor,
    torch.BoolTensor,
    torch.ByteTensor
]

class Agent:
    def __init__(self, env: gym.Env, exp_buffer: ExperienceBuffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self.state:  tt.Optional[np.ndarray] = None
        self._reset()

    def _reset(self):
        self.state, _ = self.env.reset()
        self.total_reward = 0.0

    @torch.no_grad()
    def play_step(
        self,
        net: DQN,
        device: torch.device,
        epsilon: float = 0.0
    ) -> tt.Optional[float] :
        done_reward = None

        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state_v = torch.as_tensor(self.state).to(device)
            state_v.unsqueeze_(0)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())
        
        new_state, reward, is_done, is_trunc, _ = self.env.step(action)
        self.total_reward += reward

        exp = Experience(
            state=self.state, 
            action=action,
            reward=float(reward),
            done_trunc=is_done or is_trunc,
            new_state=new_state
        )

        self.exp_buffer.append(exp)
        self.state = new_state

        if is_done or is_trunc:
            done_reward = self.total_reward
            self._reset()


        return done_reward
    
def batch_to_tensors(batch: tt.List[Experience], device: torch.device) -> BatchTensors:
    states, actions, rewards, dones, new_state = [], [], [], [], []

    for e in batch:
        states.append(e.state)
        actions.append(e.action)
        rewards.append(e.reward)
        dones.append(e.done_trunc)
        new_state.append(e.new_state)
    
    states_t = torch.as_tensor(np.asarray(states))
    actions_t = torch.LongTensor(actions)
    rewards_t = torch.FloatTensor(rewards)
    dones_t = torch.BoolTensor(dones)
    new_states_t = torch.as_tensor(np.asarray(new_state))

    return states_t.to(device), actions_t.to(device), rewards_t.to(device), dones_t.to(device), new_states_t.to(device)

def calc_loss(
        batch: tt.List[Experience],
        net: DQN,
        target_net: DQN,
        device: torch.device
) -> torch.Tensor :
    states_t, actions_t, rewards_t, dones_t, new_states_t = batch_to_tensors(batch, device)

    state_action_values = net(states_t).gather(
        1, actions_t.unsqueeze(-1)
    ).squeeze(-1)

    with torch.no_grad():
        next_state_values = target_net(new_states_t).max(1)[0]
        next_state_values[dones_t] = 0.0
        next_state_values = next_state_values.detach()
    
    expected_state_action_values = next_state_values * GAMMA + rewards_t

    return nn.MSELoss()(state_action_values, expected_state_action_values)