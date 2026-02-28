import time
import gymnasium as gym
import torch
import numpy as np
import argparse
from torch.utils.tensorboard.writer import SummaryWriter
import ale_py

from dqn_atari.agents.dqn_agent import Agent
from dqn_atari.agents.per_dqn_agent import calc_per_loss
from dqn_atari.envs.atari_wrappers import make_env
from dqn_atari.models.dqn import DQN
from dqn_atari.replay.prioritized_buffer import PrioritizedReplayBuffer

gym.register_envs(ale_py)

ENV_NAME = "PongNoFrameskip-v4"
MEAN_REWARD_BOUND = 19

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000
REPLAY_START_SIZE = 10000

EPSILON_DECAY_LAST_FRAME = 150000
EPSILON_START = 1.0
EPSILON_FINAL = 0.01

BETA_START = 0.4
BETA_FRAMES = 200000

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", default="cpu", help="device name, default=cpu")
    parser.add_argument("--env", default=ENV_NAME, help="Name of environment")
    args = parser.parse_args()

    device = torch.device(args.dev)
    print(f"Device : {device}")

    env = make_env(args.env)
    net = DQN(env.observation_space.shape, env.action_space.n).to(device)
    target_net = DQN(env.observation_space.shape, env.action_space.n).to(device)

    writer = SummaryWriter(comment="-" + args.env + "-per_dqn")
    print(net)

    buffer = PrioritizedReplayBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START

    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_m_reward = None

    while True:
        frame_idx += 1
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)
        beta = min(1.0, BETA_START + frame_idx * (1.0 - BETA_START) / BETA_FRAMES)

        reward = agent.play_step(net, device, epsilon)
        if reward is not None:
            total_rewards.append(reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            m_reward = np.mean(total_rewards[-100:])

            print(f"{frame_idx}: done {len(total_rewards)} games, reward {m_reward:.3f}, eps {epsilon:.2f}, speed {speed:.2f} f/s")
            writer.add_scalar("epsilon", epsilon, frame_idx)
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("reward_100", m_reward, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)

            if best_m_reward is None or best_m_reward < m_reward:
                torch.save(net.state_dict(), f"checkpoints/{args.env}-per_dqn-best_{m_reward:.0f}.dat")
                if best_m_reward is not None:
                    print(f"Best reward updated {best_m_reward:.3f} -> {m_reward:.3f}")
                best_m_reward = m_reward
            
            if m_reward > MEAN_REWARD_BOUND:
                print("Solved in %d frames!" % frame_idx)
                break
        
        if len(buffer) < REPLAY_START_SIZE:
            continue
        if frame_idx % SYNC_TARGET_FRAMES == 0:
            target_net.load_state_dict(net.state_dict())
        
        optimizer.zero_grad()

        batch, indices, weights = buffer.sample(BATCH_SIZE, beta)
        weights = torch.as_tensor(weights, device=device)

        loss_t, td_errors = calc_per_loss(
            batch, weights, net, target_net, GAMMA, device
        )

        new_prios = td_errors.detach().cpu().numpy() + 1e-5
        buffer.update_priorities(indices, new_prios)

        loss_t.backward()
        optimizer.step()
    
    writer.close()