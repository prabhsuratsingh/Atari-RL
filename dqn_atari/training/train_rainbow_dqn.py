import time
import gymnasium as gym
import torch
import numpy as np
import argparse
from torch.utils.tensorboard.writer import SummaryWriter
import ale_py

from dqn_atari.agents.rainbow_agent import RainbowAgent
from dqn_atari.agents.rainbow_loss import calc_rainbow_loss
from dqn_atari.envs.atari_wrappers import make_env
from dqn_atari.models.rainbow_dqn import RainbowDQN
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
    net = RainbowDQN(env.observation_space.shape, env.action_space.n).to(device)
    target_net = RainbowDQN(env.observation_space.shape, env.action_space.n).to(device)

    writer = SummaryWriter(comment="-" + args.env + "-rainbow_dqn")
    print(net)

    buffer = PrioritizedReplayBuffer(REPLAY_SIZE)
    agent = RainbowAgent(env, buffer)

    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_m_reward = None

    while True:
        frame_idx += 1

        beta = min(
            1.0,
            BETA_START + frame_idx * (1.0 - BETA_START) / BETA_FRAMES
        )

        net.reset_noise()
        target_net.reset_noise()
        reward = agent.play_step(net, device)

        if reward is not None:
            total_rewards.append(reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            m_reward = np.mean(total_rewards[-100:])

            print(f"{frame_idx}: done {len(total_rewards)} games, reward {m_reward:.3f}, speed {speed:.2f} f/s")

            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("reward_100", m_reward, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)

            if best_m_reward is None or best_m_reward < m_reward:
                torch.save(
                    net.state_dict(),
                    f"checkpoints/{args.env}-rainbow_dqn-best_{m_reward:.0f}.dat"
                )
                best_m_reward = m_reward

            if m_reward > MEAN_REWARD_BOUND:
                print(f"Solved in {frame_idx} frames!")
                break

        if len(buffer) < REPLAY_START_SIZE:
            continue

        if frame_idx % SYNC_TARGET_FRAMES == 0:
            target_net.load_state_dict(net.state_dict())

        net.reset_noise()
        target_net.reset_noise()

        batch, indices, weights = buffer.sample(BATCH_SIZE, beta)
        weights = torch.FloatTensor(weights).to(device)

        optimizer.zero_grad()
        loss_t, per_sample = calc_rainbow_loss(
            batch, weights, net, target_net, GAMMA, device
        )
        loss_t.backward()
        optimizer.step()

        buffer.update_priorities(
            indices,
            per_sample.detach().cpu().numpy() + 1e-6
        )
    
    writer.close()