import time
import gymnasium as gym
import torch
import numpy as np
import argparse
from torch.utils.tensorboard.writer import SummaryWriter
import ale_py

from dqn_atari.agents.dqn_agent import Agent, calc_loss
from dqn_atari.envs.atari_wrappers import make_env
from dqn_atari.models.noisy_dqn import NoisyDQN
from dqn_atari.replay.buffer import ExperienceBuffer

gym.register_envs(ale_py)

ENV_NAME = "PongNoFrameskip-v4"
MEAN_REWARD_BOUND = 19

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000
REPLAY_START_SIZE = 10000

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", default="cpu", help="device name, default=cpu")
    parser.add_argument("--env", default=ENV_NAME, help="Name of environment")
    args = parser.parse_args()

    device = torch.device(args.dev)
    print(f"Device : {device}")

    env = make_env(args.env)
    net = NoisyDQN(env.observation_space.shape, env.action_space.n).to(device)
    target_net = NoisyDQN(env.observation_space.shape, env.action_space.n).to(device)

    writer = SummaryWriter(comment="-" + args.env + "-noisy_dqn")
    print(net)

    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)

    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_m_reward = None

    while True:
        frame_idx += 1

        net.reset_noise()
        reward = agent.play_step(net, device, epsilon=0.0)

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
                torch.save(net.state_dict(), f"checkpoints/{args.env}-noisy_dqn-best_{m_reward:.0f}.dat")
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

        net.reset_noise()    
        target_net.reset_noise()

        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_t = calc_loss(batch, net, target_net, device)
        loss_t.backward()
        optimizer.step()
    
    writer.close()