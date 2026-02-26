import gymnasium as gym 
import argparse
import numpy as np
import typing as tt
import torch
import collections
import ale_py

from dqn_atari.envs.atari_wrappers import make_env
from dqn_atari.models.dqn import DQN

gym.register_envs(ale_py)

ENV_NAME = "PongNoFrameskip-v4"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Model file to load")
    parser.add_argument("-e", "--env", default=ENV_NAME, help="Name of environment")
    parser.add_argument("-r", "--record", required=True, help="Directory for video recording")
    args = parser.parse_args()

    env = make_env(args.env, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, video_folder=args.record)
    net = DQN(env.observation_space.shape, env.action_space.n)
    state = torch.load(args.model, map_location=lambda stg, _: stg)
    net.load_state_dict(state)

    state, _ = env.reset()
    total_reward = 0.0
    c: tt.Dict[int,int] = collections.Counter()

    while True:
        state_v = torch.as_tensor(state).unsqueeze(0)
        q_vals = net(state_v).detach().cpu().numpy()[0]
        action = int(np.argmax(q_vals))
        c[action] += 1

        state, reward, is_done, is_trunc, _ = env.step(action)
        total_reward += reward

        if is_done or is_trunc:
            break
    
    print(f"Total reward: {total_reward:.2f}")
    print(f"Action counts: {c}")

    env.close()