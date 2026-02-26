import typing as tt
import torch
import collections
import ale_py
import argparse
import gymnasium as gym
import numpy as np
import os
import glob
import shutil
import time

from dqn_atari.envs.atari_wrappers import make_env
from dqn_atari.models.dqn import DQN

gym.register_envs(ale_py)

ENV_NAME = "PongNoFrameskip-v4"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("-e", "--env", default=ENV_NAME)
    parser.add_argument("-r", "--record", required=True)
    parser.add_argument("-n", "--video-name", required=True)
    args = parser.parse_args()

    os.makedirs(args.record, exist_ok=True)

    # mark start time
    start_time = time.time()

    env = make_env(args.env, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, video_folder=args.record)

    net = DQN(env.observation_space.shape, env.action_space.n)
    net.load_state_dict(torch.load(args.model, map_location="cpu"))

    state, _ = env.reset()
    total_reward = 0.0
    c: tt.Dict[int, int] = collections.Counter()

    while True:
        state_v = torch.as_tensor(state).unsqueeze(0)
        q_vals = net(state_v).detach().cpu().numpy()[0]
        action = int(np.argmax(q_vals))
        c[action] += 1

        state, reward, is_done, is_trunc, _ = env.step(action)
        total_reward += reward

        if is_done or is_trunc:
            break

    env.close()

    # only videos created in THIS run
    pattern = os.path.join(args.record, "rl-video-*.mp4")
    candidates = [f for f in glob.glob(pattern) if os.path.getmtime(f) >= start_time]

    if candidates:
        src = max(candidates, key=os.path.getmtime)
        dst = os.path.join(args.record, args.video_name + ".mp4")
        shutil.move(src, dst)
        print(f"Saved video: {dst}")
    else:
        print("No new video found!")

    print(f"Total reward: {total_reward:.2f}")
    print(f"Action counts: {c}")

if __name__ == "__main__":
    main()