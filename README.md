# Atari-RL
## Deep Q-Networks on Atari

Implementation of Deep Q-Networks (DQN) for Atari 2600 environments using the Arcade Learning Environment (ALE) via Gymnasium.
This project reproduces the core DQN pipeline and demonstrates training on Atari Games.

---

## Overview

This repository implements the Deep Q-Network algorithm introduced in:

> Mnih et al., *Playing Atari with Deep Reinforcement Learning*, 2013, [Arxiv](https://arxiv.org/pdf/1312.5602)
>
> Mnih et al., *Human-level control through deep reinforcement learning*, Nature 2015, [Paper](https://training.incf.org/sites/default/files/2023-05/Human-level%20control%20through%20deep%20reinforcement%20learning.pdf)

The agent learns directly from pixel observations using convolutional neural networks and experience replay.

Current status:

* DQN implemented
* Training completed on Pong
* Evaluation gameplay recorded

## Features

* Atari preprocessing (frame stacking, grayscale, resizing)
* Experience replay buffer
* Target network
* Epsilon-greedy exploration schedule
* Training and evaluation scripts
* Gameplay recording and export

1. DQN on Atari Pong
Hyperparameters :
```python
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
```

Neural Network Architecture:
```python
DQN(
  (conv): Sequential(
    (0): Conv2d(4, 32, kernel_size=(8, 8), stride=(4, 4))
    (1): ReLU()
    (2): Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))
    (3): ReLU()
    (4): Conv2d(64, 64, kernel_size=(2, 2), stride=(1, 1))
    (5): ReLU()
    (6): Flatten(start_dim=1, end_dim=-1)
  )
  (fc): Sequential(
    (0): Linear(in_features=4096, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=6, bias=True)
  )
)
```
Loss:

```
L = (r + γ max_a' Q_target(s', a') − Q(s,a))²
```

Convergence :
```pwsh
691306: done 362 games, reward 18.980, eps 0.01, speed 130.05 f/s
692979: done 363 games, reward 18.990, eps 0.01, speed 128.31 f/s
695345: done 364 games, reward 19.090, eps 0.01, speed 132.28 f/s
Best reward updated 18.990 -> 19.090
Solved in 695345 frames!
```

Gameplay :

<video src="videos/rl-video-episode-0.mp4" controls width="600"></video>


All models trained and experimented on :
```
NVIDIA GeForce GTX 1650
4GB VRAM
CUDA Version: 12.5
```

## Roadmap

* Double DQN
* Dueling architecture
* Prioritized replay
* Multi-game training
* TensorBoard logging

## References

* Mnih et al., 2013 — Playing Atari with Deep Reinforcement Learning
* Mnih et al., 2015 — Human-level Control Through Deep RL
* Gymnasium Atari / ALE

