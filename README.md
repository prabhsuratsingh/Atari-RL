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

## Hyperparameters

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

## Implementations 

### 1. DQN on Pong

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

▶️ [Watch training episode](videos/rl-video-episode-0.mp4)

---

### 2. Double DQN on Pong
```
y = r + γ * Q_target​(s′, argamax​ Q_online​ (s′,a))
```

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
604734: done 296 games, reward 18.940, eps 0.01, speed 104.77 f/s
Best reward updated 18.890 -> 18.940
606364: done 297 games, reward 19.000, eps 0.01, speed 97.85 f/s
Best reward updated 18.940 -> 19.000
607995: done 298 games, reward 19.050, eps 0.01, speed 83.80 f/s
Best reward updated 19.000 -> 19.050
Solved in 607995 frames!
```

Gameplay :

▶️ [Watch training episode](videos/pong-double-dqn.mp4)

---

### 3. Prioritized Replay Buffer DQN on Pong

Convergence:
```pwsh
485171: done 287 games, reward 18.860, eps 0.01, speed 120.07 f/s
Best reward updated 18.790 -> 18.860
486855: done 288 games, reward 18.910, eps 0.01, speed 116.04 f/s
Best reward updated 18.860 -> 18.910
488694: done 289 games, reward 18.950, eps 0.01, speed 115.99 f/s
Best reward updated 18.910 -> 18.950
490377: done 290 games, reward 19.010, eps 0.01, speed 119.78 f/s
Best reward updated 18.950 -> 19.010
Solved in 490377 frames!
```

## All models trained and experimented on :
```
NVIDIA GeForce GTX 1650
4GB VRAM
CUDA Version: 12.5
```

## Roadmap


* [x] Vanilla DQN
* [x] n-Step DQN
* [x] Double DQN
* [x] Noisy Networks
* [x] Prioritized replay buffer
* [x] Dueling DQN
* [x] Categorized DQN (C51)
* [x] Rainbow (Dueling + Noisy + C51)

## References

* Mnih et al., 2013 — Playing Atari with Deep Reinforcement Learning
* Mnih et al., 2015 — Human-level Control Through Deep RL
* Gymnasium Atari / ALE

