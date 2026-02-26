import collections
from dataclasses import dataclass
import typing as tt
import numpy as np

State = np.ndarray
Action = int

@dataclass
class NStepExperience:
    state: State
    action: Action
    reward: float
    done_trunc: bool
    next_state: State


class NStepBuffer:
    def __init__(self, n: int, gamma: float, capacity: int):
        self.n = n
        self.gamma = gamma
        self.buffer = collections.deque(maxlen=capacity)
        self.nstep_queue = collections.deque(maxlen=n)

    def __len__(self):
        return len(self.buffer)

    def _get_nstep(self):
        reward, next_state, done = 0.0, None, False

        for i, exp in enumerate(self.nstep_queue):
            reward += (self.gamma ** i) * exp.reward
            next_state = exp.next_state
            done = exp.done_trunc
            if done:
                break

        first = self.nstep_queue[0]
        return NStepExperience(
            state=first.state,
            action=first.action,
            reward=reward,
            done_trunc=done,
            next_state=next_state,
        )

    def append(self, exp):
        self.nstep_queue.append(exp)

        if len(self.nstep_queue) < self.n:
            return

        nstep_exp = self._get_nstep()
        self.buffer.append(nstep_exp)

        if self.nstep_queue[0].done_trunc:
            self.nstep_queue.clear()

    def sample(self, batch_size):
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in idx]