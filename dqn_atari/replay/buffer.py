import collections
from dataclasses import dataclass
import torch
import numpy as np
import typing as tt

State = np.ndarray
Action = int
BatchTensors = tt.Tuple[
    torch.ByteTensor,
    torch.LongTensor,
    torch.Tensor,
    torch.BoolTensor,
    torch.ByteTensor
]

@dataclass
class Experience:
    state: State
    action: Action
    reward: float
    done_trunc: bool
    new_state: State

class ExperienceBuffer:
    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)
    
    def __len__(self):
        return len(self.buffer)

    def append(self, experience: Experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> tt.List[Experience]:
        indices = np.random.choice(len(self), batch_size, replace=False)

        return [self.buffer[idx] for idx in indices]