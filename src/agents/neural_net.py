from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from src.agents.base import BaseAgent


class InferenceWrapper(BaseAgent, ABC):
    """Inference wrapper for a trained policy."""

    @abstractmethod
    def get_policy(self):
        pass

    @torch.no_grad()
    def get_action(self, state: np.ndarray) -> np.ndarray:
        relative_goal = state[:3] - self.goal_state[:3]
        trunc_state = state[3:]
        inputs_np = np.concatenate([trunc_state, relative_goal], axis=0)
        inputs = torch.from_numpy(inputs_np).float().view(1, -1)

        policy = self.get_policy()
        nn_action = policy(inputs)
        nn_action = nn_action.detach().clone().numpy()[0]
        return nn_action
