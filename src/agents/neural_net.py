from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from src.agents.base import BaseAgent


class InferenceWrapper(BaseAgent):
    """Inference wrapper for a trained policy."""

    def __init__(self, algorithm: LightningModule, checkpoint_path: str = None):
        if checkpoint_path is not None:
            algorithm = algorithm.load_from_checkpoint(checkpoint_path)
        self.policy = algorithm.policy

    def reset(self, env):
        self._goal_state = env.goal_state

    @torch.no_grad()
    def get_action(self, state: np.ndarray) -> np.ndarray:
        relative_goal = state[:3] - self._goal_state[:3]
        trunc_state = state[3:]
        inputs_np = np.concatenate([trunc_state, relative_goal], axis=0)
        inputs = torch.from_numpy(inputs_np).float().view(1, -1)

        nn_action = self.policy(inputs)
        nn_action = nn_action.detach().clone().numpy()[0]
        return nn_action
