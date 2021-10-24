from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F


class InferenceWrapper(torch.nn.Module):
    """Inference wrapper for a trained policy."""

    def __init__(self, policy: torch.nn.Module):
        super(InferenceWrapper, self).__init__()
        self.policy = policy

    def reset(self, env):
        self.goal_state = env.goal_state
        self.obstacle_centers = env.obstacle_centers
        self.obstacle_radii = env.obstacle_radii

    def update_policy(self, policy):
        self.policy.load_state_dict(policy.state_dict())

    def get_action(self, state: np.ndarray) -> np.ndarray:
        relative_goal = state[:3] - self.goal_state[:3]
        trunc_state = state[3:]
        inputs_np = np.concatenate([trunc_state, relative_goal], axis=0)
        inputs = torch.from_numpy(inputs_np).float().view(1, -1)

        nn_action = self.policy(inputs)
        nn_action = nn_action.detach().clone().numpy()[0]
        return nn_action
