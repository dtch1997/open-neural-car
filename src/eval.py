from pathlib import Path
from typing import List

import hydra
import numpy as np
from omegaconf import DictConfig
from pytorch_lightning import seed_everything

from src.utils import utils

SAVE_DIR = Path("data")
log = utils.get_logger(__name__)


def _generate_obstacles():
    """ Generate 5 random obstacles in XY coordinates """
    relative_obstacle_centers = np.random.uniform(low=-10, high=10, size=(5, 2))
    obstacle_radii = np.ones(shape=(5, 1), dtype=np.float32)
    return relative_obstacle_centers, obstacle_radii


def _generate_goal(rearward_allowed=False):
    """ Generate a random relative goal in XY coordinates """
    if not rearward_allowed:
        r, phi, delta_th = (
            30 * np.random.uniform(),
            np.pi * np.random.uniform() - np.pi / 2,
            np.pi * np.random.uniform() - np.pi / 2,
        )
    else:
        r, phi, delta_th = (
            30 * np.random.uniform(),
            2 * np.pi * np.random.uniform() - np.pi,
            np.pi * np.random.uniform() - np.pi / 2,
        )

    relative_goal = np.array([r, phi, delta_th])
    return relative_goal


def _normalize_angle(angle):
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


def eval(config: DictConfig):
    """Contains evaluation pipeline.

    Args:
        config (DictConfig): Configuration composed by Hydra.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    agent = hydra.utils.instantiate(config.agent)
    env = hydra.utils.instantiate(config.env)

    if "callbacks" in config:
        callback = hydra.utils.instantiate(config.callbacks)

    sim_params = config.simulation

    callback.update_locals(locals())
    callback.on_simulation_start()

    for episode_number in range(sim_params.num_episodes):
        # Reset environment. Done once per episode
        env.reset(disable_view=False)
        env.update_obstacles(*_generate_obstacles())
        current_state = env.current_state
        t = -1  # Keeps track of how many steps have passed

        callback.update_locals(locals())
        callback.on_episode_start()

        for goal_number in range(sim_params.num_goals):
            env.update_goal(_generate_goal(rearward_allowed=False))
            agent.reset(env)

            callback.update_locals(locals())
            callback.on_goal_start()

            for step_number in range(sim_params.num_simulation_time_steps):
                if sim_params.render:
                    env.render()
                t += 1  # Increment the timer
                action = agent.get_action(current_state)
                next_state, reward, done, info = env.take_action(action)

                callback.update_locals(locals())
                callback.on_take_action()

                current_state = next_state
                diff = env.goal_state[:3] - current_state[:3]
                # Normalize theta to be between -pi and pi when calculating difference
                diff[2] = _normalize_angle(diff[2])

                if np.linalg.norm(diff).item() < sim_params.dist_threshold:
                    break

            callback.update_locals(locals())
            callback.on_goal_end()

        callback.update_locals(locals())
        callback.on_episode_end()

    callback.update_locals(locals())
    callback.on_simulation_end()
