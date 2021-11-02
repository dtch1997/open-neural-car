"""Callbacks used in src/eval.py"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from src.utils.statistics import Average


class BaseCallback(ABC):
    def __init__(self, verbose: int = 0):
        super(BaseCallback, self).__init__()
        self.verbose = verbose
        self.locals: Dict[str, Any] = {}

    method_names = (
        "on_simulation_start",
        "on_simulation_end",
        "on_episode_start",
        "on_episode_end",
        "on_goal_start",
        "on_goal_end",
        "on_take_action",
        "update_locals",
        "update_child_locals",
    )

    def on_simulation_start(self):
        pass

    def on_simulation_end(self):
        pass

    def on_episode_start(self):
        pass

    def on_episode_end(self):
        pass

    def on_goal_start(self):
        pass

    def on_goal_end(self):
        pass

    def on_take_action(self):
        pass

    def update_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables.
        :param locals_: the local variables during rollout collection
        """
        self.locals.update(locals_)
        self.update_child_locals(locals_)

    def update_child_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables on sub callbacks.
        :param locals_: the local variables during rollout collection
        """
        pass


class ListOfCallbacks(BaseCallback):
    def __init__(self, callbacks: List[BaseCallback]):
        self.callbacks = callbacks

    def __getattribute__(self, name):
        """Override callback interface methods with a list comprehension

        Otherwise, default getattr behaviour"""
        if name in BaseCallback.method_names:

            def F(*args, **kwargs):
                return_vals = []
                for callback in self.callbacks:
                    f = getattr(callback, name)
                    assert callable(f)
                    return_vals.append(f(*args, **kwargs))
                return return_vals

            return F
        else:
            return super(ListOfCallbacks, self).__getattribute__(name)


class SaveTrajectoryToHDF5Callback(BaseCallback):
    """ Saves the trajectory to a HDF5 file. """

    def __init__(self, save_dir: str, file_name: str):
        super(SaveTrajectoryToHDF5Callback, self).__init__()
        import h5py

        Path(save_dir).mkdir(exist_ok=True, parents=True)
        self.save_path = Path(save_dir) / file_name
        self.output_file = h5py.File(str(self.save_path), "w")

    def __del__(self):
        self.output_file.close()

    def on_simulation_start(self):
        self.simulation_grp = self.output_file.create_group("simulation")
        self.simulation_grp.attrs["num_simulation_steps"] = 0
        self.simulation_grp.attrs["num_episodes"] = 0

    def on_episode_start(self):
        self.simulation_grp.attrs["num_episodes"] += 1
        self.episode_grp = self.simulation_grp.create_group(
            f"episode_{self.locals['episode_number']}"
        )
        self.episode_grp.attrs["num_episode_steps"] = 0
        self.episode_grp.attrs["num_goals"] = 0

    def on_goal_start(self):
        self.episode_grp.attrs["num_goals"] += 1
        self.goal_grp = self.episode_grp.create_group(f"goal_{self.locals['goal_number']}")

        sim_params = self.locals["sim_params"]
        env = self.locals["env"]

        # By default the state/input trajectories assume max possible simulation steps
        self.state_trajectory = self.goal_grp.create_dataset(
            "state_trajectory", shape=(sim_params.num_simulation_time_steps + 1, 7), dtype="f"
        )
        self.input_trajectory = self.goal_grp.create_dataset(
            "input_trajectory", shape=(sim_params.num_simulation_time_steps, 2), dtype="f"
        )

        self.goal_grp.attrs["goal_state"] = env.goal_state
        self.goal_grp.attrs["obstacle_centers"] = env.obstacle_centers
        self.goal_grp.attrs["obstacle_radii"] = env.obstacle_radii
        self.goal_grp.attrs["num_goal_steps"] = 0

    def on_take_action(self):
        self.goal_grp.attrs["num_goal_steps"] += 1
        self.episode_grp.attrs["num_episode_steps"] += 1
        self.simulation_grp.attrs["num_simulation_steps"] += 1
        self.state_trajectory[self.locals["step_number"]] = self.locals["current_state"]
        self.input_trajectory[self.locals["step_number"]] = self.locals["action"]

    def on_goal_end(self):
        self.state_trajectory[self.locals["step_number"] + 1] = self.locals["current_state"]
        del self.goal_grp
        del self.state_trajectory
        del self.input_trajectory

    def on_episode_end(self):
        del self.episode_grp

    def on_simulation_end(self):
        del self.simulation_grp


class RecordAverageAgentTimeCallback(BaseCallback):
    def __init__(self):
        super(RecordAverageAgentTimeCallback, self).__init__()
        self.average_time = Average()

    def on_episode_start(self):
        self.average_time.reset()

    def on_take_action(self):
        elapsed_time = self.locals["elapsed_time"]
        if elapsed_time is not None:
            self.average_time.update(elapsed_time)

    def on_episode_end(self):
        print(f"Episode average elapsed time: {self.average_time.value}")
