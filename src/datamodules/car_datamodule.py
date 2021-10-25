from typing import Optional, Tuple

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
from torch._C import TracingState
from torch.utils.data import DataLoader, Dataset, random_split


class CarDataset(Dataset):
    """Initializes a dataset from the output of
    src.callbacks.eval_callbacks.SaveTrajectoryToHDF5Callback"""

    def __init__(self, data_filepath, transform=None, target_transform=None):
        # By assumption, there is only one simulation
        if transform is None:
            transform = self._torchify
        if target_transform is None:
            target_transform = self._torchify
        self.transform = transform
        self.target_transform = target_transform
        self.simulation_grp = h5py.File(data_filepath, "r")["simulation"]

    def __len__(self):
        return self.simulation_grp.attrs["num_simulation_steps"]

    def _idx_to_episode_goal_step(self, idx):
        """Convert a single integer idx in [0, len(self)) to the
        corresponding episode number, goal, and the step within that goal"""
        # Calculate the correct episode_number
        next_episode_number = 0
        episode_number = None
        next_cumulative_episode_steps = 0
        cumulative_episode_steps = None
        while next_cumulative_episode_steps <= idx:
            cumulative_episode_steps = next_cumulative_episode_steps
            episode_number = next_episode_number
            next_cumulative_episode_steps += self._get_episode_grp(episode_number).attrs[
                "num_episode_steps"
            ]
            next_episode_number += 1
        episode_step = idx - cumulative_episode_steps

        # Calculate the correct goal_number
        next_goal_number = 0
        goal_number = None
        next_cumulative_goal_steps = 0
        cumulative_goal_steps = None
        while next_cumulative_goal_steps <= episode_step:
            cumulative_goal_steps = next_cumulative_goal_steps
            goal_number = next_goal_number
            next_cumulative_goal_steps += self._get_goal_grp(episode_number, goal_number).attrs[
                "num_goal_steps"
            ]
            next_goal_number += 1
        goal_step = episode_step - cumulative_goal_steps
        return episode_number, goal_number, goal_step

    def _get_episode_grp(self, episode_number):
        """ Get the subgroup of data corresponding to given episode number """
        return self.simulation_grp[f"episode_{episode_number}"]

    def _get_goal_grp(self, episode_number, goal_number):
        """ Get the subgroup of data corresponding to given goal in a given episode """
        episode_grp = self._get_episode_grp(episode_number)
        return episode_grp[f"goal_{goal_number}"]

    @staticmethod
    def _torchify(x: np.ndarray):
        return torch.from_numpy(x.astype(np.float32))

    def __getitem__(self, idx):
        episode_number, goal_number, goal_step = self._idx_to_episode_goal_step(idx)
        goal_group = self._get_goal_grp(episode_number, goal_number)

        current_state = goal_group["state_trajectory"][goal_step]
        action = goal_group["input_trajectory"][goal_step]
        goal_state = goal_group.attrs["goal_state"]

        trunc_state = current_state[3:]
        relative_goal = current_state[:3] - goal_state[:3]
        inp = np.concatenate([relative_goal, trunc_state])
        oup = action
        return self.transform(inp), self.target_transform(oup)


class CarDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_filepath: str,
        train_val_test_split: Tuple[float, float, float] = (0.7, 0.2, 0.1),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        data_seed: int = 0,
        transform=None,
        target_transform=None,
    ):
        super().__init__()
        self.data_filepath = data_filepath
        self.train_val_test_split = train_val_test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.data_seed = data_seed
        self.transform = transform
        self.target_transform = target_transform

    @property
    def train_fraction(self):
        return self.train_val_test_split[0]

    @property
    def val_fraction(self):
        return self.train_val_test_split[1]

    @property
    def test_fraction(self):
        return self.train_val_test_split[2]

    def prepare_data(self):
        self.dataset = CarDataset(self.data_filepath, self.transform, self.target_transform)

    def setup(self, stage: Optional[str] = None):
        train_len = int(self.train_fraction * len(self.dataset))
        val_len = int(self.val_fraction * len(self.dataset))
        test_len = len(self.dataset) - train_len - val_len
        # Split data randomly, but set random seed for reproducibility
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset,
            lengths=(train_len, val_len, test_len),
            generator=torch.Generator().manual_seed(self.data_seed),
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
