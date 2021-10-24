from typing import Optional, Tuple

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split


class CarDataset(Dataset):
    def __init__(self, data_filepath, transform=None, target_transform=None):
        self.dataset = h5py.File(data_filepath, "r")["simulation_0"]

    def __len__(self):
        return self.dataset.attrs["num_total_steps"]

    def __getitem__(self, idx):
        goal = 0
        for current_goal, end_marker in enumerate(self.dataset.attrs["end_markers"]):
            if end_marker >= idx:
                goal = current_goal
                break

        start_marker = self.dataset.attrs["end_markers"][goal - 1] if goal > 0 else 0
        traj_time = idx - start_marker - 1 if start_marker > 0 else idx

        sub_data = self.dataset[f"goal_{goal}"]
        current_state = torch.from_numpy(
            sub_data["state_trajectory"][traj_time, :].astype(np.float32)
        )
        action = torch.from_numpy(sub_data["input_trajectory"][traj_time, :].astype(np.float32))
        goal_state = torch.from_numpy(sub_data.attrs["goal_state"].astype(np.float32))

        trunc_state = current_state[3:]
        relative_goal = current_state[:3] - goal_state[:3]
        inp = torch.cat([relative_goal, trunc_state])
        oup = action

        return inp, oup


class CarDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_filepath: str,
        train_val_test_split: Tuple[float, float, float] = (0.7, 0.2, 0.1),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        data_seed: int = 0,
    ):
        super().__init__()
        self.data_filepath = data_filepath
        self.train_val_test_split = train_val_test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.data_seed = data_seed

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
        self.dataset = CarDataset(self.data_filepath)

    def setup(self):
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
