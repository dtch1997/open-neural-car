from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics.regression.mean_absolute_error import MeanAbsoluteError

from src.agents.neural_net import InferenceWrapper


class BehaviouralCloning(LightningModule, InferenceWrapper):
    """A wrapper to train a policy via behavioural cloning"""

    def __init__(
        self,
        lr: float = 0.001,
        weight_decay: float = 0.0005,
    ):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()
        # loss function
        self.criterion = torch.nn.SmoothL1Loss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()
        self.test_mae = MeanAbsoluteError()

    def get_policy(self):
        return self.policy

    def update_policy(self, policy: torch.nn.Module):
        self.policy = policy

    def forward(self, x: torch.Tensor):
        return self.policy(x)

    def step(self, batch: Any):
        observations, actions_true = batch
        actions_pred = self.forward(observations)
        loss = self.criterion(actions_true, actions_pred)
        return loss, actions_pred, actions_true

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log train metrics
        mae = self.train_mae(preds, targets)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/mae", mae, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in training_epoch_end() below
        # remember to always return loss from training_step, or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log val metrics
        mae = self.val_mae(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/mae", mae, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log test metrics
        mae = self.test_mae(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/mae", mae, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
