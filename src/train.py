"""
Training script using PyTorch Lightning.

Features:
    - Pretrained EfficientNet-B0 backbone (frozen by default)
    - Optional backbone unfreezing at a configurable epoch with discriminative LR
    - ModelCheckpoint, EarlyStopping, optional W&B logging

Usage:
    python train.py --config ../config.yaml
"""

import argparse
import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy

from dataset import ESC50Dataset
from model import AudioClassifier


# ---------------------------------------------------------------------------
# LightningModule
# ---------------------------------------------------------------------------
class AudioEventDetector(pl.LightningModule):
    """Lightning wrapper for AudioClassifier.

    Handles training / validation / test steps, metrics, optimizer,
    scheduler, and optional backbone unfreezing mid-training.
    """

    def __init__(self, model_cfg: dict, train_cfg: dict):
        super().__init__()
        self.save_hyperparameters()

        self.model = AudioClassifier(
            n_classes=model_cfg["n_classes"],
            freeze_backbone=model_cfg.get("freeze_backbone", True),
        )
        self.criterion = nn.CrossEntropyLoss()
        self.train_cfg = train_cfg
        self.model_cfg = model_cfg

        n_classes = model_cfg["n_classes"]
        self.train_acc = MulticlassAccuracy(num_classes=n_classes, average="micro")
        self.val_acc = MulticlassAccuracy(num_classes=n_classes, average="micro")
        self.test_acc = MulticlassAccuracy(num_classes=n_classes, average="micro")

    # -- forward -----------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    # -- steps -------------------------------------------------------------

    def _shared_step(self, batch):
        mel, labels = batch
        logits = self(mel)
        loss = self.criterion(logits, labels)
        preds = logits.argmax(dim=1)
        return loss, preds, labels

    def training_step(self, batch, batch_idx):
        loss, preds, labels = self._shared_step(batch)
        self.train_acc(preds, labels)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self._shared_step(batch)
        self.val_acc(preds, labels)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, preds, labels = self._shared_step(batch)
        self.test_acc(preds, labels)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True)

    # -- backbone unfreeze -------------------------------------------------

    def on_train_epoch_start(self):
        unfreeze_epoch = self.model_cfg.get("unfreeze_epoch")
        if unfreeze_epoch is None:
            return
        if self.current_epoch == unfreeze_epoch and self.model.backbone_frozen:
            self.model.unfreeze_backbone()

            # Add backbone params to optimizer with lower LR
            factor = self.model_cfg.get("backbone_lr_factor", 0.1)
            backbone_lr = self.train_cfg["lr"] * factor

            optimizer = self.trainer.optimizers[0]
            optimizer.add_param_group({
                "params": list(self.model.backbone.parameters()),
                "lr": backbone_lr,
            })

            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(
                f"\n>>> Backbone unfrozen at epoch {self.current_epoch}  |  "
                f"backbone LR={backbone_lr:.1e}  |  "
                f"trainable params={trainable / 1e6:.2f}M\n"
            )

    # -- optimizer ---------------------------------------------------------

    def configure_optimizers(self):
        # Only include parameters that currently require gradients
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.train_cfg["lr"],
            weight_decay=self.train_cfg["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.train_cfg["epochs"],
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


# ---------------------------------------------------------------------------
# LightningDataModule
# ---------------------------------------------------------------------------
class ESC50DataModule(pl.LightningDataModule):
    """Data module wrapping ESC-50 dataset splits."""

    def __init__(self, data_cfg: dict, batch_size: int, num_workers: int = 4):
        super().__init__()
        self.data_cfg = data_cfg
        self.batch_size = batch_size
        self.num_workers = num_workers

    def _make_dataset(self, folds: list, augment: bool) -> ESC50Dataset:
        return ESC50Dataset(
            root=self.data_cfg["root"],
            folds=folds,
            sample_rate=self.data_cfg["sample_rate"],
            duration=self.data_cfg["duration"],
            n_mels=self.data_cfg["n_mels"],
            augment=augment,
        )

    def setup(self, stage=None):
        if stage in ("fit", None):
            self.train_ds = self._make_dataset(self.data_cfg["train_folds"], augment=True)
            self.val_ds = self._make_dataset(self.data_cfg["val_folds"], augment=False)
        if stage in ("test", None):
            self.test_ds = self._make_dataset(self.data_cfg["test_folds"], augment=False)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train audio event detector (Lightning)")
    parser.add_argument("--config", type=str, default="./config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ---- Module & Data ----
    module = AudioEventDetector(
        model_cfg=cfg["model"],
        train_cfg=cfg["train"],
    )
    datamodule = ESC50DataModule(
        data_cfg=cfg["data"],
        batch_size=cfg["train"]["batch_size"],
    )

    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"Total params    : {total / 1e6:.2f}M")
    print(f"Trainable params: {trainable / 1e6:.2f}M")
    if cfg["model"].get("freeze_backbone", True):
        print(f"Backbone: FROZEN (EfficientNet-B0 ImageNet)")
        ue = cfg["model"].get("unfreeze_epoch")
        if ue is not None:
            print(f"  → will unfreeze at epoch {ue}")

    # ---- Logger ----
    wandb_cfg = cfg.get("wandb", {})
    logger = None
    if wandb_cfg.get("enabled", False):
        logger = WandbLogger(project=wandb_cfg["project"], config=cfg)
        print("Weights & Biases logging enabled.")

    # ---- Callbacks ----
    save_dir = cfg["train"].get("save_dir", "./checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        dirpath=save_dir,
        filename="best-{epoch:02d}-{val/acc:.4f}",
        monitor="val/acc",
        mode="max",
        save_top_k=1,
        verbose=True,
    )
    early_stop_cb = EarlyStopping(
        monitor="val/acc",
        mode="max",
        patience=15,
        verbose=True,
    )

    # ---- Trainer ----
    trainer = pl.Trainer(
        max_epochs=cfg["train"]["epochs"],
        accelerator="auto",
        devices="auto",
        logger=logger,
        callbacks=[checkpoint_cb, early_stop_cb],
        log_every_n_steps=5,
    )

    # ---- Train ----
    trainer.fit(module, datamodule=datamodule)

    # ---- Test with best checkpoint ----
    print(f"\nBest checkpoint: {checkpoint_cb.best_model_path}")
    trainer.test(module, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()