"""
Evaluation script with per-class classification report.
Compatible with PyTorch Lightning checkpoints.

Usage:
    python evaluate.py --config ../config.yaml --checkpoint ../checkpoints/best-epoch=XX-val/acc=X.XXXX.ckpt
"""

import argparse

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader

from dataset import ESC50Dataset
from train import AudioEventDetector


def main():
    parser = argparse.ArgumentParser(description="Evaluate audio event detector")
    parser.add_argument("--config", type=str, default="../config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Data ----
    data_cfg = cfg["data"]
    test_ds = ESC50Dataset(
        root=data_cfg["root"],
        folds=data_cfg["test_folds"],
        sample_rate=data_cfg["sample_rate"],
        duration=data_cfg["duration"],
        n_mels=data_cfg["n_mels"],
        augment=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=4,
    )
    print(f"Test samples: {len(test_ds)}")

    # ---- Load from Lightning checkpoint ----
    module = AudioEventDetector.load_from_checkpoint(
        args.checkpoint,
        model_cfg=cfg["model"],
        train_cfg=cfg["train"],
    )
    module.to(device)
    module.eval()

    # ---- Inference ----
    all_preds, all_labels = [], []
    with torch.no_grad():
        for mel, labels in test_loader:
            mel = mel.to(device)
            logits = module(mel)
            preds = logits.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # ---- Report ----
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, digits=3))


if __name__ == "__main__":
    main()