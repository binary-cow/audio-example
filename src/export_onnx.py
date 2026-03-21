"""
Export trained model to ONNX format for edge/cross-platform deployment.
Compatible with PyTorch Lightning checkpoints.

Usage:
    python export_onnx.py --config ../config.yaml \
        --checkpoint ../checkpoints/best-epoch=XX-val/acc=X.XXXX.ckpt \
        --output ../model.onnx
"""

import argparse
import os

import torch
import yaml

from train import AudioEventDetector


def main():
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument("--config", type=str, default="../config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="../model.onnx")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ---- Load from Lightning checkpoint ----
    module = AudioEventDetector.load_from_checkpoint(
        args.checkpoint,
        model_cfg=cfg["model"],
        train_cfg=cfg["train"],
    )
    module.eval()

    # Extract the underlying nn.Module for cleaner ONNX graph
    model = module.model

    # ---- Dummy input ----
    sr = cfg["data"]["sample_rate"]
    duration = cfg["data"]["duration"]
    hop_length = 512
    n_frames = int(sr * duration / hop_length) + 1
    n_mels = cfg["data"]["n_mels"]

    dummy_input = torch.randn(1, 1, n_mels, n_frames)

    # ---- Export ----
    torch.onnx.export(
        model,
        dummy_input,
        args.output,
        input_names=["mel_spectrogram"],
        output_names=["logits"],
        dynamic_axes={
            "mel_spectrogram": {0: "batch_size", 3: "time_frames"},
            "logits": {0: "batch_size"},
        },
        opset_version=13,
    )

    # ---- Summary ----
    file_size_mb = os.path.getsize(args.output) / (1024 * 1024)
    param_count = sum(p.numel() for p in model.parameters())

    print(f"Exported: {args.output}")
    print(f"  ONNX file size : {file_size_mb:.2f} MB")
    print(f"  Parameters     : {param_count / 1e6:.2f}M")
    print(f"  Input shape    : (batch, 1, {n_mels}, {n_frames})")
    print(f"  Output shape   : (batch, {cfg['model']['n_classes']})")


if __name__ == "__main__":
    main()