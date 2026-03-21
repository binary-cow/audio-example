# Efficient Audio Event Detection Pipeline

An end-to-end ML research pipeline for audio event detection, emphasizing **efficient model architectures** suitable for real-time and edge deployment, **rapid experimentation** with structured experiment tracking, and **deployment-ready model export**.

## Motivation

Audio event detection is a core component of interactive systems that sense and respond to environmental sounds. This project demonstrates a complete ML research workflow — from data preprocessing through model training to deployment-ready export — using lightweight architectures optimized for resource-constrained environments.

## Key Features

- **Audio preprocessing** with mel spectrogram extraction and data augmentation (SpecAugment, Gaussian noise injection, gain perturbation)
- **Efficient model architecture** using depthwise separable convolutions (MobileNet-style), designed for low-latency inference
- **Experiment tracking** with optional [Weights & Biases](https://wandb.ai) integration
- **Systematic evaluation** with per-class classification reports via scikit-learn
- **ONNX export** for cross-platform deployment and efficient edge inference

## Dataset

This pipeline uses [ESC-50](https://github.com/karolpiczak/ESC-50) (Environmental Sound Classification), a labeled collection of 2,000 short environmental audio recordings across 50 classes (e.g., dog bark, rain, clapping, footsteps). The dataset provides 5 pre-defined folds for cross-validation.

## Project Structure


├── config.yaml # Experiment configuration (data, model, training)
├── requirements.txt # Python dependencies
├── scripts/
│ └── download_data.sh # ESC-50 dataset download script
└── src/
├── dataset.py # Data loading, preprocessing, augmentation
├── model.py # Efficient audio classifier architecture
├── train.py # Training loop with W&B logging
├── evaluate.py # Evaluation with classification report
└── export_onnx.py # ONNX model export for deployment

basic

## Quick Start

1. Install dependencies
```bash
pip install -r requirements.txt
```

2. Download dataset
```bash
bash scripts/download_data.sh
```
3. Train
```bash
cd src
python train.py --config ../config.yaml
```
To enable Weights & Biases logging, set wandb.enabled: true in config.yaml and run wandb login first.

4. Evaluate
```bash
python evaluate.py --config ../config.yaml --checkpoint ../checkpoints/best_model.pt
```
5. Export to ONNX
```bash
python export_onnx.py --config ../config.yaml --checkpoint ../checkpoints/best_model.pt --output ../model.onnx
```


## To-Do / Roadmap

- [X] ~~Revise model with pretrained encoder~~
- [X] ~~Add WandB support~~

### Data & Augmentation
- [ ] Apply SpecAugment (time/frequency masking) 
- [ ] Introduce further datasets (UrbanSound8K, AudioSet)

### Model & Training
- [ ] Backbone change (AST, CLAP-based)
- [ ] Knowledge Distillation 


### Evaluation & Analysis
- [ ] Precision / Recall / F1 by classes
- [ ] Confusion Matrices



## Acknowledgments
- ESC-50 dataset by Karol J. Piczak
- Built in collaboration with Claude (Anthropic)

