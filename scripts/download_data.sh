#!/bin/bash
set -e

echo "============================================"
echo " Downloading ESC-50 dataset"
echo "============================================"

mkdir -p data
cd data

if [ -d "ESC-50" ]; then
    echo "ESC-50 directory already exists. Skipping download."
else
    git clone https://github.com/karolpiczak/ESC-50.git
    echo "Done! Dataset saved to data/ESC-50/"
fi

echo ""
echo "Dataset structure:"
echo "  data/ESC-50/audio/    - 2000 audio files (.wav)"
echo "  data/ESC-50/meta/     - metadata (esc50.csv)"
echo "============================================"