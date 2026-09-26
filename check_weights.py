"""Run once before deploying: python check_weights.py

Confirms the .pth files from Colab load into model.py with strict key
matching and that a single image runs through each network.
"""
from pathlib import Path

import torch

from model import CNN_model, MLP, load_model

WEIGHTS = Path(__file__).parent / "weights"

for name, cls, file in [
    ("CNN", CNN_model, "best_mnist_cnn_model.pth"),
    ("MLP", MLP, "best_mnist_model.pth"),
]:
    path = WEIGHTS / file
    if not path.exists():
        print(f"{name}: {path} not found (skipped)")
        continue
    model = load_model(cls, path)
    with torch.no_grad():
        out = model(torch.zeros(1, 1, 28, 28))
    print(f"{name}: loaded OK, output shape {tuple(out.shape)}")
