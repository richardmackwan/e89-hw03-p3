"""Step 1: imports and device selection (cuda -> mps -> cpu)."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

n_epochs = 20

print(f"PyTorch {torch.__version__}, using device: {device}")
