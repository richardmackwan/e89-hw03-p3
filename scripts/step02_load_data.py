"""Step 2: load Fashion MNIST with torchvision.transforms.v2, split 55,000/5,000
(seed 42) and build DataLoaders with batch size 32."""
# [script-only]
import torch
from torch.utils.data import DataLoader
# [/script-only]
import torchvision
import torchvision.transforms.v2 as T

toTensor = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])

train_and_valid_data = torchvision.datasets.FashionMNIST(
    root="datasets", train=True, download=True, transform=toTensor)
test_data = torchvision.datasets.FashionMNIST(
    root="datasets", train=False, download=True, transform=toTensor)

torch.manual_seed(42)
train_data, valid_data = torch.utils.data.random_split(
    train_and_valid_data, [55_000, 5_000])

torch.manual_seed(42)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=32)
test_loader = DataLoader(test_data, batch_size=32)

# Each entry is an (image, target) tuple; images are [channels, rows, columns]
X_sample, y_sample = train_data[0]
print(f"train: {len(train_data)}, valid: {len(valid_data)}, "
      f"test: {len(test_data)}")
print(f"sample shape: {tuple(X_sample.shape)}, dtype: {X_sample.dtype}, "
      f"class: {train_and_valid_data.classes[y_sample]}")
