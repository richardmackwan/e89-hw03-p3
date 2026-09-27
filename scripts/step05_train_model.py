"""Step 5: train for 20 epochs with SGD (lr=0.1) and torchmetrics Accuracy,
then save the weights and the per-epoch history."""
# [script-only]
import torch
from step01_setup_device import device, n_epochs
from step02_load_data import train_loader, valid_loader
from step03_training_helpers import train2
from step04_build_model import model, xentropy
# [/script-only]
import json
import torchmetrics

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)
history = train2(model, optimizer, xentropy, accuracy, train_loader,
                 valid_loader, n_epochs)

torch.save(model.state_dict(), "fashion_mnist_mlp.pt")
with open("history.json", "w") as f:
    json.dump(history, f, indent=2)
print("Saved fashion_mnist_mlp.pt and history.json")
