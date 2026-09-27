"""Step 6: predict the first 3 validation images, then show the softmax
probabilities and the top-4 class probabilities."""
# [script-only]
import torch
from step01_setup_device import device
from step02_load_data import train_and_valid_data, valid_loader
from step04_build_model import model
model.load_state_dict(torch.load("fashion_mnist_mlp.pt", map_location=device))
# [/script-only]
import torch.nn.functional as F

model.eval()
X_new, y_new = next(iter(valid_loader))
X_new = X_new[:3].to(device)
with torch.no_grad():
    y_pred_logits = model(X_new)
y_pred = y_pred_logits.argmax(dim=1)  # index of the largest logit
print("Predicted indices:", y_pred.tolist())
print("Predicted classes:",
      [train_and_valid_data.classes[index] for index in y_pred])
print("True indices:     ", y_new[:3].tolist())
print("True classes:     ",
      [train_and_valid_data.classes[index] for index in y_new[:3]])

y_proba = F.softmax(y_pred_logits, dim=1).cpu()
print("Class probabilities:\n", y_proba.round(decimals=3))

y_top4_values, y_top4_indices = torch.topk(y_pred_logits, k=4, dim=1)
y_top4_probas = F.softmax(y_top4_values, dim=1).cpu()
print("Top-4 probabilities:\n", y_top4_probas.round(decimals=3))
print("Top-4 class indices:\n", y_top4_indices.cpu())
