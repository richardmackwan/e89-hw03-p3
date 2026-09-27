"""Step 8: plot training and validation accuracy per epoch and save it to
training_accuracy.png."""
# [script-only]
import json
import matplotlib
matplotlib.use("Agg")  # headless when run as a script
with open("history.json") as f:
    history = json.load(f)
# [/script-only]
import matplotlib.pyplot as plt

epochs = range(1, len(history["train_metrics"]) + 1)
plt.figure(figsize=(8, 5))
plt.plot(epochs, history["train_metrics"], ".--", label="Training accuracy")
plt.plot(epochs, history["valid_metrics"], ".-", label="Validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Fashion MNIST MLP: accuracy per epoch")
plt.xticks(list(epochs))
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("training_accuracy.png", dpi=150)
print("Saved training_accuracy.png")
plt.show()
