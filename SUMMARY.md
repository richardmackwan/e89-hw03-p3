# Summary — Assignment 03, Problem 3 (Richard Mackwan)

Reproduces the section **"Building an Image Classifier with PyTorch"** of
`reference/10_neural_nets_with_pytorch.ipynb`, plus a training/validation
accuracy plot.

## Scripts (run from the repository root, in this order)

| Step | Script | What it does |
|---|---|---|
| 1 | `scripts/step01_setup_device.py` | Imports; picks device `cuda` → `mps` → `cpu`; `n_epochs = 20` |
| 2 | `scripts/step02_load_data.py` | Fashion MNIST via `torchvision.transforms.v2` (`ToImage` + `ToDtype(float32, scale=True)`), 55,000/5,000 train/valid split with seed 42, DataLoaders with batch size 32 |
| 3 | `scripts/step03_training_helpers.py` | `evaluate_tm` and `train2` helpers from the notebook |
| 4 | `scripts/step04_build_model.py` | `ImageClassifier` MLP 784→300→100→10 with ReLU; `CrossEntropyLoss` |
| 5 | `scripts/step05_train_model.py` | SGD (lr=0.1) + `torchmetrics.Accuracy`, 20 epochs; saves `fashion_mnist_mlp.pt` and `history.json` |
| 6 | `scripts/step06_predict.py` | Predicts 3 validation images; softmax probabilities; top-4 probabilities and class indices |
| 7 | `scripts/step07_count_parameters.py` | Counts parameters |
| 8 | `scripts/step08_plot_accuracy.py` | Plots training and validation accuracy per epoch → `training_accuracy.png` |

`python scripts/step05_train_model.py` trains (steps 1–4 are imported), and the
later steps load the saved weights/history, so each script can be run on its own
after step 5. Code only needed for standalone running (cross-script imports,
reloading saved files) is marked with `# [script-only]` blocks.

`scripts/build_notebook.py` assembles **`e89_Mackwan_Richard_HW03_Prob3.ipynb`**:
a title markdown cell (name + "Assignment 03, Problem 3"), then for each script a
markdown cell naming the script and describing it, followed by its code with the
script-only blocks removed. The notebook is self-contained and committed with
outputs after a full top-to-bottom run (0 errors, ~4 min on CPU).

## Results (CPU, seed 42)

- Final epoch: train loss 0.1876, train accuracy **92.86%**, validation accuracy
  **87.88%** (best validation 88.86% at epoch 16).
- Predictions on the first 3 validation images: Sneaker, Coat, Pullover — all
  correct. Top-4 probabilities: `[0.859, 0.141, 0, 0]`, `[0.993, 0.007, 0, 0]`,
  `[0.593, 0.212, 0.191, 0.004]`.
- Parameter count: **266,610** (784·300+300 + 300·100+100 + 100·10+10), matching
  the reference notebook.
- `training_accuracy.png`: training accuracy keeps rising while validation
  plateaus around 87–89% after ~epoch 10, i.e. mild overfitting.

## Notes

- Differences from the reference: outputs are `print`ed (so scripts show them),
  and probabilities are always moved to CPU before display (the reference did
  this only for `mps`).
- Results were produced on CPU; on CUDA/MPS numbers may differ slightly.
- Environment: Python 3.11, torch 2.14, torchvision 0.29, torchmetrics 1.9.
  The dataset downloads to `datasets/` (git-ignored, as are the weights and
  `history.json`).
