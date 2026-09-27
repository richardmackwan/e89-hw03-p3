"""Step 7: count the model's trainable parameters."""
# [script-only]
from step04_build_model import model
# [/script-only]

n_params = sum([param.numel() for param in model.parameters()])
print(f"Total parameters: {n_params:,}")  # 784*300+300 + 300*100+100 + 100*10+10
