
# Model Checkpoints

This directory contains serialized checkpoints (`.pkl`) and benchmark data (`.csv`).

## How to Load a Model
To restore a saved model, use the static `load` method defined in the notebook:

```python
import torch
# Ensure the QuantumModel class is defined in your scope
model = QuantumModel.load("models/test_checkpoint.pkl")

# Verify configuration
print(f"Loaded {model.n_qubits}-qubit model.") 
