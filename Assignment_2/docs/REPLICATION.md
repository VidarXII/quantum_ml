# How to Replicate Results

This guide details the steps to set up the environment, train the Neural Quantum State Reconstruction (NQSR) model, and replicate the performance metrics.

## 1. Prerequisites

### Python Version
**Strict Requirement:** Python **3.11**.
* Later versions (e.g., 3.12+) may have compatibility issues with specific quantum/ML libraries used in this project.
* Earlier versions may lack necessary PyTorch features.

### Directory Structure
Ensure your project is organized as follows:

```text
quantum-tomography/
├── src/
│   ├── data/                  # Stores generated training datasets
│   ├── data_gen.py            # Physics engine & data loader
│   ├── model.py               # Transformer & Physics Head architecture
│   ├── utils.py               # Fidelity & Trace Distance metrics
│   └── train.py               # Main training & evaluation script
├── outputs/
│   └── best_model_opt.pt      # Saved model weights (generated after training)
├── docs/
│   ├── model_working.md       # Architectural logic explanation
│   └── REPLICATION.md         # This file
└── requirements.txt 
```

## 2. Environment Setup
It is highly recommended to use a virtual environment to avoid dependency conflicts.
```
Linux / MacOS
Bash
```

### 1. Create a virtual environment with Python 3.11
``` 
python3.11 -m venv venv
```
### 2. Activate the environment
``` 
source venv/bin/activate
```
# 3. Install dependencies
```
pip install -r requirements.txt
Windows (PowerShell)
PowerShell
```
### 1. Create a virtual environment
```
python -m venv venv
```
### 2. Activate the environment
```
.\venv\Scripts\Activate.ps1
```
### 3. Install dependencies

```
pip install -r requirements.txt
requirements.txt
```
Create this file in your root directory if it does not exist:
``` Plaintext
numpy>=1.24.0
pandas>=2.0.0
torch>=2.0.0
scikit-learn>=1.2.0
```
## 3. Running the Code
### A. Data Generation & Training
The entry point for the project is src/train.py. This script handles:

Data Generation: Automatically generates synthetic quantum measurements if src/data/train_dataset_opt.csv is missing.

Training: Trains the Transformer model.

Evaluation: reports Mean Fidelity and Trace Distance.

To run the training loop:


### Run from the root directory
```
python src/train.py
```
Note: The first run will take a few seconds longer as it generates the dataset.

B. Accessing Trained Weights
Upon completion, the model with the highest validation fidelity will be saved.
```
Location: outputs/

Filename: best_model_opt.pt (or similar .pt file)
```
To load these weights for inference in your own script:
```
Python

import torch
from src.model import QubitTransformer
```
### Initialize model structure
```
model = QubitTransformer(embed_dim=64)
```
### Load weights
```
weights_path = "outputs/best_model_opt.pt"
model.load_state_dict(torch.load(weights_path))
model.eval()
```
### Expected Performance
When replicating this work on a standard GPU (e.g., T4, RTX 3060) or modern CPU, you should expect:

Training Time: ~2-5 minutes for 100 epochs.

Mean Fidelity: > 99.0%

Trace Distance: < 0.02

Inference Latency: < 200 μs per state.

###  Troubleshooting
#### "ModuleNotFoundError: No module named 'src'": Ensure you are running the command from the root folder (quantum-tomography/), not inside src/.

Correct: python src/train.py

Incorrect: cd src && python train.py

CUDA/GPU Errors: If you do not have a GPU, the code will automatically default to CPU. To force CPU usage, modify CONFIG['device'] in src/train.py.