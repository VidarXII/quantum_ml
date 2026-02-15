# Machine Learning for Scalable Quantum State Tomography

## Problem Statement
Quantum State Tomography (QST) is essential for verifying quantum hardware, but it suffers from an exponential bottleneck: reconstructing an $N$-qubit state requires resources that scale as $O(2^N)$. This project explores computational models to efficiently reconstruct density matrices from measurement data while strictly enforcing physical constraints.

## Approach & Methodology
* **Single-Qubit Baseline:** Simulated Pauli Projective measurements and performed linear inversion.
* **Track 1 Transformer Model:** Trained a sequence model to map Pauli measurement probabilities to a physically valid density matrix.
* **Scalable N-Qubit Surrogate:** Implemented a Variational Quantum Tomography ansatz using PyTorch to benchmark scalability up to 12 qubits.

## Workflow & Repository Structure
* `/data`: Contains synthetic measurement outcomes and target states.
* `/models`: Saved checkpoints for the Transformer and Cholesky surrogate models.
* `/notebooks`: Executed Jupyter notebooks for data generation, training, and benchmarking.
* `/src`: Core deep learning architecture and training loop scripts.
* `/results`: Scalability benchmarks and fidelity plots.
