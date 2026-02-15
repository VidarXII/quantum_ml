# Quantum State Reconstruction: Model & Math

This document details the architecture and mathematical logic behind the Neural Quantum State Reconstruction (NQSR) model. Our goal is to map noisy experimental measurement statistics directly to a valid physical density matrix $\rho$.

## 1. The Core Problem

We want to reconstruct the quantum state $\rho$ of a qubit given a set of Pauli measurements.

* **Input:** Measurement probabilities $\{P_X, P_Y, P_Z\}$.
    * $P_M$: Probability of measuring outcome $|0\rangle$ in basis $M$.
* **Output:** A $2 \times 2$ Density Matrix $\rho$.
* **Constraint:** The output $\rho$ must be physically valid:
    1.  **Hermitian:** $\rho = \rho^\dagger$
    2.  **Positive Semi-Definite (PSD):** eigenvalues $\geq 0$
    3.  **Unit Trace:** $\text{Tr}(\rho) = 1$

---

## 2. Mathematical Logic

### A. Input Transformation
Raw probabilities lie in $[0, 1]$. Neural networks prefer symmetric inputs centered around zero. We map probabilities to **Expectation Values**:
$$\langle \sigma_i \rangle = 2 P_i - 1$$
This centers the input data to $[-1, 1]$, aiding gradient flow and symmetry.

### B. The Cholesky Ansatz (Physics Head)
A standard neural network outputs unbounded real numbers. Directly predicting the complex entries of $\rho$ usually results in unphysical states (e.g., negative probabilities).

We solve this by learning the **Cholesky Factor** $L$ instead of $\rho$ itself. We define:

$$\rho_{\text{un}} = L L^\dagger$$

where $L$ is a lower-triangular matrix:
$$L = \begin{pmatrix} l_{00} & 0 \\ l_{10} & l_{11} \end{pmatrix}$$

* $l_{00}, l_{11} \in \mathbb{R}^+$ (Real, positive parameters)
* $l_{10} \in \mathbb{C}$ (Complex parameter)

**Why this guarantees physical validity:**
1.  **Hermiticity:** $(L L^\dagger)^\dagger = (L^\dagger)^\dagger L^\dagger = L L^\dagger$.
2.  **PSD:** For any vector $v$, $v^\dagger (L L^\dagger) v = (L^\dagger v)^\dagger (L^\dagger v) = ||L^\dagger v||^2 \geq 0$.

**Normalization:**
Finally, to enforce the unit trace constraint (conservation of probability):
$$\rho = \frac{\rho_{\text{un}}}{\text{Tr}(\rho_{\text{un}}) + \epsilon}$$

---

## 3. Neural Architecture

We use a **Transformer-based** architecture. Transformers are excellent at capturing correlations between different measurement bases, treating the tomography data as a sequence.



### Step 1: Embedding
We treat the three measurements ($P_X, P_Y, P_Z$) as a sequence of length 3.
* **Input:** `[Batch, 3]`
* **Projection:** Each scalar is projected to a vector of size `embed_dim`.
* **Positional Encoding:** Learned vectors are added so the model can distinguish X-data from Y-data.

### Step 2: Transformer Encoder
* **Self-Attention:** This allows the model to "compare" the X-basis result with Y and Z simultaneously to determine the state's orientation on the Bloch sphere.
* **Feed-Forward:** Processes the contextualized features.

### Step 3: Decoding
* The sequence is flattened.
* An MLP projects the features down to **4 real numbers**:
    1.  $raw(\_l_{00})$
    2.  $raw(\_l_{11})$
    3.  $real(l_{10})$
    4.  $imag(l_{10})$

### Step 4: Physics Head
The 4 outputs are passed through `softplus` (for diagonals) to ensure positivity, constructed into the matrix $L$, and multiplied to form $\rho$.

---

## 4. Loss Function

We use a hybrid loss function to ensure both element-wise accuracy and high quantum fidelity.

### A. Frobenius Loss (MSE)
$$\mathcal{L}_{\text{Frob}} = || \rho_{\text{pred}} - \rho_{\text{true}} ||_F^2$$
Standard MSE on matrix elements. Good for initial rough convergence.

### B. Fidelity Loss
Fidelity $F$ measures how "close" two quantum states are. For pure states (our training data), this simplifies to the overlap:
$$F(\rho, \sigma) \approx \text{Tr}(\rho_{\text{pred}} \cdot \rho_{\text{true}})$$
$$\mathcal{L}_{\text{Fidelity}} = 1 - F$$

**Total Loss:**
$$\mathcal{L} = \alpha \mathcal{L}_{\text{Fidelity}} + (1-\alpha) \mathcal{L}_{\text{Frob}}$$
*Typically $\alpha=0.7$ gives the best results.*

---

## 5. Metrics

1.  **Fidelity:** $1.0$ means perfect reconstruction.
2.  **Trace Distance:** $D(\rho, \sigma) = \frac{1}{2}\text{Tr}|\rho - \sigma|$.
    * $0.0$ means perfect reconstruction.
    * Physically represents the maximum probability of distinguishing the two states in a single measurement.

### **4. Written Summary (`docs/report.md`)**

# Task 6: Technical Report & Reflection

## Method: Cholesky Surrogate
To reconstruct density matrices for $N$ qubits, we utilized a
 **Cholesky Ansatz** where $\rho = \frac{L L^\dagger}{\text{Tr}(L L^\dagger)}$.
* **Why:** This strictly enforces physical constraints (Positive Semi-Definite, Unit Trace) by construction.
* **Optimization:** We treated the elements of $L$ as learnable PyTorch parameters and optimized them using gradient descent to maximize Fidelity with the target state.

## Scalability Limits
Our benchmarks (N=2 to N=12) revealed two distinct scaling barriers:
1.  **Runtime (Exponential):** The runtime exhibits an $O(2^{3N})$ complexity curve due
to the matrix multiplication $L L^\dagger$. While execution is sub-second for $N \le 10$
on a T4 GPU, it spikes significantly at $N=12$, confirming that full-rank density matrix reconstruction is intractable for large $N$.
2.  **Fidelity (Curse of Dimensionality):** The initial (untrained) fidelity drops as $1/2^N$. For $N=12$, the random starting fidelity is effectively $0.00$. This confirms that random guessing fails at scale and gradient-based optimization is strictly required.

## Ablation Findings: Scale Invariance
We hypothesized that increasing initialization noise would degrade performance. However, our ablation study showed **identical fidelity** across different noise scales (0.01 vs 5.0).
* **Finding:** The model is **scale-invariant**. Because we normalize the trace ($\rho / \text{Tr}(\rho)$), multiplying the parameters $L$ by a scalar $k$ cancels out ($k^2/k^2 = 1$).
* **Implication:** Future ablations must vary the *structure* of the noise (e.g., adding noise to an Identity matrix) rather than just the magnitude to impact the optimization landscape.

## Future Experiments
* **Low-Rank Approximation:** Instead of a full-rank $L$ ($2^N \times 2^N$), we could learn a rectangular $L$ ($2^N \times r$) where $r \ll 2^N$. This would drastically reduce memory usage.
* **Classical Shadows:** Implementing shadow tomography could estimate observables without reconstructing the full $2^N \times 2^N$ matrix.

