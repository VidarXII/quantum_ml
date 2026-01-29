### **4. Written Summary (`docs/report.md`)**
```markdown
# Task 6: Technical Report & Reflection

## Method: Cholesky Surrogate
To reconstruct density matrices for $N$ qubits, we utilized a **Cholesky Ansatz** where $\rho = \frac{L L^\dagger}{\text{Tr}(L L^\dagger)}$.
* **Why:** This strictly enforces physical constraints (Positive Semi-Definite, Unit Trace) by construction.
* **Optimization:** We treated the elements of $L$ as learnable PyTorch parameters and optimized them using gradient descent to maximize Fidelity with the target state.

## Scalability Limits
Our benchmarks (N=2 to N=12) revealed two distinct scaling barriers:
1.  **Runtime (Exponential):** The runtime exhibits an $O(2^{3N})$ complexity curve due to the matrix multiplication $L L^\dagger$. While execution is sub-second for $N \le 10$ on a T4 GPU, it spikes significantly at $N=12$, confirming that full-rank density matrix reconstruction is intractable for large $N$.
2.  **Fidelity (Curse of Dimensionality):** The initial (untrained) fidelity drops as $1/2^N$. For $N=12$, the random starting fidelity is effectively $0.00$. This confirms that random guessing fails at scale and gradient-based optimization is strictly required.

## Ablation Findings: Scale Invariance
We hypothesized that increasing initialization noise would degrade performance. However, our ablation study showed **identical fidelity** across different noise scales (0.01 vs 5.0).
* **Finding:** The model is **scale-invariant**. Because we normalize the trace ($\rho / \text{Tr}(\rho)$), multiplying the parameters $L$ by a scalar $k$ cancels out ($k^2/k^2 = 1$).
* **Implication:** Future ablations must vary the *structure* of the noise (e.g., adding noise to an Identity matrix) rather than just the magnitude to impact the optimization landscape.

## Future Experiments
* **Low-Rank Approximation:** Instead of a full-rank $L$ ($2^N \times 2^N$), we could learn a rectangular $L$ ($2^N \times r$) where $r \ll 2^N$. This would drastically reduce memory usage.
* **Classical Shadows:** Implementing shadow tomography could estimate observables without reconstructing the full $2^N \times 2^N$ matrix.
