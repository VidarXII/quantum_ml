import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import time
import os


def get_random_state_data_vectorized(n_samples=2000, shots=1024):
    """
    Generates random states and measurement data using vectorized NumPy operations
    """
    print(f"Generating {n_samples} samples (Vectorized)...")
    t0 = time.time()

    
    phi = np.random.uniform(0, 2*np.pi, n_samples)
    theta = np.random.uniform(0, 2*np.pi, n_samples)
    omega = np.random.uniform(0, 2*np.pi, n_samples)

    
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    e_mi_phi = np.exp(-1j * phi / 2)
    e_pi_phi = np.exp(1j * phi / 2)
    e_mi_omega = np.exp(-1j * omega / 2)
    e_pi_omega = np.exp(1j * omega / 2)

    
    
    psi_0 = e_mi_omega * c * e_mi_phi
    psi_1 = e_pi_omega * s * e_mi_phi
    
    states = np.stack([psi_0, psi_1], axis=1)

    states_conj = states.conj()
    rho_true = np.einsum('bi,bj->bij', states, states_conj)

    p_z_exact = np.abs(psi_0)**2
    
    p_x_exact = np.abs(psi_0 + psi_1)**2 / 2.0
    
    p_y_exact = np.abs(psi_0 - 1j * psi_1)**2 / 2.0

    p_x_noisy = np.random.binomial(shots, p_x_exact) / shots
    p_y_noisy = np.random.binomial(shots, p_y_exact) / shots
    p_z_noisy = np.random.binomial(shots, p_z_exact) / shots

    rho_real = rho_true.real.reshape(n_samples, -1)
    rho_imag = rho_true.imag.reshape(n_samples, -1)
    
    data_dict = {
        "P_X": p_x_noisy,
        "P_Y": p_y_noisy,
        "P_Z": p_z_noisy
    }
    
    for i in range(4):
        data_dict[f"rho_real_{i}"] = rho_real[:, i]
    for i in range(4):
        data_dict[f"rho_imag_{i}"] = rho_imag[:, i]

    df = pd.DataFrame(data_dict)
    print(f"Dataset generated in {time.time() - t0:.4f} seconds.")
    return df
