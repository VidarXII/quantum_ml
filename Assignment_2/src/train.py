from data_gen import get_random_state_data_vectorized, get_loaders
from model import QubitTransformer, DensityMatrixLoss

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import time
import os

# --- 3. Training Utilities ---

def compute_exact_trace_distance(rho_pred, rho_true):
    """
    Computes exact Trace Distance: T(ρ, σ) = 0.5 * Tr|ρ - σ|
    For Hermitian matrices, this is 0.5 * sum(|eigenvalues(ρ - σ)|)
    """
    diff = rho_pred - rho_true
    # Compute eigenvalues of the difference matrix (Hermitian)
    eigvals = torch.linalg.eigvalsh(diff)
    # Sum of absolute eigenvalues
    trace_dist = 0.5 * torch.sum(torch.abs(eigvals), dim=-1)
    return trace_dist

def benchmark_inference_latency(model, device, input_shape=(1, 3), n_warmup=50, n_runs=1000):
    """
    Measures inference latency per sample in microseconds (μs).
    Includes GPU synchronization for accurate timing.
    """
    model.eval()
    dummy_input = torch.randn(input_shape, device=device)
    
    # 1. Warmup (to initialize CUDA context/caches)
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy_input)
    
    # 2. Timing
    if device.type == 'cuda':
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        with torch.no_grad():
            for _ in range(n_runs):
                _ = model(dummy_input)
        end_event.record()
        torch.cuda.synchronize()
        total_time_ms = start_event.elapsed_time(end_event) # Returns ms
        avg_time_us = (total_time_ms * 1000) / n_runs
        
    else: # CPU timing
        t0 = time.time()
        with torch.no_grad():
            for _ in range(n_runs):
                _ = model(dummy_input)
        total_time_s = time.time() - t0
        avg_time_us = (total_time_s * 1e6) / n_runs

    return avg_time_us

def evaluate_comprehensive(model, test_loader, device):
    """
    Runs full evaluation: Fidelity, Trace Distance, and Latency.
    """
    model.eval()
    fidelities = []
    trace_dists = []
    
    # 1. Physics Metrics
    print("\nComputing Physics Metrics...")
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            rho_pred = model(x)
            
            # Fidelity
            prod = torch.bmm(rho_pred, y)
            fid = torch.real(prod.diagonal(dim1=-2, dim2=-1).sum(-1))
            fidelities.extend(fid.cpu().numpy())
            
            # Trace Distance
            td = compute_exact_trace_distance(rho_pred, y)
            trace_dists.extend(td.cpu().numpy())
    
    # 2. Latency Benchmark
    print("Benchmarking Latency...")
    # Measure latency for batch_size=1 (single state reconstruction)
    latency_us = benchmark_inference_latency(model, device, input_shape=(1, 3))
    
    results = {
        "mean_fidelity": np.mean(fidelities),
        "std_fidelity": np.std(fidelities),
        "mean_trace_dist": np.mean(trace_dists),
        "std_trace_dist": np.std(trace_dists),
        "latency_us": latency_us
    }
    return results

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Setup
    CONFIG = {
        'n_samples': 5000,
        'batch_size': 128,
        'epochs': 100,
        'lr': 1e-3,
        'embed_dim': 64,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    }
    
    # 2. Data (Calls data_gen.py)
    if os.path.exists("data/train_dataset_opt.csv"):
        df = pd.read_csv("data/train_dataset_opt.csv")
    else:
        # calling imported function
        df = get_random_state_data_vectorized(CONFIG['n_samples'])
        df.to_csv("data/train_dataset_opt.csv", index=False)

    # calling imported function
    train_loader, val_loader = get_loaders(df, CONFIG['batch_size'], num_workers=0)

    # 3. Model (Calls model.py)
    model = QubitTransformer(embed_dim=CONFIG['embed_dim']) # Imported class
    model.to(CONFIG['device'])
    
    criterion = DensityMatrixLoss() # Imported class
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'])

    # --- Training Loop ---
    print("\nStarting Training...")
    best_fid = 0.0
    
    for epoch in range(CONFIG['epochs']):
        # 1. Training Phase
        model.train()
        train_loss_accum = 0.0
        train_fid_accum = 0.0
        
        for x, y in train_loader:
            # Move data to GPU
            x = x.to(CONFIG['device'], non_blocking=True)
            y = y.to(CONFIG['device'], non_blocking=True)
            
            # Optimization step
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            
            # Gradient clipping (prevents exploding gradients)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Track metrics (no_grad for speed)
            with torch.no_grad():
                train_loss_accum += loss.item()
                # Use local utility function
                train_fid_accum += compute_fidelity_batch(pred, y).mean().item()

        # 2. Validation Phase
        model.eval()
        val_fid_accum = 0.0
        
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(CONFIG['device'], non_blocking=True)
                y = y.to(CONFIG['device'], non_blocking=True)
                
                pred = model(x)
                val_fid_accum += compute_fidelity_batch(pred, y).mean().item()

        # 3. Stats & Logging
        avg_train_loss = train_loss_accum / len(train_loader)
        avg_train_fid = train_fid_accum / len(train_loader)
        avg_val_fid = val_fid_accum / len(val_loader)
        
        # Update learning rate scheduler
        scheduler.step(avg_val_fid)
        
        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"Ep {epoch+1:3d} | Loss: {avg_train_loss:.4f} | "
                  f"Train Fid: {avg_train_fid:.4f} | Val Fid: {avg_val_fid:.4f}")

        # Save Checkpoint
        if avg_val_fid > best_fid:
            best_fid = avg_val_fid
            torch.save(model.state_dict(), "best_model_opt.pt")

    print(f"\nTraining Complete. Best Validation Fidelity: {best_fid:.4f}")
    
    # Load best weights for final evaluation
    model.load_state_dict(torch.load("best_model_opt.pt"))
    # 5. Evaluation (Calls local utility functions)
    # We reuse val_loader for the final comprehensive test in this simplified setup
    results = evaluate_comprehensive(model, val_loader, CONFIG['device'])
    
    print("\nFinal Results:")
    print(results)
  
if __name__ == "__main__":
    # Settings
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 128
    
    # [Assuming data generation and training happened as per previous script...]
    # For this snippet, let's assume we are loading the saved model
    
    # 1. Load Data (Reusable from previous step)
    if os.path.exists("train_dataset_opt.csv"):
        df = pd.read_csv("train_dataset_opt.csv")
    else:
        # Generate small set if file missing for demo
        from previous_script import get_random_state_data_vectorized # (Pseudo-import)
        # Note: In real run, ensure get_random_state_data_vectorized is defined
        pass 
        
    # Recreate validation loader for testing
    from sklearn.model_selection import train_test_split
    # ... (Data loading logic same as before) ...
    # Quick reconstruction for context:
    X = df[["P_X", "P_Y", "P_Z"]].values.astype(np.float32)
    r_real = df[[f"rho_real_{i}" for i in range(4)]].values
    r_imag = df[[f"rho_imag_{i}" for i in range(4)]].values
    y = (r_real + 1j * r_imag).reshape(-1, 2, 2).astype(np.complex64)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Load Model
    model = QubitTransformer(embed_dim=64)
    model.to(DEVICE)
    
    if os.path.exists("best_model_opt.pt"):
        model.load_state_dict(torch.load("best_model_opt.pt"))
        print("Loaded best model weights.")
    else:
        print("Warning: No trained weights found. Running with random weights.")

    # 3. Run Evaluation
    metrics = evaluate_comprehensive(model, test_loader, DEVICE)
    
    print("\n" + "="*40)
    print("       FINAL PERFORMANCE REPORT       ")
    print("="*40)
    print(f"Mean Fidelity:       {metrics['mean_fidelity']:.4f} ± {metrics['std_fidelity']:.4f}")
    print(f"Mean Trace Dist:     {metrics['mean_trace_dist']:.4f} ± {metrics['std_trace_dist']:.4f}")
    print(f"Inference Latency:   {metrics['latency_us']:.2f} μs / state")
    print("="*40)
