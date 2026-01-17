import torch
import torch.nn as nn
import torch.nn.functional as F

class CholeskyPhysicsHead(nn.Module):
    """
    Physical constraint layer enforcing positive semi-definite density matrices.
    """
    def __init__(self):
        super().__init__()
        # Buffer for trace calculation to avoid re-creation
        self.register_buffer('epsilon', torch.tensor(1e-8))

    def forward(self, x):
        batch_size = x.shape[0]
        
        # x: [l00_raw, l11_raw, l10_real, l10_imag]
        l_diag = F.softplus(x[:, :2]) + 1e-4
        l00, l11 = l_diag[:, 0], l_diag[:, 1]
        l_off_r, l_off_i = x[:, 2], x[:, 3]

        # Construct L matrix efficiently
        # L = [[l00, 0], [l10, l11]]
        L = torch.zeros((batch_size, 2, 2), dtype=torch.complex64, device=x.device)
        
        # Set diagonals
        L[:, 0, 0] = l00.to(torch.complex64)
        L[:, 1, 1] = l11.to(torch.complex64)
        
        # Set off-diagonal L[1,0]
        L[:, 1, 0] = torch.complex(l_off_r, l_off_i)
        
        # rho = L @ L^dag
        # batch_matrix_multiply
        rho_un = torch.bmm(L, L.mH) # .mH is 'conjugate transpose' shorthand in newer PyTorch
        
        # Normalize trace
        trace = rho_un.diagonal(dim1=-2, dim2=-1).sum(-1)
        rho = rho_un / (trace.view(-1, 1, 1) + self.epsilon)
        
        return rho

class QubitTransformer(nn.Module):
    """
    Optimized Transformer for Qubit Tomography.
    Reduced embed_dim for efficiency on single-qubit tasks.
    """
    def __init__(self, embed_dim=64, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        
        self.input_proj = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        self.pos_embed = nn.Parameter(torch.randn(1, 3, embed_dim) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2, # Reduced expansion factor
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim), # Flattened (B, 3*dim) -> (B, dim)
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 4) # Output Cholesky params
        )
        
        self.physics_head = CholeskyPhysicsHead()

    def forward(self, x):
        # x: (Batch, 3) -> [P_X, P_Y, P_Z]
        # Normalize inputs [-1, 1]
        x_scaled = 2.0 * x - 1.0
        
        # (Batch, 3) -> (Batch, 3, 1) -> (Batch, 3, Embed)
        x_emb = self.input_proj(x_scaled.unsqueeze(-1)) + self.pos_embed
        
        features = self.transformer(x_emb)
        
        # Flatten all basis tokens: (Batch, 3*Embed)
        features_flat = features.reshape(features.shape[0], -1)
        
        l_params = self.decoder(features_flat)
        rho = self.physics_head(l_params)
        return rho

class DensityMatrixLoss(nn.Module):
    def __init__(self, alpha=0.7):
        super().__init__()
        self.alpha = alpha

    def forward(self, rho_pred, rho_true):
        # Frobenius Norm (fast)
        diff = rho_pred - rho_true
        # sum(|diff|^2)
        frobenius = torch.real((diff * diff.conj()).sum(dim=(-2, -1))).mean()

        # Fidelity Approximation: Tr(rho_pred @ rho_true) for pure states
        # Full fidelity involves sqrtm which is unstable on GPU, product trace is stable proxy
        product = torch.matmul(rho_pred, rho_true)
        fidelity = torch.real(product.diagonal(dim1=-2, dim2=-1).sum(-1)).mean()
        
        return self.alpha * (1 - fidelity) + (1 - self.alpha) * frobenius
