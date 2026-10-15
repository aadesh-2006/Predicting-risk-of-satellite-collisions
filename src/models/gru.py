"""
PyTorch GRU Baseline Model for Satellite Collision Risk Prediction
Faithful reproduction of Tulczyjew et al. (2021) / DunderMifflin.

Architecture:
- Recurrent Core: 2-layer GRU (hidden size 36, batch_first=True, dropout=0.0)
- MLP Regression Head: Linear(36, 128) -> PReLU -> Linear(128, 256) -> PReLU -> Linear(256, 64) -> PReLU -> Linear(64, 1)
- Loss Function: Weighted MSE (weight=1.0 for risk >= -6, weight=0.1 for risk < -6)
- Optimizer: Adam(lr=0.001)
"""

from typing import Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam


class SatelliteGRU(nn.Module):
    """
    GRU Regressor baseline predicting final satellite collision risk (log10 Pc)
    from historical sequential Conjunction Data Messages (CDMs).
    """
    def __init__(
        self,
        input_size: int = 64,
        hidden_size: int = 36,
        num_layers: int = 2,
        dropout: float = 0.0,
        lr: float = 0.001,
        device: torch.device = torch.device("cpu")
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        # Recurrent Core
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        ).to(device)

        # Multi-layer Perceptron (MLP) Regression Head
        self.hat = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.PReLU(),
            nn.Linear(128, 256),
            nn.PReLU(),
            nn.Linear(256, 64),
            nn.PReLU(),
            nn.Linear(64, 1)
        ).to(device)

        self.optimizer = Adam(self.parameters(), lr=lr)
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a sequence or batch of sequences.
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size) or (seq_len, input_size)
        Returns:
            Scalar risk prediction (batch_size, 1) or (1,)
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Shape: (1, seq_len, input_size)

        assert x.dim() == 3, f"Expected 3D tensor (B, T, D), got {x.shape}"
        assert x.shape[-1] == self.input_size, (
            f"Expected input dimension {self.input_size}, got {x.shape[-1]}"
        )
        assert torch.isfinite(x).all(), "Input tensor contains non-finite values (NaN/Inf)"

        x = x.to(self.device)

        # GRU returns (output, h_n). output shape: (batch_size, seq_len, hidden_size)
        gru_out, _ = self.gru(x)

        # Extract last timestep's hidden state: shape (batch_size, hidden_size)
        last_hidden = gru_out[:, -1, :]

        # Pass through MLP head to predict scalar risk
        y_hat = self.hat(last_hidden)  # Shape: (batch_size, 1)

        assert torch.isfinite(y_hat).all(), "Predicted output contains non-finite values"
        return y_hat.squeeze(-1)  # Shape: (batch_size,)

    def compute_weighted_mse_loss(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        threshold: float = -6.0,
        low_risk_weight: float = 0.1,
        high_risk_weight: float = 1.0
    ) -> torch.Tensor:
        """
        Computes the authors' weighted MSE loss:
        Penalizes critical high-risk events (target >= -6.0) with weight 1.0,
        and low-risk events (target < -6.0) with weight 0.1.
        """
        assert y_hat.shape == y.shape, f"Shape mismatch: y_hat {y_hat.shape} vs y {y.shape}"
        assert torch.isfinite(y).all(), "Ground truth targets contain NaN/Inf"

        y = y.to(self.device)
        y_hat = y_hat.to(self.device)

        # Assign sample weights based on ground truth risk threshold
        weights = torch.where(
            y >= threshold,
            torch.tensor(high_risk_weight, device=self.device, dtype=torch.float32),
            torch.tensor(low_risk_weight, device=self.device, dtype=torch.float32)
        )

        squared_errors = (y_hat - y) ** 2
        weighted_loss = torch.mean(weights * squared_errors)

        assert torch.isfinite(weighted_loss), "Computed loss is non-finite (NaN/Inf)"
        return weighted_loss

    def compute_standard_mse_loss(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes standard unweighted MSE loss as explicitly stated in
        Section 3.2 of Tulczyjew et al. (2021).
        """
        assert y_hat.shape == y.shape, f"Shape mismatch: y_hat {y_hat.shape} vs y {y.shape}"
        assert torch.isfinite(y).all(), "Ground truth targets contain NaN/Inf"

        y = y.to(self.device)
        y_hat = y_hat.to(self.device)

        loss = torch.mean((y_hat - y) ** 2)
        assert torch.isfinite(loss), "Computed standard MSE loss is non-finite (NaN/Inf)"
        return loss

    def compute_loss(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        loss_type: str = "weighted_mse",
        threshold: float = -6.0
    ) -> torch.Tensor:
        """
        Dispatch method supporting both authors_weighted_mse and paper_standard_mse.
        """
        if loss_type.lower() in ("standard_mse", "mse"):
            return self.compute_standard_mse_loss(y_hat, y)
        return self.compute_weighted_mse_loss(y_hat, y, threshold=threshold)
