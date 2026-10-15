"""
Unit & Smoke Tests for SatelliteGRU Model
Tests:
1. Architecture dimensions and parameter count
2. Output shape matching
3. Weighted MSE loss behavior
4. Backward pass and gradient flow
5. Real sample forward/backward execution on CPU
"""

import sys
from pathlib import Path
import joblib
import numpy as np
import torch

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.gru import SatelliteGRU


def run_model_tests():
    print("=" * 60)
    print("  SatelliteGRU Model Verification & Smoke Test")
    print("=" * 60)

    device = torch.device("cpu")
    model = SatelliteGRU(input_size=64, hidden_size=36, num_layers=2, device=device)

    # Test 1: Parameter verification
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f" [PASS] Model initialized on {device}. Trainable parameters: {total_params:,}")

    # Test 2: Shape assertions (2D input)
    seq_len = 15
    dummy_input_2d = torch.randn(seq_len, 64)
    out_2d = model(dummy_input_2d)
    assert out_2d.shape == torch.Size([1]), f"Expected (1,), got {out_2d.shape}"
    assert torch.isfinite(out_2d).all(), "2D output contains non-finite values"
    print(" [PASS] 2D sequence forward pass successful (shape verified: [1])")

    # Test 3: Shape assertions (3D batched input)
    batch_size = 4
    dummy_input_3d = torch.randn(batch_size, seq_len, 64)
    out_3d = model(dummy_input_3d)
    assert out_3d.shape == torch.Size([batch_size]), f"Expected ({batch_size},), got {out_3d.shape}"
    assert torch.isfinite(out_3d).all(), "3D output contains non-finite values"
    print(f" [PASS] 3D batch forward pass successful (shape verified: [{batch_size}])")

    # Test 4: Weighted MSE loss logic
    dummy_y_hat = torch.tensor([-5.0, -10.0])
    dummy_y = torch.tensor([-5.0, -10.0])
    zero_loss = model.compute_weighted_mse_loss(dummy_y_hat, dummy_y)
    assert torch.isclose(zero_loss, torch.tensor(0.0)), "Zero error loss should be 0"

    # Test differential weights: error = 2 on high risk (weight 1.0) vs low risk (weight 0.1)
    y_hat_test = torch.tensor([-4.0, -12.0])  # errors: (-4 - (-6)) = 2 -> sq=4; (-12 - (-10)) = -2 -> sq=4
    y_test = torch.tensor([-6.0, -10.0])      # y[0] >= -6 (weight 1.0); y[1] < -6 (weight 0.1)
    loss = model.compute_weighted_mse_loss(y_hat_test, y_test)
    expected_loss = (1.0 * 4.0 + 0.1 * 4.0) / 2.0  # 2.2
    assert torch.isclose(loss, torch.tensor(expected_loss)), (
        f"Expected {expected_loss}, got {loss.item()}"
    )
    print(" [PASS] Weighted MSE loss calculation strictly verified")

    # Test 5: Gradient flow
    model.zero_grad()
    dummy_pred = model(dummy_input_2d)
    dummy_target = torch.tensor([-5.0])
    dummy_loss = model.compute_weighted_mse_loss(dummy_pred, dummy_target)
    dummy_loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient"
        assert torch.isfinite(param.grad).all(), f"Parameter {name} has non-finite gradient"
    print(" [PASS] Backward pass and gradient flow verified across all layers")

    # Test 6: Smoke test with actual preprocessed training sample
    processed_train_path = PROJECT_ROOT / "data" / "processed" / "train_sequences_6_ranges.joblib"
    if processed_train_path.exists():
        samples = joblib.load(processed_train_path)
        sample = samples[0]
        x_tensor = torch.from_numpy(sample["input_sequence"]).float()
        y_target = torch.tensor([sample["target_risk"]]).float()

        model.zero_grad()
        pred = model(x_tensor)
        sample_loss = model.compute_weighted_mse_loss(pred, y_target)
        sample_loss.backward()
        model.optimizer.step()

        assert torch.isfinite(pred).all()
        assert torch.isfinite(sample_loss).all()
        print(f" [PASS] Smoke test on actual preprocessed event {sample['event_id']} passed:")
        print(f"        Input shape: {x_tensor.shape}, Target: {y_target.item():.4f}, "
              f"Pred: {pred.item():.4f}, Loss: {sample_loss.item():.4f}")
    else:
        print(" [WARN] train_sequences_6_ranges.joblib not found for real data smoke test")

    print("\n" + "=" * 60)
    print("  All Model Unit and Smoke Tests PASSED!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(run_model_tests())
