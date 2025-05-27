import torch
import torch.nn.functional as F

__all__ = ['mape_loss', 'rgb_loss']

def rgb_loss(pred_rgb: torch.Tensor, true_rgb: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Simple L1 loss on RGB values (optionally scaled by *alpha*)."""
    return alpha * (pred_rgb - true_rgb).abs().mean()

def mape_loss(pred: torch.Tensor, target: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    """Mean Absolute Percentage Error–style loss used in the original extra/loss.py.

    This matches the previous implementation: squared error scaled by the
    inverse magnitude of the target, with a small constant for numerical
    stability.
    """
    diff_sq = (pred - target) ** 2  # squared error
    scale = 1.0 / (target.abs() + 1e-1)
    loss = diff_sq * scale
    if reduction == 'mean':
        return loss.mean()
    return loss 