# metrics.py - Evaluation metrics for drought forecasting

import torch


def _valid_mask(yt: torch.Tensor, yp: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """Combine finiteness with an optional external validity mask (>0.5 = valid)."""
    valid = torch.isfinite(yt) & torch.isfinite(yp)
    if mask is not None:
        valid = valid & (mask > 0.5)
    return valid


def wi(yt: torch.Tensor, yp: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """
    Willmott's Index of Agreement (WI).

    Measures how well predictions match observations, ranging from 0 to 1,
    where 1 indicates perfect agreement.

    Args:
        yt: Ground truth values
        yp: Predicted values
        mask: Optional validity mask, same shape as yt/yp (>0.5 = valid).
            Required whenever yt/yp may contain zero-filled placeholders for
            pixels/months where SPI could not be computed: those
            placeholders are finite (0.0) and would otherwise be silently
            scored as real observations.

    Returns:
        WI value as a torch.Tensor (scalar)
    """
    valid = _valid_mask(yt, yp, mask)
    yt, yp = yt[valid], yp[valid]

    if yt.numel() == 0:
        return torch.tensor(float("nan"))

    ybar = yt.mean()
    sse = ((yt - yp) ** 2).sum()
    denom = ((yp - ybar).abs() + (yt - ybar).abs()).pow(2).sum()

    return 1 - sse / denom


def rmse(yt: torch.Tensor, yp: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """
    Root Mean Square Error (RMSE).

    Args:
        yt: Ground truth values
        yp: Predicted values
        mask: Optional validity mask, same shape as yt/yp (>0.5 = valid).

    Returns:
        RMSE value as a torch.Tensor (scalar)
    """
    valid = _valid_mask(yt, yp, mask)
    yt, yp = yt[valid], yp[valid]

    if yt.numel() == 0:
        return torch.tensor(float("nan"))

    return torch.sqrt(torch.mean((yt - yp) ** 2))


def mae(yt: torch.Tensor, yp: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """
    Mean Absolute Error (MAE).

    Args:
        yt: Ground truth values
        yp: Predicted values
        mask: Optional validity mask, same shape as yt/yp (>0.5 = valid).

    Returns:
        MAE value as a torch.Tensor (scalar)
    """
    valid = _valid_mask(yt, yp, mask)
    yt, yp = yt[valid], yp[valid]

    if yt.numel() == 0:
        return torch.tensor(float("nan"))

    return torch.mean(torch.abs(yt - yp))


def compute_all_metrics(yt: torch.Tensor, yp: torch.Tensor, mask: torch.Tensor = None) -> dict:
    """
    Compute all three metrics (WI, RMSE, MAE) at once.

    Args:
        yt: Ground truth values
        yp: Predicted values
        mask: Optional validity mask, same shape as yt/yp (>0.5 = valid).

    Returns:
        Dictionary with keys: "wi", "rmse", "mae"
    """
    return {
        "wi": float(wi(yt, yp, mask)),
        "rmse": float(rmse(yt, yp, mask)),
        "mae": float(mae(yt, yp, mask))
    }