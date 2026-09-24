# visualization_spi.py - Visualization generation for SPI forecasts

import os

import numpy as np
import torch

from dataset import SPIDataset
from plots import plot_map, save_geotiff


# ============================================================================
# SPI CLASSIFICATION
# ============================================================================

def spi_to_class(spi: np.ndarray) -> np.ndarray:
    """
    Convert SPI values to drought/wetness classes (7 classes).

    Class mapping:
        0: Extreme drought (SPI ≤ -2.0)
        1: Severe drought (-2.0 < SPI ≤ -1.5)
        2: Moderate drought (-1.5 < SPI ≤ -1.0)
        3: Near normal (-1.0 < SPI ≤ 0.0)
        4: Near normal (0.0 < SPI ≤ 1.0)
        5: Moderate wet (1.0 < SPI ≤ 1.5)
        6: Severe/Extreme wet (SPI > 1.5)

    Args:
        spi: Array of SPI values

    Returns:
        Integer class array of same shape
    """
    out = np.zeros_like(spi, dtype=np.int64)

    out[spi <= -2.0] = 0
    out[(spi > -2.0) & (spi <= -1.5)] = 1
    out[(spi > -1.5) & (spi <= -1.0)] = 2
    out[(spi > -1.0) & (spi <= 0.0)] = 3
    out[(spi > 0.0) & (spi <= 1.0)] = 4
    out[(spi > 1.0) & (spi <= 1.5)] = 5
    out[spi > 1.5] = 6

    return out


# ============================================================================
# PREDICTION EXTRACTION
# ============================================================================

def extract_predictions_lstm(model, dataset, device) -> tuple:
    """
    Extract predictions from the (single-instant) ConvLSTM3D model.

    Each sample in `dataset` already targets one SPI instant, Q steps after
    its input window (see SPIDataset) - a single forward pass per sample,
    no autoregressive rollout.

    Args:
        model: Trained ConvLSTM3D model
        dataset: SPIDataset instance
        device: Torch device

    Returns:
        tuple: (real_list, pred_list), each one (H, W) grid per sample
    """
    model.eval()
    real = []
    pred = []

    with torch.no_grad():
        for x, y, _ in dataset:
            x = x.unsqueeze(0).to(device)
            pred_grid = model(x).squeeze(0).cpu().numpy()

            real.append(y.numpy())
            pred.append(pred_grid)

    return real, pred


def extract_predictions_classic(model, dataset, H: int, W: int) -> tuple:
    """
    Extract predictions from a classical model (RF or XGBoost), which
    predicts the same single target instant as the ConvLSTM3D.

    Args:
        model: Trained sklearn/XGBoost regressor (single-output)
        dataset: SPIDataset instance
        H: Number of latitude pixels
        W: Number of longitude pixels

    Returns:
        tuple: (real_list, pred_list), each one (H, W) grid per sample
    """
    all_real = []
    all_pred = []

    for idx in range(len(dataset)):
        x, y, _ = dataset[idx]

        X_all = []
        pixel_positions = []

        x_np = x.numpy()

        # Extract features for all valid pixels
        for i in range(H):
            for j in range(W):
                window = x_np[:, :, i, j]
                features = window.reshape(-1)

                # Skip pixels with excessive NaN values
                if np.isnan(features).sum() > 0.5 * len(features):
                    continue

                X_all.append(features)
                pixel_positions.append((i, j))

        if len(X_all) == 0:
            continue

        X_all = np.asarray(X_all, dtype=np.float32)
        X_all = np.nan_to_num(X_all, nan=0.0)

        preds_all = model.predict(X_all)

        y_np = y.numpy()
        pred_grid = np.full((H, W), np.nan)
        for pred_val, (i, j) in zip(preds_all, pixel_positions):
            pred_grid[i, j] = pred_val

        all_real.append(y_np)
        all_pred.append(pred_grid)

    return all_real, all_pred


# ============================================================================
# MAIN VISUALIZATION GENERATOR
# ============================================================================

def generate_visualizations(model, df_pr, df_spi, P: int, Q: int, indices,
                            device, model_name: str, out_dir: str,
                            period: str = "test", model_type: str = "lstm") -> None:
    """
    Generate visualizations: observed SPI, predicted SPI, MAE, and accuracy
    maps for the single target instant (Q steps after the P-step input
    window - see SPIDataset).

    Args:
        model: Trained model (ConvLSTM3D or sklearn model)
        df_pr: Precipitation dataframe
        df_spi: SPI dataframe
        P: Input sequence length
        Q: Forecast horizon, in instants ahead (for logging only)
        indices: Train/val/test indices
        device: Torch device
        model_name: Name of the model (for logging)
        out_dir: Output directory for visualizations
        period: "train", "val", or "test"
        model_type: "lstm" or "classic"
    """
    os.makedirs(out_dir, exist_ok=True)

    # Create dataset for the specified period
    ds = SPIDataset(df_pr, df_spi, P, Q, period, indices)

    if len(ds) == 0:
        print(f"  ⚠ No {period} data for {model_name}")
        return

    # Get spatial dimensions
    lats = np.array(sorted(df_spi.index.get_level_values(0).unique(), reverse=True))
    lons = np.array(sorted(df_spi.index.get_level_values(1).unique()))
    H, W = len(lats), len(lons)

    print(f"  Generating visualizations for {model_name} ({period} period, {len(ds)} samples)...")

    # Extract predictions (one (H, W) grid per sample - single target instant)
    if model_type == "lstm":
        real, pred = extract_predictions_lstm(model, ds, device)
    else:
        real, pred = extract_predictions_classic(model, ds, H, W)

    if len(real) == 0:
        print(f"  ⚠ No valid predictions extracted for {model_name}")
        return

    # Calculate mean maps
    mean_real = np.nanmean(real, axis=0)
    mean_pred = np.nanmean(pred, axis=0)

    # Calculate MAE (mean absolute error)
    mae_map = np.nanmean([np.abs(r - p) for r, p in zip(real, pred)], axis=0)

    # Calculate classification accuracy
    acc_map = np.nanmean(
        [spi_to_class(r) == spi_to_class(p) for r, p in zip(real, pred)],
        axis=0
    ) * 100

    # Save GeoTIFFs
    save_geotiff(mean_real, lats, lons, os.path.join(out_dir, "spi_observed.tif"))
    save_geotiff(mean_pred, lats, lons, os.path.join(out_dir, "spi_predicted.tif"))
    save_geotiff(mae_map, lats, lons, os.path.join(out_dir, "mae_spatial.tif"))
    save_geotiff(acc_map, lats, lons, os.path.join(out_dir, "accuracy_spatial.tif"))

    # Save PDF plots
    plot_map(
        mean_real, lats, lons, cmap='RdBu', vmin=-3, vmax=3,
        cbar_label="SPI", save_path=os.path.join(out_dir, "spi_observed.pdf")
    )
    plot_map(
        mean_pred, lats, lons, cmap='RdBu', vmin=-3, vmax=3,
        cbar_label="SPI", save_path=os.path.join(out_dir, "spi_predicted.pdf")
    )
    plot_map(
        mae_map, lats, lons, cmap='YlOrRd', cbar_label="|SPI error|",
        save_path=os.path.join(out_dir, "mae_spatial.pdf")
    )
    plot_map(
        acc_map, lats, lons, cmap='YlGn', vmin=0, vmax=100,
        cbar_label="Accuracy (%)", save_path=os.path.join(out_dir, "accuracy_spatial.pdf")
    )

    print(f"  ✅ Visualizations saved to {out_dir}")
