# generate_panel.py - Generate compact panel figure for all horizons with TIFF export

"""
Generate compact panel figure with predictions for all horizons (Q=1,3,6,9,12).
Single P fixed. Each row = horizon, columns = Observed, RF, XGBoost, ConvLSTM3D.
Each horizon corresponds to a specific future month from the base date.
Also exports each image as individual GeoTIFF files.
"""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
from joblib import load
from matplotlib.colors import TwoSlopeNorm
from rasterio.crs import CRS
from rasterio.transform import from_origin

from config import BASE_DIR, DATA_PATH, REF_DATE, SPI_SCALE_FIXED, TRAIN_END_YEAR
from dataset import SPIDataset
from model_convlstm3d import ConvLSTM3D
from plots import CMAP_SPI, set_journal_style
from utils_data import load_grid_data, load_or_calculate_spi

# ============================================================================
# CONFIGURATION
# ============================================================================

P_FIXED = 12  # Fixed past window (P = 3, 6, 9, or 12)

# Base date for prediction (last observed date in the input window)
BASE_DATE = "2024-12"  # December 2024 is the last observed month

# Horizons in months (lead times) - must match trained models
HORIZONS = [1, 3, 6, 9, 12]

# Model names and display names
MODELS = ["RF", "XGBoost", "ConvLSTM3D"]
MODEL_DISPLAY = ["RF", "XGBoost", "ConvLSTM3D"]

# Paths
EXPERIMENTS_BASE = BASE_DIR
OUTPUT_DIR = Path("panel_figures")
OUTPUT_DIR.mkdir(exist_ok=True)

TIFF_OUTPUT_DIR = OUTPUT_DIR / "tiff_exports"
TIFF_OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# GEOTIFF UTILITIES
# ============================================================================

def save_as_tiff(data: np.ndarray, lats: np.ndarray, lons: np.ndarray,
                 output_path: str, description: str = "",
                 nodata: float = -9999.0) -> None:
    """
    Save a 2D numpy array as GeoTIFF file.

    Args:
        data: 2D array [H, W]
        lats: Array of latitudes (sorted descending for origin='upper')
        lons: Array of longitudes (sorted ascending)
        output_path: Path to save the TIFF file
        description: Description to add to TIFF metadata
        nodata: Value to use for NoData pixels
    """
    H, W = data.shape

    lon_min, lon_max = lons.min(), lons.max()
    lat_min, lat_max = lats.min(), lats.max()

    lon_res = (lon_max - lon_min) / (W - 1) if W > 1 else 1.0
    lat_res = (lat_max - lat_min) / (H - 1) if H > 1 else 1.0

    transform = from_origin(lon_min, lat_max, lon_res, lat_res)
    crs = CRS.from_epsg(4326)

    data_with_nodata = np.where(np.isnan(data), nodata, data)

    with rasterio.open(
        output_path, 'w', driver='GTiff',
        height=H, width=W, count=1, dtype=data_with_nodata.dtype,
        crs=crs, transform=transform, nodata=nodata,
        compress='lzw', tiled=True, blockxsize=256, blockysize=256
    ) as dst:
        dst.write(data_with_nodata, 1)
        dst.update_tags(description=description)

    print(f"    TIFF saved: {output_path}")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_target_date(base_date: str, horizon_months: int) -> pd.Timestamp:
    """Calculate target date by adding horizon months to base date."""
    base_dt = pd.to_datetime(base_date)
    return base_dt + pd.DateOffset(months=horizon_months)


def load_convlstm_model(exp_dir: str, device) -> ConvLSTM3D:
    """Load trained ConvLSTM3D model."""
    model_path = os.path.join(exp_dir, "ConvLSTM3D", "best_model.pt")

    if not os.path.exists(model_path):
        return None

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    hidden = checkpoint.get('hidden', (64, 32, 16))
    dropout_p = checkpoint.get('dropout', 0.3)

    model = ConvLSTM3D(hidden=hidden, dropout_p=dropout_p).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model


def load_classic_model(exp_dir: str, model_name: str):
    """Load trained classical model (RF or XGBoost)."""
    model_path = os.path.join(exp_dir, model_name, "best_model.joblib")

    if not os.path.exists(model_path):
        return None

    return load(model_path)


def get_input_for_date(date: str, df_pr, df_spi, dates, lats, lons, P: int) -> torch.Tensor:
    """Get input tensor for a specific date (the last observed date)."""
    H, W = len(lats), len(lons)
    date_dt = pd.to_datetime(date)

    # Find closest available date
    try:
        idx_target = list(dates).index(date_dt)
    except ValueError:
        date_mask = dates <= date_dt
        if not date_mask.any():
            raise ValueError(f"Date {date} is before first available date {dates[0]}")
        idx_target = np.where(date_mask)[0][-1]
        date_dt = dates[idx_target]

    idx_start = idx_target - P

    if idx_start < 0:
        raise ValueError(f"Insufficient data before {date_dt.date()} (need P={P} months)")

    x_pr = np.full((P, H, W), np.nan, dtype=np.float32)
    x_spi = np.full((P, H, W), np.nan, dtype=np.float32)

    for t, idx_t in enumerate(range(idx_start, idx_target)):
        date_t = dates[idx_t]

        # Precipitation
        grid_pr = (df_pr[date_t].unstack(level=1).reindex(index=lats, columns=lons))
        x_pr[t] = grid_pr.values.astype(np.float32)

        # SPI
        grid_spi = (df_spi[date_t].unstack(level=1).reindex(index=lats, columns=lons))
        x_spi[t] = grid_spi.values.astype(np.float32)

    # Compute delta SPI
    x_dspi = np.zeros_like(x_spi)
    if P > 1:
        x_dspi[1:] = x_spi[1:] - x_spi[:-1]

    # Stack channels
    x = np.stack([x_pr, x_spi, x_dspi], axis=1)  # [P, 3, H, W]
    x = np.nan_to_num(x, nan=0.0)

    return torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # [1, P, 3, H, W]


def get_observed_spi_for_date(date, df_spi, lats, lons) -> np.ndarray:
    """Get observed SPI for a specific target date."""
    date_dt = pd.to_datetime(date)
    return (df_spi[date_dt].unstack(level=1).reindex(index=lats, columns=lons).values)


def get_predictions_for_horizon(model, model_type: str, x_tensor: torch.Tensor,
                                lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """
    Get the single-instant prediction from a model trained for one specific
    (P, Q) - a single forward pass, no autoregressive rollout (each model
    already predicts exactly one target instant, Q steps past its input
    window - see SPIDataset). Every caller of this function loads a model
    trained specifically for the Q it wants, so there is no separate
    "target_horizon" to select afterward.

    Args:
        model: Trained model
        model_type: "ConvLSTM3D", "RF", or "XGBoost"
        x_tensor: Input tensor [1, P, 3, H, W]
        lats, lons: Grid coordinates

    Returns:
        2D numpy array of predictions [H, W]
    """
    H, W = len(lats), len(lons)

    if model_type == "ConvLSTM3D":
        with torch.no_grad():
            pred = model(x_tensor)
        return pred.detach().cpu().numpy().squeeze()

    else:  # RF or XGBoost
        x_np = x_tensor.detach().cpu().numpy()

        # Remove batch dimension if present
        while x_np.ndim > 4:
            x_np = x_np.squeeze(0)

        if x_np.ndim == 3:
            x_np = x_np.reshape(1, *x_np.shape)

        X_all = []
        pixel_positions = []

        for i in range(H):
            for j in range(W):
                window = x_np[:, :, i, j]
                features = window.reshape(-1)

                if np.isnan(features).sum() > 0.5 * len(features):
                    continue

                X_all.append(features)
                pixel_positions.append((i, j))

        if len(X_all) == 0:
            return np.full((H, W), np.nan)

        X_all = np.asarray(X_all, dtype=np.float32)
        X_all = np.nan_to_num(X_all, nan=0.0)

        preds_all = model.predict(X_all)

        pred_grid = np.full((H, W), np.nan)
        for pred, (i, j) in zip(preds_all, pixel_positions):
            pred_grid[i, j] = pred

        return pred_grid


def find_model_config(model_name: str, target_P: int, target_Q: int,
                      experiments_base: str):
    """Find experiment directory for given P and Q."""
    exp_dir = os.path.join(experiments_base, f"P{target_P}_Q{target_Q}")

    if not os.path.exists(exp_dir):
        return None

    if model_name == "ConvLSTM3D":
        model_path = os.path.join(exp_dir, model_name, "best_model.pt")
    else:
        model_path = os.path.join(exp_dir, model_name, "best_model.joblib")

    if os.path.exists(model_path):
        return exp_dir
    return None


def add_thin_border(ax, linewidth: float = 0.5, color: str = 'black') -> None:
    """Add thin border around axes."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(linewidth)
        spine.set_color(color)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)


# ============================================================================
# MAIN PANEL GENERATION
# ============================================================================

def main():
    print("=" * 70)
    print("GENERATING COMPACT PANEL FOR ALL HORIZONS")
    print(f"Fixed P = {P_FIXED}")
    print(f"Base date (last observed): {BASE_DATE}")
    print(f"Horizons (months ahead): {HORIZONS}")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    # =================================================================
    # 1. LOAD DATA
    # =================================================================
    print("\n1. LOADING DATA")
    df_pr = load_grid_data(DATA_PATH)

    df_spi, indices = load_or_calculate_spi(
        df_pr,
        scale=SPI_SCALE_FIXED,
        train_end_year=TRAIN_END_YEAR,
        ref_date=REF_DATE,
        cache_dir=str(BASE_DIR),
        force_recompute=False
    )

    # Get grid coordinates
    lats = np.array(sorted(df_spi.index.get_level_values(0).unique(), reverse=True))
    lons = np.array(sorted(df_spi.index.get_level_values(1).unique()))
    H, W = len(lats), len(lons)
    dates = pd.to_datetime(df_pr.columns)

    print(f"  Grid: {H} × {W} pixels")
    print(f"  Available dates: {dates[0].date()} to {dates[-1].date()}")

    # =================================================================
    # 2. CALCULATE TARGET DATES
    # =================================================================
    print("\n2. CALCULATING TARGET DATES")
    target_dates = {}
    for q in HORIZONS:
        target_date = get_target_date(BASE_DATE, q)
        target_dates[q] = target_date
        print(f"  Q={q:2d} months → Target date: {target_date.date()}")

    # =================================================================
    # 3. PREPARE INPUT TENSOR
    # =================================================================
    print("\n3. PREPARING INPUT TENSOR")
    x_tensor = get_input_for_date(BASE_DATE, df_pr, df_spi, dates, lats, lons, P_FIXED)
    print(f"  Input tensor shape: {x_tensor.shape}")

    # =================================================================
    # 4. LOAD MODELS AND GENERATE PREDICTIONS
    # =================================================================
    print("\n4. LOADING MODELS AND GENERATING PREDICTIONS")
    predictions_by_horizon = {}

    for q in HORIZONS:
        print(f"\n  --- Q={q} (Target: {target_dates[q].date()}) ---")
        predictions_by_horizon[q] = {}

        # Get observed SPI for this target date
        spi_real = get_observed_spi_for_date(target_dates[q], df_spi, lats, lons)
        predictions_by_horizon[q]["observed"] = spi_real
        predictions_by_horizon[q]["target_date"] = target_dates[q]

        for model_name in MODELS:
            # Load model trained EXACTLY for this Q
            exp_dir = find_model_config(model_name, P_FIXED, q, EXPERIMENTS_BASE)

            if exp_dir is None:
                print(f"    ⚠ {model_name}: P={P_FIXED}, Q={q} not found")
                predictions_by_horizon[q][model_name] = None
                continue

            print(f"    ✓ {model_name}: loading P={P_FIXED}, Q={q}")

            if model_name == "ConvLSTM3D":
                model = load_convlstm_model(exp_dir, DEVICE)
                x_tensor_device = x_tensor.to(DEVICE)
            else:
                model = load_classic_model(exp_dir, model_name)
                x_tensor_device = x_tensor

            if model is None:
                predictions_by_horizon[q][model_name] = None
                continue

            pred = get_predictions_for_horizon(
                model, model_name, x_tensor_device, lats, lons
            )

            predictions_by_horizon[q][model_name] = pred
            print(f"      Prediction shape: {pred.shape}")

    # =================================================================
    # 5. SAVE INDIVIDUAL TIFFS
    # =================================================================
    print("\n5. SAVING INDIVIDUAL TIFF FILES")
    print(f"  Output directory: {TIFF_OUTPUT_DIR}")

    # Create a subdirectory for this specific prediction
    base_name = f"P{P_FIXED}_{BASE_DATE.replace('-', '')}"
    prediction_dir = TIFF_OUTPUT_DIR / base_name
    prediction_dir.mkdir(exist_ok=True)

    tiff_files = []

    for q in HORIZONS:
        q_dir = prediction_dir / f"Q{q}"
        q_dir.mkdir(exist_ok=True)

        target_date = predictions_by_horizon[q]["target_date"]
        date_str = target_date.strftime('%Y%m')

        # Save observed SPI
        observed_data = predictions_by_horizon[q]["observed"]
        if observed_data is not None:
            tiff_path = q_dir / f"observed_SPI_Q{q}_{date_str}.tif"
            save_as_tiff(
                observed_data, lats, lons, tiff_path,
                description=f"Observed SPI for Q={q}, target date={target_date.date()}"
            )
            tiff_files.append(tiff_path)

        # Save predictions for each model
        for model_name in MODELS:
            pred_data = predictions_by_horizon[q].get(model_name)
            if pred_data is not None:
                tiff_path = q_dir / f"{model_name}_prediction_Q{q}_{date_str}.tif"
                save_as_tiff(
                    pred_data, lats, lons, tiff_path,
                    description=f"{model_name} prediction for Q={q}, target date={target_date.date()}"
                )
                tiff_files.append(tiff_path)

    print(f"\n  ✓ Saved {len(tiff_files)} TIFF files")

    # =================================================================
    # 6. CREATE PANEL FIGURE
    # =================================================================
    print("\n6. CREATING PANEL FIGURE")
    set_journal_style()

    n_rows = len(HORIZONS)
    n_cols = 1 + len(MODELS)  # Observed + models

    lon_range = lons.max() - lons.min()
    lat_range = lats.max() - lats.min()
    aspect_ratio = lon_range / lat_range

    subplot_width = 2.2
    subplot_height = subplot_width / aspect_ratio

    fig_width = subplot_width * n_cols + 1.2
    fig_height = subplot_height * n_rows + 0.8

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))

    if n_rows == 1:
        axes = axes.reshape(1, -1)

    vmin, vmax = -3, 3
    cmap = CMAP_SPI  # low SPI (drought) -> red, high SPI (wet) -> blue; shared with generate_monthly_maps.py
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    extent = [lons.min(), lons.max(), lats.min(), lats.max()]
    first_im = None

    # Store label info for later positioning
    label_info = []

    for row_idx, q in enumerate(HORIZONS):
        target_date = predictions_by_horizon[q]["target_date"]

        # Column 0: Observed SPI
        ax_obs = axes[row_idx, 0]
        observed_data = predictions_by_horizon[q]["observed"]

        im = ax_obs.imshow(observed_data, cmap=cmap, norm=norm,
                          extent=extent, origin='upper', aspect='equal',
                          interpolation='bilinear')

        if first_im is None:
            first_im = im

        add_thin_border(ax_obs, linewidth=0.3, color='black')
        label_info.append((row_idx, q, ax_obs))

        # Columns 1+: Model predictions
        for col_idx, model_name in enumerate(MODELS):
            ax = axes[row_idx, col_idx + 1]
            pred = predictions_by_horizon[q].get(model_name)

            if pred is not None:
                ax.imshow(pred, cmap=cmap, norm=norm,
                         extent=extent, origin='upper', aspect='equal',
                         interpolation='bilinear')
            else:
                ax.imshow(np.full((H, W), np.nan), cmap=cmap, norm=norm,
                         extent=extent, origin='upper', aspect='equal')

            add_thin_border(ax, linewidth=0.3, color='black')

    # Column headers
    axes[0, 0].set_title('Observed SPI', fontsize=9, fontweight='bold', pad=8)
    for col_idx, model_name in enumerate(MODELS):
        axes[0, col_idx + 1].set_title(MODEL_DISPLAY[col_idx], fontsize=9, fontweight='bold', pad=8)

    # Apply tight layout
    plt.tight_layout(rect=[0.02, 0.02, 0.92, 0.96])

    # Add row labels (Q values) after tight_layout for correct positioning
    for row_idx, q, ax_obs in label_info:
        label_x = ax_obs.get_position().x0 - 0.02
        label_y = ax_obs.get_position().y0 + ax_obs.get_position().height / 2
        fig.text(label_x, label_y, f'Q={q}',
                fontsize=7, fontweight='bold',
                va='center', ha='center',
                rotation=90)

    # Colorbar
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(first_im, cax=cbar_ax)
    cbar.set_label('SPI', fontsize=9)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_ticks([-3, -2, -1, 0, 1, 2, 3])

    # Save figure
    output_path = OUTPUT_DIR / f"panel_all_horizons_P{P_FIXED}_base{BASE_DATE.replace('-', '')}.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✅ Panel saved to: {output_path}")
    print(f"   Figure size: {fig_width:.1f} × {fig_height:.1f} inches")

    # =================================================================
    # 7. SAVE METADATA FILE WITH TIFF REFERENCES
    # =================================================================
    print("\n7. SAVING METADATA FILE")

    metadata = {
        'config': {
            'P_fixed': P_FIXED,
            'base_date': BASE_DATE,
            'horizons': HORIZONS,
            'models': MODELS,
            'grid_shape': [H, W],
            'latitudes': lats.tolist(),
            'longitudes': lons.tolist()
        },
        'predictions': {},
        'tiff_files': [str(f) for f in tiff_files]
    }

    for q in HORIZONS:
        metadata['predictions'][f'Q{q}'] = {
            'target_date': str(predictions_by_horizon[q]['target_date'].date()),
            'files': {
                'observed': str(prediction_dir / f"Q{q}" / f"observed_SPI_Q{q}_{predictions_by_horizon[q]['target_date'].strftime('%Y%m')}.tif"),
                **{
                    model: str(prediction_dir / f"Q{q}" / f"{model}_prediction_Q{q}_{predictions_by_horizon[q]['target_date'].strftime('%Y%m')}.tif")
                    for model in MODELS if predictions_by_horizon[q].get(model) is not None
                }
            }
        }

    metadata_path = prediction_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Metadata saved to: {metadata_path}")

    # =================================================================
    # 8. SUMMARY
    # =================================================================
    print("\n" + "=" * 70)
    print("PREDICTIONS SUMMARY")
    print("=" * 70)
    print(f"{'Horizon':>8} | {'Target Date':>12} | {'Observed':>10} | {'RF':>10} | {'XGBoost':>10} | {'ConvLSTM3D':>12}")
    print("-" * 75)

    for q in HORIZONS:
        target_date = predictions_by_horizon[q]["target_date"]
        observed_ok = "✓" if predictions_by_horizon[q]["observed"] is not None else "✗"
        rf_ok = "✓" if predictions_by_horizon[q].get("RF") is not None else "✗"
        xgb_ok = "✓" if predictions_by_horizon[q].get("XGBoost") is not None else "✗"
        conv_ok = "✓" if predictions_by_horizon[q].get("ConvLSTM3D") is not None else "✗"

        print(f"Q={q:>3}     | {target_date.strftime('%Y-%m'):>12} | {observed_ok:>10} | {rf_ok:>10} | {xgb_ok:>10} | {conv_ok:>12}")

    print("\n" + "=" * 70)
    print("TIFF EXPORT SUMMARY")
    print("=" * 70)
    print(f"Total TIFF files exported: {len(tiff_files)}")
    print(f"TIFF directory: {prediction_dir}")
    print("\nDirectory structure:")
    print(f"  {prediction_dir}/")
    for q in HORIZONS:
        print(f"    Q{q}/")
        print(f"      - observed_SPI_Q{q}_*.tif")
        for model in MODELS:
            if predictions_by_horizon[q].get(model) is not None:
                print(f"      - {model}_prediction_Q{q}_*.tif")

    print("\n" + "=" * 70)
    print("PANEL GENERATION AND TIFF EXPORT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()