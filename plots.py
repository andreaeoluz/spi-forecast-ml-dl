# plots.py - Visualization utilities for maps and training curves

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FuncFormatter

# ============================================================================
# CONSTANTS
# ============================================================================

CMAP_SPI = 'RdBu'          # low SPI (drought) -> red, high SPI (wet) -> blue
CMAP_SPI_CLASSES = ListedColormap([
    "#8B0000", "#CD5C5C", "#F4A460", "#FFD700",
    "#ADFF2F", "#32CD32", "#006400"
])
CMAP_ERROR = 'YlOrRd'
CMAP_ACC = 'YlGn'
CMAP_WI = 'RdYlBu_r'       # low WI (poor skill) -> red, high WI (good skill) -> blue

# Consistent per-model identity (color + marker) for line/bar plots comparing
# ConvLSTM3D, RF and XGBoost across figures.
MODEL_COLORS = {
    "ConvLSTM3D": "#1b4f72",
    "RF": "#c0392b",
    "XGBoost": "#1e8449",
}
MODEL_MARKERS = {
    "ConvLSTM3D": "o",
    "RF": "s",
    "XGBoost": "^",
}


# ============================================================================
# STYLE CONFIGURATION
# ============================================================================

def set_journal_style() -> None:
    """Set matplotlib parameters for publication-ready figures."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 300,
    })


# ============================================================================
# MAP PLOTTING
# ============================================================================

def plot_map(data: np.ndarray, lats: np.ndarray, lons: np.ndarray,
             title: str = None, cmap: str = "RdBu",
             vmin: float = None, vmax: float = None,
             cbar_label: str = None, save_path: str = None,
             discrete: bool = False) -> None:
    """
    Plot a 2D spatial map with colorbar.

    Args:
        data: 2D array of shape (H, W)
        lats: Array of latitude values (sorted descending)
        lons: Array of longitude values (sorted ascending)
        title: Plot title (optional)
        cmap: Colormap name
        vmin, vmax: Colorbar limits
        cbar_label: Label for colorbar
        save_path: Path to save the figure (if None, display instead)
        discrete: If True, use nearest interpolation for discrete classes
    """
    set_journal_style()

    # Calculate aspect ratio
    lon_range = lons.max() - lons.min()
    lat_range = lats.max() - lats.min()
    aspect = lon_range / lat_range if lat_range > 0 else 1

    fig_width = 8
    fig_height = max(4, min(8 / aspect, 10))

    fig, (ax, cax) = plt.subplots(
        1, 2, figsize=(fig_width, fig_height),
        gridspec_kw={"width_ratios": [20, 0.8], "wspace": 0.15}
    )

    extent = [lons.min(), lons.max(), lats.min(), lats.max()]

    im = ax.imshow(
        data, cmap=cmap, vmin=vmin, vmax=vmax,
        origin="upper", extent=extent, aspect='equal',
        interpolation='nearest' if discrete else 'bilinear'
    )

    if title:
        ax.set_title(title)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Format tick labels without trailing decimals
    formatter = FuncFormatter(lambda x, _: f"{int(x)}" if abs(x - int(x)) < 1e-6 else f"{x:.2f}".rstrip('0').rstrip('.'))
    ax.set_xticks(np.linspace(lons.min(), lons.max(), 5))
    ax.set_yticks(np.linspace(lats.min(), lats.max(), 5))
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

    # Colorbar
    cbar = fig.colorbar(im, cax=cax)
    if cbar_label:
        cbar.set_label(cbar_label, fontsize=9)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
    else:
        plt.show()


# ============================================================================
# TRAINING CURVES
# ============================================================================

def plot_training_curves(history_loss: list, history_wi: list,
                         history_rmse: list, P: int, Q: int,
                         save_dir: str = "EXPERIMENTS/curves") -> None:
    """
    Plot training loss, validation WI, and validation RMSE.

    Args:
        history_loss: List of loss values per epoch
        history_wi: List of WI values per epoch
        history_rmse: List of RMSE values per epoch
        P: Input sequence length
        Q: Output sequence length
        save_dir: Directory to save the figure
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(8, 10))

    # Loss subplot
    axes[0].plot(history_loss)
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f"Training Curve - P={P}, Q={Q}")

    # WI subplot
    axes[1].plot(history_wi)
    axes[1].set_ylabel("Validation WI")
    axes[1].grid(True, alpha=0.3)

    # RMSE subplot
    axes[2].plot(history_rmse)
    axes[2].set_ylabel("Validation RMSE")
    axes[2].set_xlabel("Epoch")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"training_curve_P{P}_Q{Q}.pdf")
    plt.savefig(save_path, dpi=300)
    plt.close()


# ============================================================================
# GEOTIFF EXPORT
# ============================================================================

def save_geotiff(array: np.ndarray, lats: np.ndarray, lons: np.ndarray,
                 out_path: str) -> None:
    """
    Save a 2D array as a GeoTIFF file.

    Args:
        array: 2D array of shape (H, W)
        lats: Array of latitude values (sorted descending)
        lons: Array of longitude values (sorted ascending)
        out_path: Output file path (must end with .tif)
    """
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import from_origin

    H, W = array.shape
    res_x = abs(lons[1] - lons[0])
    res_y = abs(lats[1] - lats[0])
    transform = from_origin(lons.min(), lats.max(), res_x, res_y)

    with rasterio.open(
        out_path, "w", driver="GTiff",
        height=H, width=W, count=1, dtype="float32",
        transform=transform, crs=CRS.from_epsg(4326)
    ) as dst:
        dst.write(array.astype(np.float32), 1)