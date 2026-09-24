# generate_heatmaps.py - Generate WI heatmaps for each model (ConvLSTM3D, RF, XGBoost)

"""
Generates one WI heatmap (P x Q) per model per area, all sharing the same
color scale (WI in [0, 1]) so that colors are directly comparable across the
six figures, and the same font/size style as the rest of the project's
figures (plots.set_journal_style).

Reads from each area's consolidated "test_results_all_models.xlsx" (written
by main.py) and writes PDFs directly into Paper.Felipe/figs/area{1,2}/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

from plots import CMAP_WI, set_journal_style

REPO_ROOT = Path(__file__).parent

MODELS = ["ConvLSTM3D", "RF", "XGBoost"]

# Shared color scale across all six heatmaps.
WI_VMIN, WI_VMAX = 0.0, 1.0
WI_LEVELS = np.linspace(WI_VMIN, WI_VMAX, 21)


def load_data(path: Path) -> pd.DataFrame:
    """Load consolidated results from Excel, without filtering."""
    df = pd.read_excel(path)
    df["P"] = df["P"].astype(int)
    df["Q"] = df["Q"].astype(int)
    return df


def generate_heatmap(df_model: pd.DataFrame, model_name: str, area_name: str,
                     output_dir: Path) -> None:
    """Generate a single WI heatmap on the shared [0, 1] color scale."""
    if df_model.empty:
        print(f"  (skip) no data for {model_name} - {area_name}")
        return

    set_journal_style()

    X = df_model["Q"].values
    Y = df_model["P"].values
    Z = df_model["wi"].values

    xi = np.linspace(min(X), max(X), 200)
    yi = np.linspace(min(Y), max(Y), 200)
    zi = griddata((X, Y), Z, (xi[None, :], yi[:, None]), method="cubic")
    zi = np.clip(zi, WI_VMIN, WI_VMAX)

    fig, ax = plt.subplots(figsize=(4.2, 3.4))

    cntr = ax.contourf(xi, yi, zi, levels=WI_LEVELS, cmap=CMAP_WI,
                       vmin=WI_VMIN, vmax=WI_VMAX)

    ax.set_xlabel("Q")
    ax.set_ylabel("P")
    ax.set_xticks(sorted(df_model["Q"].unique()))
    ax.set_yticks(sorted(df_model["P"].unique()))
    ax.invert_yaxis()
    ax.grid(True, linestyle=":", alpha=0.3)

    cb = fig.colorbar(cntr, ax=ax, format="%.2f", ticks=np.linspace(WI_VMIN, WI_VMAX, 6))
    cb.set_label("WI")

    plt.tight_layout()

    out_path = output_dir / f"heatmap_wi_{area_name}_{model_name}.pdf"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")


def main() -> None:
    results = {
        "Area1": (REPO_ROOT / "EXPERIMENTS_AREA1" / "metrics" / "test_results_all_models.xlsx",
                  REPO_ROOT / "Paper.Felipe" / "figs" / "area1"),
        "Area2": (REPO_ROOT / "EXPERIMENTS_AREA2" / "metrics" / "test_results_all_models.xlsx",
                  REPO_ROOT / "Paper.Felipe" / "figs" / "area2"),
    }

    for area_name, (in_path, out_dir) in results.items():
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{area_name}")
        df = load_data(in_path)
        for model_name in MODELS:
            df_model = df[df["model"] == model_name].copy()
            generate_heatmap(df_model, model_name, area_name, out_dir)


if __name__ == "__main__":
    main()
