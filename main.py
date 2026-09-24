# main.py - Main execution script for SPI forecasting experiments

# These two side-effecting calls must run before matplotlib.pyplot or torch
# are imported anywhere else in the process (directly or transitively), so
# they are deliberately kept ahead of the normal stdlib/third-party import
# block below instead of being grouped alphabetically with it.
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for servers

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import dump
from torch.utils.data import DataLoader

from config import (
    BASE_DIR, METRICS_DIR, DATA_PATH, TRAIN_END_YEAR,
    SPI_SCALE_FIXED, P_VALUES, Q_VALUES,
    CLASSIC_PARAMS, MIN_TEST_SAMPLES, USE_VAL_AS_TEST_FALLBACK,
    DEVICE, RANDOM_SEED, GENERATE_VISUALIZATIONS
)
from utils_data import load_grid_data, load_or_calculate_spi
from dataset import SPIDataset
from data_preparation import prepare_classic_data
from model_classic import run_classic, evaluate_with_fallback
from train_model import random_search_convlstm3d
from metrics import compute_all_metrics
from visualization_spi import generate_visualizations


# ============================================================================
# INITIALIZATION & HELPERS
# ============================================================================

def setup_environment() -> None:
    """Configure environment, seed, and GPU settings for reproducibility."""
    # GPU memory cleanup
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB\n")

    # Set random seeds for reproducibility
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_test_dataset(ds_test, ds_val, P: int, Q: int):
    """
    Determine which dataset to use for testing based on sample availability.

    Args:
        ds_test: Test dataset
        ds_val: Validation dataset (fallback)
        P: Input sequence length
        Q: Output sequence length

    Returns:
        tuple: (dataset, source_name, n_samples) or (None, "none", 0)
    """
    # len(ds_test) already counts valid (P, Q) windows for this split
    # (SPIDataset lets the input look back across the split boundary), so no
    # further subtraction of (P + Q) is needed here.
    n_test = len(ds_test)

    if n_test >= MIN_TEST_SAMPLES:
        return ds_test, "test", n_test
    elif USE_VAL_AS_TEST_FALLBACK and len(ds_val) >= MIN_TEST_SAMPLES:
        return ds_val, "validation_as_test", len(ds_val)

    return None, "none", 0


def evaluate_dl_on_test(model, ds_test, ds_val, Q: int, device, P: int) -> dict:
    """
    Evaluate the ConvLSTM3D model on the test set (or fallback to
    validation) with a single forward pass per sample - there is no
    autoregressive rollout, since each model predicts one target instant,
    Q steps ahead of its input window (see SPIDataset).

    Args:
        model: Trained ConvLSTM3D model
        ds_test: Test dataset
        ds_val: Validation dataset (fallback)
        Q: Forecast horizon, in instants ahead (for metadata)
        device: Torch device
        P: Input sequence length (for fallback logic)

    Returns:
        dict: Evaluation metrics
    """
    ds_use, source, n_samples = get_test_dataset(ds_test, ds_val, P, Q)

    if ds_use is None:
        return {
            "wi": np.nan, "rmse": np.nan, "mae": np.nan,
            "test_source": "none", "n_test_samples": 0
        }

    loader = DataLoader(ds_use, batch_size=4, shuffle=False)
    model.eval()

    yt_all, yp_all, mask_all = [], [], []

    with torch.no_grad():
        for x, y, y_mask in loader:
            x = x.to(device)
            y = y.to(device)
            y_mask = y_mask.to(device)

            y_pred = model(x)

            yt_all.append(y.reshape(-1))
            yp_all.append(y_pred.reshape(-1))
            mask_all.append(y_mask.reshape(-1))

    yt_all = torch.cat(yt_all, dim=0)
    yp_all = torch.cat(yp_all, dim=0)
    mask_all = torch.cat(mask_all, dim=0)

    metrics = compute_all_metrics(yt_all, yp_all, mask=mask_all)
    metrics["test_source"] = source
    metrics["n_test_samples"] = n_samples

    return metrics


def save_results_summary(df_test: pd.DataFrame) -> tuple:
    """
    Generate summary tables from test results.

    Args:
        df_test: DataFrame with test metrics

    Returns:
        tuple: (summary_by_model, best_per_model, wi_heatmap)
    """
    # Summary statistics per model
    summary_by_model = df_test.groupby("model").agg({
        "wi": ["mean", "std", "min", "max", "count"],
        "rmse": ["mean", "std", "min", "max"],
        "mae": ["mean", "std", "min", "max"],
    }).round(6)

    summary_by_model.columns = ['_'.join(col).strip() for col in summary_by_model.columns.values]
    summary_by_model = summary_by_model.reset_index()

    # Best configuration per model
    best_per_model = []
    for model_name in df_test["model"].unique():
        df_model = df_test[df_test["model"] == model_name]
        best_idx = df_model["wi"].idxmax()
        best_per_model.append(df_model.loc[best_idx])

    df_best_per_model = pd.DataFrame(best_per_model)

    # Heatmap of best WI per (P, Q) combination
    wi_heatmap = df_test.pivot_table(
        index="P", columns="Q", values="wi", aggfunc="max"
    ).round(4)

    return summary_by_model, df_best_per_model, wi_heatmap


def print_experiment_header() -> None:
    """Print experiment configuration header."""
    print("=" * 70)
    print("SPI FORECASTING FRAMEWORK")
    print(f"SPI scale: {SPI_SCALE_FIXED}")
    print(f"Device: {DEVICE}")
    print(f"P values: {P_VALUES}")
    print(f"Q values: {Q_VALUES}")
    print(f"Total combinations: {len(P_VALUES) * len(Q_VALUES)}")
    print("=" * 70)


def print_data_split_info(indices: tuple, dates) -> None:
    """Print data split information."""
    train_idx, val_idx, test_idx = indices
    train_start_year = dates[train_idx[0]].year if len(train_idx) > 0 else "?"
    test_start_date = dates[test_idx[0]] if len(test_idx) > 0 else None
    test_end_date = dates[test_idx[-1]] if len(test_idx) > 0 else None

    print("\nPeriod sizes:")
    print(f"  Training   : {len(train_idx)} months ({train_start_year}-{TRAIN_END_YEAR})")

    if test_start_date:
        print(f"  Validation : {len(val_idx)} months ({TRAIN_END_YEAR + 1} to {test_start_date.year - 1}-12)")
        print(f"  Test       : {len(test_idx)} months ({test_start_date.strftime('%Y-%m')} to {test_end_date.strftime('%Y-%m')})")
    else:
        print(f"  Validation : {len(val_idx)} months")
        print(f"  Test       : {len(test_idx)} months")

    print("=" * 70)


def save_classic_model_artifacts(res: dict, combo_dir: Path, model_name: str,
                                  test_metrics: dict, best_params_list: list) -> None:
    """Save classical model artifacts (model, params, metrics, visualizations)."""
    classic_dir = combo_dir / model_name
    classic_dir.mkdir(exist_ok=True)

    # Save model
    dump(res["model"], classic_dir / "best_model.joblib")

    # Save best parameters
    if res["best_params"]:
        with open(classic_dir / "best_params.json", "w") as f:
            json.dump(res["best_params"], f, indent=4)
        best_params_list.append({
            "model": model_name,
            "P": res["P"],
            "Q": res["Q"],
            "params": json.dumps(res["best_params"])
        })

    # Save metrics
    metrics_data = {
        "metric": ["wi", "rmse", "mae", "wi_val", "test_source", "n_test_samples"],
        "value": [
            test_metrics["wi"],
            test_metrics["rmse"],
            test_metrics["mae"],
            res["val_metrics"]["wi"],
            test_metrics.get("test_source", "unknown"),
            test_metrics.get("n_test_samples", 0)
        ]
    }
    pd.DataFrame(metrics_data).to_excel(classic_dir / "metrics.xlsx", index=False)

    # Save cross-validation results
    if res.get("cv_results"):
        cv_df = pd.DataFrame(res["cv_results"])
        cv_df.to_excel(classic_dir / "cv_results.xlsx", index=False)


def save_dl_model_artifacts(model, best_params: dict, combo_dir: Path,
                             test_metrics: dict, P: int, Q: int) -> None:
    """Save deep learning model artifacts (best candidate from the
    hyperparameter search - see train_model.random_search_convlstm3d)."""
    conv_dir = combo_dir / "ConvLSTM3D"
    conv_dir.mkdir(exist_ok=True)

    # Save model checkpoint
    torch.save({
        "model_state_dict": model.state_dict(),
        "P": P,
        "Q": Q,
        "best_wi_val": model.best_wi,
        "test_wi": test_metrics["wi"],
        "hidden": best_params["hidden"],
        "dropout": best_params["dropout"],
        "lr": best_params["lr"],
        "batch_size": best_params["batch_size"],
    }, conv_dir / "best_model.pt")

    # Save best hyperparameters (search over CONVLSTM3D_SPACE)
    with open(conv_dir / "best_params.json", "w") as f:
        json.dump(best_params, f, indent=4)

    # Save metrics
    metrics_data = {
        "metric": ["wi", "rmse", "mae", "wi_val", "test_source", "n_test_samples"],
        "value": [
            test_metrics["wi"],
            test_metrics["rmse"],
            test_metrics["mae"],
            model.best_wi,
            test_metrics.get("test_source", "unknown"),
            test_metrics.get("n_test_samples", 0)
        ]
    }
    pd.DataFrame(metrics_data).to_excel(conv_dir / "metrics.xlsx", index=False)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main() -> None:
    """Main execution function."""
    setup_environment()
    print_experiment_header()

    # ========== DATA LOADING ==========
    print("\n[1] Loading precipitation data...")
    df_pr = load_grid_data(DATA_PATH)

    print("\n[2] Calculating SPI...")
    df_spi, indices = load_or_calculate_spi(df_pr, SPI_SCALE_FIXED, TRAIN_END_YEAR)
    train_idx, val_idx, test_idx = indices

    dates = pd.to_datetime(df_spi.columns)
    print_data_split_info(indices, dates)

    # ========== STORAGE ==========
    results_val = []
    results_test = []
    best_params_list = []

    total_combos = len(P_VALUES) * len(Q_VALUES)
    combo_count = 0

    # ========== EXPERIMENT LOOP ==========
    for P in P_VALUES:
        for Q in Q_VALUES:
            combo_count += 1
            print(f"\n{'=' * 70}")
            print(f"[{combo_count}/{total_combos}] P = {P}, Q = {Q}")
            print(f"{'=' * 70}")

            # Create datasets
            ds_train = SPIDataset(df_pr, df_spi, P, Q, "train", indices)
            ds_val = SPIDataset(df_pr, df_spi, P, Q, "val", indices)
            ds_test_raw = SPIDataset(df_pr, df_spi, P, Q, "test", indices)

            print(f"Train samples: {len(ds_train)}")
            print(f"Validation samples: {len(ds_val)}")
            print(f"Test raw samples: {len(ds_test_raw)}")

            # Validate test data availability
            ds_test, test_source, n_test = get_test_dataset(ds_test_raw, ds_val, P, Q)
            if ds_test is None:
                print(f"⚠ Warning: insufficient test data (needs P+Q={P + Q})")
                continue

            print(f"Test source: {test_source} ({n_test} samples)")

            # Create combination directory
            combo_dir = BASE_DIR / f"P{P}_Q{Q}"
            combo_dir.mkdir(exist_ok=True)

            # ==================== CONVLSTM3D ====================
            # Hyperparameter search over CONVLSTM3D_SPACE (config.py),
            # analogous to RF/XGBoost's RandomizedSearchCV below - see
            # train_model.random_search_convlstm3d for why it can't reuse
            # RandomizedSearchCV directly (no k-fold CV, single train/val
            # split per candidate).
            print("\n[ConvLSTM3D]")
            dl_res = random_search_convlstm3d(ds_train, ds_val, P, Q, device=DEVICE)
            model = dl_res["model"]

            if model is None:
                print("  No training data available")
            else:
                test_metrics = evaluate_dl_on_test(model, ds_test_raw, ds_val, Q, DEVICE, P)
                print(f"Test WI = {test_metrics['wi']:.4f}, RMSE = {test_metrics['rmse']:.4f}")

                # Store results
                results_test.append({"model": "ConvLSTM3D", "P": P, "Q": Q, **test_metrics})
                results_val.append({"model": "ConvLSTM3D", "P": P, "Q": Q, "wi_val": model.best_wi})
                best_params_list.append({
                    "model": "ConvLSTM3D", "P": P, "Q": Q,
                    "params": json.dumps(dl_res["best_params"])
                })

                # Generate visualizations
                if GENERATE_VISUALIZATIONS:
                    vis_period = "test" if test_source == "test" else "val"
                    if vis_period == "val":
                        print("  Note: Using validation period for visualizations (fallback)")

                    vis_dir = combo_dir / "ConvLSTM3D" / "visualizations"
                    vis_dir.mkdir(parents=True, exist_ok=True)

                    generate_visualizations(
                        model, df_pr, df_spi, P, Q, indices, DEVICE,
                        "ConvLSTM3D", str(vis_dir), vis_period, "lstm"
                    )

                save_dl_model_artifacts(model, dl_res["best_params"], combo_dir, test_metrics, P, Q)

            # ==================== CLASSICAL MODELS ====================
            print("\n[Classical models] Preparing data...")

            X_train, Y_train, X_val, Y_val, X_test, Y_test, H, W = prepare_classic_data(
                df_pr, df_spi, P, Q, indices,
                sampling_rate=CLASSIC_PARAMS["sampling_rate"],
                max_samples=CLASSIC_PARAMS["max_samples"]
            )

            print(f"Classical data shapes: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

            for model_name in ["RF", "XGBoost"]:
                print(f"\n[{model_name}]")

                if len(X_train) == 0:
                    print("  No training data available")
                    continue

                # Train model
                res = run_classic(
                    model_name, X_train, Y_train, X_val, Y_val, P, Q,
                    n_iter=CLASSIC_PARAMS["n_iter"],
                    cv=CLASSIC_PARAMS["cv"]
                )

                if res["model"] is None:
                    print("  Training failed")
                    continue

                # Evaluate
                test_metrics = evaluate_with_fallback(
                    res["model"], X_test, Y_test, X_val, Y_val,
                    min_samples=MIN_TEST_SAMPLES
                )

                print(f"  Test WI = {test_metrics['wi']:.4f}, RMSE = {test_metrics['rmse']:.4f}")
                print(f"  Validation WI = {res['val_metrics']['wi']:.4f}")
                print(f"  Test source: {test_metrics['test_source']} ({test_metrics['n_test_samples']} samples)")

                # Store results
                results_test.append({"model": model_name, "P": P, "Q": Q, **test_metrics})
                results_val.append({"model": model_name, "P": P, "Q": Q, "wi_val": res["val_metrics"]["wi"]})

                # Save artifacts
                save_classic_model_artifacts(res, combo_dir, model_name, test_metrics, best_params_list)

                # Generate visualizations
                if GENERATE_VISUALIZATIONS:
                    vis_period = test_metrics.get("test_source", "unknown")
                    if vis_period == "validation_as_test":
                        vis_period = "val"
                        print("  Note: Using validation period for visualizations (fallback)")
                    elif vis_period != "test":
                        vis_period = "val"

                    vis_dir = combo_dir / model_name / "visualizations"
                    vis_dir.mkdir(parents=True, exist_ok=True)

                    generate_visualizations(
                        res["model"], df_pr, df_spi, P, Q, indices, DEVICE,
                        model_name, str(vis_dir), vis_period, "classic"
                    )

    # ========== SAVE AGGREGATED RESULTS ==========
    print("\n" + "=" * 70)
    print("SAVING AGGREGATED RESULTS")
    print("=" * 70)

    df_test = pd.DataFrame(results_test)

    if not df_test.empty:
        # Generate summaries
        summary_by_model, best_per_model, wi_heatmap = save_results_summary(df_test)

        # Save to Excel
        excel_path = METRICS_DIR / "test_results_all_models.xlsx"
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            df_test.to_excel(writer, sheet_name="Test_Metrics", index=False)
            summary_by_model.to_excel(writer, sheet_name="Model_Summary", index=False)
            best_per_model.to_excel(writer, sheet_name="Best_Per_Model", index=False)
            wi_heatmap.to_excel(writer, sheet_name="WI_Heatmap", index=False)

        if best_params_list:
            pd.DataFrame(best_params_list).to_excel(METRICS_DIR / "best_params_all_models.xlsx", index=False)

        print(f"Results saved to: {excel_path}")

        # ========== PRINT BEST RESULTS ==========
        print("\n" + "=" * 70)
        print("BEST RESULTS SUMMARY")
        print("=" * 70)

        if df_test['wi'].notna().any():
            best_idx = df_test['wi'].idxmax()
            best = df_test.loc[best_idx]
            print("\nBest overall configuration:")
            print(f"  Model: {best['model']}")
            print(f"  P = {int(best['P'])}, Q = {int(best['Q'])}")
            print(f"  WI = {best['wi']:.4f}, RMSE = {best['rmse']:.4f}, MAE = {best['mae']:.4f}")
            print(f"  Test source: {best['test_source']}")

        print("\nBest per model:")
        for model_name in ["ConvLSTM3D", "RF", "XGBoost"]:
            df_m = df_test[df_test['model'] == model_name]
            if not df_m.empty and df_m['wi'].notna().any():
                best_m = df_m.loc[df_m['wi'].idxmax()]
                print(f"  {model_name:12s}: WI = {best_m['wi']:.4f} | P = {int(best_m['P'])}, Q = {int(best_m['Q'])}")

    print("\n" + "=" * 70)
    print(f"All results saved to: {METRICS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()