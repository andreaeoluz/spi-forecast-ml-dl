# data_preparation.py - Data loading utilities for classical ML models

import random
import numpy as np
from dataset import SPIDataset


def create_datasets(df_pr, df_spi, P, Q, indices):
    """
    Create train, validation, and test SPIDataset instances.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    return (
        SPIDataset(df_pr, df_spi, P, Q, "train", indices),
        SPIDataset(df_pr, df_spi, P, Q, "val", indices),
        SPIDataset(df_pr, df_spi, P, Q, "test", indices),
    )


def prepare_classic_data(df_pr, df_spi, P, Q, indices,
                         sampling_rate=0.1, max_samples=10000,
                         random_seed=123):
    """
    Convert spatiotemporal datasets to flat feature matrices for classical ML.

    Spatial sampling is applied to keep memory usage feasible.
    Each sample is a (pixel, time) pair flattened into a feature vector.

    Returns:
        tuple: (X_train, Y_train, X_val, Y_val, X_test, Y_test, H, W)
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    ds_train = SPIDataset(df_pr, df_spi, P, Q, "train", indices)
    ds_val = SPIDataset(df_pr, df_spi, P, Q, "val", indices)
    ds_test = SPIDataset(df_pr, df_spi, P, Q, "test", indices)

    # Infer spatial dimensions (H, W) from the first available dataset
    H, W = None, None
    for ds in [ds_train, ds_val, ds_test]:
        if len(ds) > 0:
            sample_x, _, _ = ds[0]
            _, _, H, W = sample_x.shape  # (P, C, H, W)
            break

    if H is None or W is None:
        # Fallback: infer from dataframe indices
        lats = sorted(df_spi.index.get_level_values(0).unique(), reverse=True)
        lons = sorted(df_spi.index.get_level_values(1).unique())
        H, W = len(lats), len(lons)

    # Sample pixels ONCE and reuse the same set for train/val/test. Previously
    # each split drew its own independent random subset (and even its own
    # subset *size*, capped by that split's max_samp // len(ds)), so the
    # classical models were trained, validated and tested on different,
    # inconsistent spatial supports -- and on a different support than the
    # ConvLSTM3D, which uses the full grid. A single shared `fixed_pixels`
    # set makes the three splits spatially consistent with each other.
    total_pixels = H * W
    n_pixels_sample = max(1, min(total_pixels, int(total_pixels * sampling_rate)))
    all_pixels = [(i, j) for i in range(H) for j in range(W)]
    fixed_pixels = random.sample(all_pixels, n_pixels_sample)

    def extract_fast(ds, max_samp, fixed_pixels):
        """Extract feature matrix from dataset using a fixed set of pixels."""
        if len(ds) == 0:
            return np.empty((0, 3 * P), dtype=np.float32), np.empty((0,), dtype=np.float32)

        max_possible = len(ds) * len(fixed_pixels)
        X = np.zeros((max_possible, 3 * P), dtype=np.float32)
        Y = np.zeros((max_possible,), dtype=np.float32)

        idx = 0
        for t in range(len(ds)):
            x, y, y_mask = ds[t]
            x_np = x.numpy() if hasattr(x, 'numpy') else x
            y_np = y.numpy() if hasattr(y, 'numpy') else y
            mask_np = y_mask.numpy() if hasattr(y_mask, 'numpy') else y_mask

            for i, j in fixed_pixels:
                features = x_np[:, :, i, j].reshape(-1)  # (P * C)
                target = y_np[i, j]                      # scalar
                target_valid = mask_np[i, j]             # bool

                # Accept sample only if at least half of the input features
                # are non-missing AND the target has a real SPI value.
                # Unlike the ConvLSTM3D loss, sklearn regressors cannot
                # exclude individual invalid entries from the fit, so a
                # sample with a zero-filled (unknown) target is dropped
                # entirely rather than trained on a fabricated 0.
                if np.isnan(features).sum() <= (3 * P) // 2 and target_valid:
                    X[idx] = np.nan_to_num(features, nan=0.0)
                    Y[idx] = target
                    idx += 1

                if idx >= max_samp:
                    break
            if idx >= max_samp:
                break

        return X[:idx], Y[:idx]

    X_train, Y_train = extract_fast(ds_train, max_samples, fixed_pixels)
    X_val, Y_val = extract_fast(ds_val, max_samples, fixed_pixels)
    X_test, Y_test = extract_fast(ds_test, max_samples, fixed_pixels)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, H, W