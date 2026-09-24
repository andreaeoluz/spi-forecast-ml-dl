# utils_data.py - Data loading, SPI calculation, and caching utilities

import hashlib
import os

import numpy as np
import pandas as pd
from scipy.stats import gamma, norm


# ============================================================================
# DATA LOADING
# ============================================================================

def load_grid_data(path: str) -> pd.DataFrame:
    """
    Load gridded data from Excel file.

    Expected format: MultiIndex (lat, lon) as rows, dates as columns.

    Args:
        path: Path to Excel file

    Returns:
        DataFrame with datetime columns and (lat, lon) MultiIndex
    """
    df = pd.read_excel(path, index_col=[0, 1])
    df.columns = pd.to_datetime(df.columns)

    # Transpose if needed (if dates are in rows)
    if not isinstance(df.index, pd.MultiIndex):
        df = df.T

    return df.astype("float32")


def get_split_indices(dates, train_end_year: int = 2018, ref_date: str = "2024-12") -> tuple:
    """
    Create train/val/test indices based on temporal split.

    - Train: up to train_end_year
    - Validation: between train_end_year and ref_date (inclusive)
    - Test: after ref_date

    Args:
        dates: Array/list of datetime objects
        train_end_year: Last year for training
        ref_date: Reference date string (YYYY-MM)

    Returns:
        tuple: (train_indices, val_indices, test_indices)
    """
    dates_dt = pd.to_datetime(dates)
    ref_date_dt = pd.to_datetime(ref_date)

    train_end = pd.to_datetime(f"{train_end_year}-12-31")
    val_end = pd.to_datetime(f"{ref_date_dt.year}-12-30")
    test_start = pd.to_datetime(f"{ref_date_dt.year + 1}-01-01")

    train_mask = dates_dt <= train_end
    val_mask = (dates_dt > train_end) & (dates_dt <= val_end)
    test_mask = dates_dt >= test_start

    return (
        np.where(train_mask)[0].tolist(),
        np.where(val_mask)[0].tolist(),
        np.where(test_mask)[0].tolist()
    )


# ============================================================================
# SPI CALCULATION
# ============================================================================

def calculate_spi_three_periods(df_pr: pd.DataFrame, scale: int = 3,
                                 train_end_year: int = 2018,
                                 ref_date: str = "2024-12") -> tuple:
    """
    Calculate SPI (Standardized Precipitation Index) using three-period approach.

    Parameters are fitted on training data only, then applied to validation and test.

    Args:
        df_pr: Precipitation dataframe with (lat, lon) index and date columns
        scale: Accumulation window in months
        train_end_year: Last year for training
        ref_date: Reference date for split

    Returns:
        tuple: (df_spi, (train_idx, val_idx, test_idx))
    """
    dates = pd.to_datetime(df_pr.columns)
    train_idx, val_idx, test_idx = get_split_indices(dates, train_end_year, ref_date)
    train_dates = dates[train_idx]

    # Rolling sums for training period
    rolling_train = {}
    for pix_idx, pix in enumerate(df_pr.index):
        series = df_pr.loc[pix].values[:len(train_dates)]
        rolling_train[pix_idx] = pd.Series(series).rolling(window=scale, min_periods=1).sum().values

    # Gamma distribution parameters
    params = {
        'alpha': np.full((len(df_pr.index), 12), np.nan, dtype=np.float32),
        'beta': np.full((len(df_pr.index), 12), np.nan, dtype=np.float32),
        'p_zero': np.full((len(df_pr.index), 12), 0.0, dtype=np.float32)
    }

    # Fit parameters per month and pixel
    for month in range(1, 13):
        month_indices = np.where(train_dates.month == month)[0]
        if len(month_indices) == 0:
            continue

        for pix_idx in range(len(df_pr.index)):
            month_values = [
                rolling_train[pix_idx][di]
                for di in month_indices
                if not np.isnan(rolling_train[pix_idx][di])
            ]

            if len(month_values) < 10:
                continue

            month_values = np.array(month_values)
            non_zero = month_values[month_values > 0]

            if len(non_zero) < 5:
                params['p_zero'][pix_idx, month - 1] = len(month_values[month_values == 0]) / len(month_values)
                continue

            mean, var = non_zero.mean(), non_zero.var()
            alpha = max(0.1, min(mean ** 2 / var, 100)) if var > 0 else 1.0
            beta = max(0.1, min(var / mean, 100)) if var > 0 else mean

            params['alpha'][pix_idx, month - 1] = alpha
            params['beta'][pix_idx, month - 1] = beta
            params['p_zero'][pix_idx, month - 1] = len(month_values[month_values == 0]) / len(month_values)

    # Calculate SPI for all periods
    spi_data = np.full((len(df_pr.index), len(dates)), np.nan, dtype=np.float32)

    for period_name, period_idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        if len(period_idx) == 0:
            continue

        period_dates = dates[period_idx]

        # Rolling sums for this period
        rolling = {}
        for pix_idx in range(len(df_pr.index)):
            series = df_pr.loc[df_pr.index[pix_idx]].values[:period_idx[-1] + 1]
            rolling[pix_idx] = pd.Series(series).rolling(window=scale, min_periods=1).sum().values

        for rel_idx, date in enumerate(period_dates):
            t_idx = period_idx[rel_idx]
            month = date.month

            for pix_idx in range(len(df_pr.index)):
                alpha_val = params['alpha'][pix_idx, month - 1]
                if np.isnan(alpha_val):
                    continue

                precip = rolling[pix_idx][t_idx]
                if np.isnan(precip):
                    continue

                if precip == 0:
                    F_x = params['p_zero'][pix_idx, month - 1]
                else:
                    F_x = params['p_zero'][pix_idx, month - 1] + (
                        1.0 - params['p_zero'][pix_idx, month - 1]
                    ) * gamma.cdf(precip, alpha_val, scale=params['beta'][pix_idx, month - 1])

                F_x = np.clip(F_x, 1e-8, 1 - 1e-8)
                spi_data[pix_idx, t_idx] = norm.ppf(F_x)

    df_spi = pd.DataFrame(spi_data, index=df_pr.index, columns=dates)
    return df_spi.astype(np.float32), (train_idx, val_idx, test_idx)


# ============================================================================
# CACHE FUNCTIONS FOR SPI
# ============================================================================

def _hash_dataframe(df: pd.DataFrame, max_cols: int = 12) -> str:
    """Create lightweight hash of dataframe for cache identification."""
    cols = df.columns[:max_cols]
    data = df[cols].values
    flat = data.ravel()

    # Sample first 1000 elements for speed
    idx = np.linspace(0, len(flat) - 1, min(1000, len(flat))).astype(int)
    sample = flat[idx]

    return hashlib.md5(sample.tobytes()).hexdigest()[:8]


def get_spi_cache_path(df_pr: pd.DataFrame, scale: int, train_end_year: int,
                       ref_date: str, cache_dir: str = "EXPERIMENTS") -> str:
    """Generate unique cache path for SPI calculation."""
    os.makedirs(cache_dir, exist_ok=True)
    pr_hash = _hash_dataframe(df_pr)
    fname = f"spi_cache_s{scale}_te{train_end_year}_ref{ref_date}_pr{pr_hash}.pkl"
    return os.path.join(cache_dir, fname)


def load_or_calculate_spi(df_pr: pd.DataFrame, scale: int = 3,
                          train_end_year: int = 2018, ref_date: str = "2024-12",
                          cache_dir: str = "EXPERIMENTS",
                          force_recompute: bool = False) -> tuple:
    """
    Load SPI from cache or calculate if not exists.

    Args:
        df_pr: Precipitation dataframe
        scale: SPI accumulation scale
        train_end_year: Last year for training
        ref_date: Reference date for split
        cache_dir: Directory for cache files
        force_recompute: If True, ignore cache and recompute

    Returns:
        tuple: (df_spi, indices)
    """
    cache_path = get_spi_cache_path(df_pr, scale, train_end_year, ref_date, cache_dir)

    if os.path.exists(cache_path) and not force_recompute:
        print(f"⚡ Loading SPI from cache: {os.path.basename(cache_path)}")
        cache_data = pd.read_pickle(cache_path)
        return cache_data['df_spi'], cache_data['indices']

    print(f"📊 Calculating SPI (scale={scale})...")
    df_spi, indices = calculate_spi_three_periods(df_pr, scale, train_end_year, ref_date)

    print(f"💾 Saving SPI to cache: {os.path.basename(cache_path)}")
    cache_data = {
        'df_spi': df_spi,
        'indices': indices,
        'metadata': {
            'scale': scale,
            'train_end_year': train_end_year,
            'ref_date': ref_date,
            'df_pr_hash': _hash_dataframe(df_pr)
        }
    }
    pd.to_pickle(cache_data, cache_path)

    return df_spi, indices


def load_multiple_spi_scales(df_pr: pd.DataFrame, scales: list,
                             train_end_year: int = 2018,
                             ref_date: str = "2024-12",
                             cache_dir: str = "EXPERIMENTS",
                             force_recompute: bool = False) -> dict:
    """
    Load or calculate multiple SPI scales efficiently.

    Args:
        df_pr: Precipitation dataframe
        scales: List of SPI scales (e.g., [3, 6, 9, 12])
        train_end_year: Last year for training
        ref_date: Reference date for split
        cache_dir: Directory for cache files
        force_recompute: If True, ignore cache and recompute

    Returns:
        dict: {scale: (df_spi, indices)}
    """
    results = {}
    for scale in scales:
        print(f"\n{'=' * 50}")
        print(f"SPI Scale: {scale}")
        print(f"{'=' * 50}")
        df_spi, indices = load_or_calculate_spi(
            df_pr, scale, train_end_year, ref_date, cache_dir, force_recompute
        )
        results[scale] = (df_spi, indices)
    return results


# ============================================================================
# CACHE FUNCTIONS FOR CLASSICAL MODELS
# ============================================================================

def _hash_array(arr: np.ndarray, max_samples: int = 1000) -> str:
    """Create lightweight hash of numpy array."""
    if arr.size == 0:
        return "empty"

    flat = arr.ravel()
    idx = np.linspace(0, len(flat) - 1, min(max_samples, len(flat))).astype(int)
    sample = flat[idx]

    return hashlib.md5(sample.tobytes()).hexdigest()[:8]


def get_cache_path(cache_dir: str, P: int, Q: int, sampling_rate: float,
                   max_samples: int, spi_scale: int, df_pr: pd.DataFrame,
                   df_spi: pd.DataFrame, version: str = "v2") -> str:
    """Generate unique cache path for classical model data."""
    pr_hash = _hash_dataframe(df_pr)
    spi_hash = _hash_dataframe(df_spi)
    fname = (f"cache_{version}_s{spi_scale}_P{P}_Q{Q}_"
             f"sr{sampling_rate}_ms{max_samples}_pr{pr_hash}_spi{spi_hash}.npz")
    return os.path.join(cache_dir, fname)


def load_or_create_cache(df_pr: pd.DataFrame, df_spi: pd.DataFrame,
                         P: int, Q: int, indices: tuple,
                         sampling_rate: float, max_samples: int,
                         spi_scale: int, cache_dir: str = "cache",
                         version: str = "v2", force_recompute: bool = False) -> tuple:
    """
    Load preprocessed data for classical models from cache, or create if missing.

    Args:
        df_pr: Precipitation dataframe
        df_spi: SPI dataframe
        P: Input sequence length
        Q: Output sequence length
        indices: Train/val/test indices
        sampling_rate: Fraction of pixels to sample
        max_samples: Maximum number of samples
        spi_scale: SPI scale used
        cache_dir: Cache directory
        version: Cache version string
        force_recompute: If True, ignore cache and recompute

    Returns:
        tuple: (X_train, Y_train_seq, X_val, Y_val_seq, X_test, Y_test_seq)
    """
    os.makedirs(cache_dir, exist_ok=True)

    cache_path = get_cache_path(
        cache_dir, P, Q, sampling_rate, max_samples, spi_scale,
        df_pr, df_spi, version
    )

    if os.path.exists(cache_path) and not force_recompute:
        print(f"⚡ Loading cache: {os.path.basename(cache_path)}")
        data = np.load(cache_path)
        return (
            data["X_train"], data["Y_train_seq"],
            data["X_val"], data["Y_val_seq"],
            data["X_test"], data["Y_test_seq"]
        )

    print(f"💾 Creating cache: {os.path.basename(cache_path)}")
    from data_preparation import prepare_classic_data

    X_train, Y_train_seq, X_val, Y_val_seq, X_test, Y_test_seq, _, _ = prepare_classic_data(
        df_pr, df_spi, P, Q, indices,
        sampling_rate=sampling_rate, max_samples=max_samples
    )

    # Ensure float32 for efficiency
    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    X_test = X_test.astype(np.float32)
    Y_train_seq = Y_train_seq.astype(np.float32)
    Y_val_seq = Y_val_seq.astype(np.float32)
    Y_test_seq = Y_test_seq.astype(np.float32)

    np.savez_compressed(
        cache_path,
        X_train=X_train, Y_train_seq=Y_train_seq,
        X_val=X_val, Y_val_seq=Y_val_seq,
        X_test=X_test, Y_test_seq=Y_test_seq
    )

    return X_train, Y_train_seq, X_val, Y_val_seq, X_test, Y_test_seq