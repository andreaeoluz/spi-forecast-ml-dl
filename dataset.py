# dataset.py - PyTorch Dataset for SPI forecasting with spatiotemporal inputs

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SPIDataset(Dataset):
    """
    Dataset for SPI drought forecasting.

    Input features for each sample:
        - Precipitation (pr)
        - Current SPI value (spi)
        - SPI temporal delta (dspi)

    Output:
        - A single SPI value, Q instants after the end of the P-step input
          window (t0+P+Q-1) - the same single-target-per-horizon convention
          used by drought_forecast_binary/regression's ClimateDataset,
          rather than a Q-length sequence of consecutive future months. One
          model is trained per (P, Q) combination.
    """

    def __init__(self, df_pr: pd.DataFrame, df_spi: pd.DataFrame,
                 P: int, Q: int, period: str, indices: tuple):
        """
        Args:
            df_pr: Precipitation dataframe with (lat, lon) index and date columns
            df_spi: SPI dataframe with (lat, lon) index and date columns
            P: Number of past time steps to use as input
            Q: Forecast horizon, in instants after the end of the input
                window (Q=1 -> the instant right after the window)
            period: "train", "val", or "test"
            indices: Tuple of (train_idx, val_idx, test_idx) column indices

        Note on windowing: the P-step input window is allowed to look back
        past the start of `period` (e.g. a test sample's input may include
        the last months of the validation period). Only the target instant
        is required to fall strictly inside `period`. This avoids discarding
        samples near a split boundary while introducing no leakage: the
        target of any sample is always strictly in the future relative to
        its own input window, and gradient updates only ever use "train"
        samples.
        """
        train_idx, val_idx, test_idx = indices
        period_idx = {"train": train_idx, "val": val_idx, "test": test_idx}[period]

        if len(period_idx) > 1 and period_idx[-1] - period_idx[0] + 1 != len(period_idx):
            raise ValueError(
                f"SPIDataset expects a contiguous '{period}' period (no gaps in the monthly series)"
            )

        # Cubes span the full date range so that val/test inputs can reach
        # back into the preceding period; only `self.starts` restricts which
        # windows belong to this split.
        self.pr = self._df_to_cube(df_pr)
        self.spi = self._df_to_cube(df_spi)
        self.P = P
        self.Q = Q

        if len(period_idx) == 0:
            self.starts = []
        else:
            lo, hi = period_idx[0], period_idx[-1]
            first_t0 = lo - P - Q + 1      # target (t0+P+Q-1) falls exactly at `lo`
            last_t0 = hi - P - Q + 1       # target (t0+P+Q-1) falls exactly at `hi`
            self.starts = [t0 for t0 in range(first_t0, last_t0 + 1) if t0 >= 0]

    def _df_to_cube(self, df: pd.DataFrame) -> np.ndarray:
        """Convert (lat, lon) indexed dataframe to 3D cube (T, H, W)."""
        lats = sorted(df.index.get_level_values(0).unique(), reverse=True)
        lons = sorted(df.index.get_level_values(1).unique())
        T, H, W = len(df.columns), len(lats), len(lons)

        lat_pos = {v: i for i, v in enumerate(lats)}
        lon_pos = {v: j for j, v in enumerate(lons)}

        cube = np.full((T, H, W), np.nan, dtype=np.float32)
        for t, col in enumerate(df.columns):
            for (lat, lon), val in df[col].items():
                cube[t, lat_pos[lat], lon_pos[lon]] = val
        return cube

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> tuple:
        """Return (input_tensor, target_tensor, target_mask_tensor).

        `target_mask` is 1.0 where the target pixel/month has a real SPI
        value and 0.0 where SPI could not be computed (insufficient
        calendar-month history at that pixel, or missing precipitation).
        Those invalid entries are zero-filled in `target_tensor` only so the
        model receives a finite tensor; the mask must be used to exclude
        them from the loss and from evaluation metrics, since 0.0 is a valid
        SPI value ("near normal") and must not be confused with "unknown".
        """
        t0 = self.starts[idx]

        # Past precipitation
        x_pr = self.pr[t0:t0 + self.P]

        # Past SPI
        x_spi = self.spi[t0:t0 + self.P]

        # SPI delta (temporal difference)
        x_dspi = np.zeros_like(x_spi)
        if self.P > 1:
            x_dspi[1:] = x_spi[1:] - x_spi[:-1]

        # Stack channels: (P, C=3, H, W)
        x = np.stack([x_pr, x_spi, x_dspi], axis=1)

        # Target: single SPI instant, Q steps after the window (H, W)
        target_idx = t0 + self.P + self.Q - 1
        y = self.spi[target_idx]
        y_mask = ~np.isnan(y)

        # Replace NaN with zero (input imputation; target validity is
        # tracked separately via y_mask, see docstring above)
        x_tensor = torch.tensor(np.nan_to_num(x, nan=0.0), dtype=torch.float32)
        y_tensor = torch.tensor(np.nan_to_num(y, nan=0.0), dtype=torch.float32)
        mask_tensor = torch.tensor(y_mask, dtype=torch.float32)

        return x_tensor, y_tensor, mask_tensor