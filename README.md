# SPI Forecasting Framework

<div align="center">

**Deep Learning and Machine Learning Framework for Standardized Precipitation Index (SPI) Forecasting**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## Overview

This framework provides a complete pipeline for **SPI (Standardized Precipitation Index)** forecasting using both a **Deep Learning** model (a deliberately simple, attention-free ConvLSTM3D with delta prediction) and **classical Machine Learning** baselines (Random Forest and XGBoost). Unlike climate-driven drought frameworks that use only exogenous variables as input, this framework is **autoregressive**: past SPI and its temporal delta are used, together with precipitation, as model input features to forecast future SPI.

- Forecasting at multiple horizons (Q = 1, 3, 6, 9, 12 instants ahead) and input window sizes (P = 3, 6, 9, 12 months) - one model trained per (P, Q) combination, predicting a single target instant rather than a multi-step sequence (the same windowing convention as this thesis's `drought_forecast_binary`/`drought_forecast_regression` frameworks)
- Random hyperparameter search for every model family, including the ConvLSTM3D (see `config.CONVLSTM3D_SPACE`), so the DL/classical comparison isn't confounded by one side being hand-tuned and the other searched
- Geospatial map visualizations and GeoTIFF exports
- Evaluation via Willmott's Index of Agreement (WI), RMSE, and MAE
- Automatic caching for reproducibility

---

## Project Structure

```
.
├── config.py                  # Central configuration (paths, split dates, hyperparameters)
├── utils_data.py               # Data loading, SPI calculation (Gamma fit), split indices, caching
├── dataset.py                  # PyTorch Dataset for spatiotemporal windows (SPIDataset)
├── data_preparation.py         # Flattens spatiotemporal windows into feature matrices for RF/XGBoost
├── model_convlstm3d.py         # Attention-free ConvLSTM3D architecture with delta prediction
├── model_classic.py            # Random Forest and XGBoost training/evaluation
├── train_model.py              # Training loop (masked loss) + random hyperparameter search for ConvLSTM3D
├── metrics.py                  # Evaluation metrics: WI, RMSE, MAE (mask-aware)
├── plots.py                    # Training-curve plots
├── visualization_spi.py        # Prediction map generation utilities
├── generate_monthly_maps.py    # Monthly prediction maps for the test period
├── generate_panel.py           # Panel figures across models/horizons
├── generate_heatmaps.py        # Per-model WI heatmaps (P x Q) from consolidated results
└── main.py                     # Main experiment runner (grid search over P, Q)
```

---

## Methodology Notes

### SPI computation
SPI is computed per pixel and calendar month via a Gamma distribution fit (method of moments) with zero-inflation handling, **fitted using training-period data only** (`utils_data.calculate_spi_three_periods`) and then applied to the validation and test periods with those same parameters — this avoids leaking future precipitation statistics into the SPI values used for evaluation. A pixel/month combination with fewer than 10 valid training samples is left as `NaN` for that calendar month across the whole series (see **Validity masking** below).

### Train / validation / test split
Splits are date-based (`config.TRAIN_END_YEAR`, `config.REF_DATE`): training ends at `TRAIN_END_YEAR-12-31`, validation runs through `REF_DATE`, and everything after is test. With CHIRPS monthly data (dates on the 1st of each month), this yields no off-by-one loss of data at the boundaries.

### Windowing across split boundaries
`SPIDataset` builds its spatiotemporal cube from the **full** date range and only requires the **Q-step target** to fall strictly inside the requested split; the **P-step input window is allowed to look back into the preceding split** (e.g. a test sample's input may include the last months of validation). This is standard practice for windowed forecasting and introduces no leakage — the target of any sample is always strictly in the future relative to its own input, and gradient updates only ever use `"train"` samples. It also means the number of valid samples per split no longer depends on P, only on `len(period) - Q + 1`; with a short test period, some (P, Q) combinations may still have too few test windows, in which case the pipeline falls back to reporting validation metrics (see `test_source` in the output, below).

### Validity masking
Not every pixel/month has a computable SPI (insufficient calendar-month history, or missing precipitation). Those entries are `NaN` in the SPI series and are zero-filled only for numerical convenience — a target mask tracks which entries are real vs. placeholders:
- **ConvLSTM3D**: trained with a masked Smooth L1 loss (`train_model.masked_smooth_l1_loss`) that excludes placeholder entries from the gradient; validation/test WI, RMSE and MAE (`metrics.py`) accept the same mask.
- **RF / XGBoost**: a (pixel, time) sample is discarded entirely if any of its Q forecast horizons is a placeholder, since scikit-learn estimators cannot exclude individual output entries from a fit.

Without this, invalid entries would be silently scored as SPI = 0 ("near normal"), contaminating both training and reported metrics.

### ConvLSTM3D: a single-instant, attention-free predictor
`model_convlstm3d.py` predicts SPI as `last_observed_SPI + predicted_delta`, where `last_observed_SPI` is read directly from the input window's final time step (the persistence anchor) and `predicted_delta` comes from a single forward pass through the encoder + refinement + head. There is no teacher forcing and no autoregressive rollout: one model is trained per `(P, Q)` combination, and it maps the `P`-step input window directly to the single target instant `Q` steps ahead (`SPIDataset`'s `t0+P+Q-1`) - the same windowing convention as this thesis's `drought_forecast_binary`/`drought_forecast_regression` frameworks. The encoder itself has no channel, spatial, or temporal attention (only the final ConvLSTM hidden state feeds the head): the model is deliberately kept simple so the DL-vs-classical comparison below asks "is a plain, well-tuned ConvLSTM competitive?" rather than "does a more elaborate architecture help?".

### Classical baselines
Random Forest and XGBoost operate on flattened per-pixel time series (`P` past months × 3 channels: precipitation, SPI, ΔSPI) with **no spatial context** (unlike the ConvLSTM3D's convolutional receptive field), and predict the single target instant `Q` steps ahead directly - the same single-target-per-`(P, Q)` strategy as the ConvLSTM3D, so the three model families are no longer compared under different multi-step strategies (see "Known Limitations" below for what was fixed here).

For memory/runtime reasons, RF/XGBoost train on a spatial subsample of pixels (`CLASSIC_PARAMS["sampling_rate"]`, default 10%). This subsample is drawn **once** and reused identically across train, validation and test (`data_preparation.prepare_classic_data`), so the three splits are evaluated over the same spatial support — they are not, however, evaluated over the *same* support as the ConvLSTM3D, which uses the full grid.

---

## Known Limitations / Comparability Caveats

These are disclosed here rather than fixed silently, since they affect how DL-vs-classical comparisons in `EXPERIMENTS/metrics/test_results_all_models.xlsx` should be read:

- **Resolved: different multi-step strategies.** ConvLSTM3D used to be recursive/autoregressive at evaluation time while RF/XGBoost were direct multi-output regressors - a confound when comparing them at large Q. All three model families now train one model per `(P, Q)` and predict a single target instant directly (see "ConvLSTM3D: a single-instant, attention-free predictor" above), so this is no longer a source of difference between them.
- **Unequal hyperparameter search budget (partially resolved).** All three model families are now searched by validation WI (`RandomizedSearchCV` with `wi_scorer` for RF/XGBoost; `train_model.random_search_convlstm3d` for ConvLSTM3D), so the comparison is no longer "tuned vs. untuned". The search budgets are still not equal in kind: RF/XGBoost get up to `CLASSIC_PARAMS["n_iter"]` candidates evaluated via `CLASSIC_PARAMS["cv"]`-fold `TimeSeriesSplit` cross-validation, while ConvLSTM3D gets `CONVLSTM3D_SEARCH_ITER` candidates each trained once against a single train/val split (no k-fold CV, since retraining a ConvLSTM `cv` times per candidate would be too expensive) - keep this asymmetry in mind at the margins.
- **`test_source` field**: for a given (P, Q), if the true test period yields fewer than `MIN_TEST_SAMPLES` valid windows, the pipeline falls back to reporting validation-set metrics labeled as `test_source = "validation_as_test"`. Because the model/hyperparameters were already selected using that same validation data, these rows should not be read as held-out generalization performance — filter on `test_source == "test"` before drawing conclusions from the aggregated results.

---

## Quick Start

```bash
# 1. Set your data path in config.py
# DATA_PATH = "data/pr_Area1.xlsx"
# (regenerated from CHIRPS via notebook/export_CHIRPS.ipynb if you need a
# different study area or date range)

# 2. Run the full experiment grid (all P x Q combinations, ConvLSTM3D + RF + XGBoost)
python main.py
```

`main.py` will, for each (P, Q) combination:
1. Compute/load cached SPI and split indices.
2. Search `CONVLSTM3D_SPACE` and keep the best-by-validation-WI ConvLSTM3D.
3. Prepare the flattened feature matrices and search/train/evaluate RF and XGBoost.
4. Save per-combination artifacts under `EXPERIMENTS/P{P}_Q{Q}/<model>/` (checkpoint or joblib model, metrics, best hyperparameters, visualizations).
5. Aggregate all combinations into `EXPERIMENTS/metrics/test_results_all_models.xlsx` (per-sample metrics, model summary, best-per-model, a WI heatmap over the P/Q grid) and `EXPERIMENTS/metrics/best_params_all_models.xlsx`.

`generate_monthly_maps.py`, `generate_panel.py`, and `generate_heatmaps.py` are standalone scripts (not called by `main.py`) for generating monthly/panel prediction maps and WI heatmaps from already-trained checkpoints and `main.py`'s aggregated `test_results_all_models.xlsx`; `visualization_spi.py` is called automatically by `main.py` after each `(P, Q)` combination trains. The three prediction-map scripts were updated for the single-instant-target rewrite: one forward pass per prediction (no autoregressive rollout), and `generate_monthly_maps.py`'s input window is now built to end exactly `Q` instants before the target date (previously always ended one instant before it, which was only correct for `Q=1`).

---

## Requirements

torch, pandas, numpy, scipy, scikit-learn, xgboost, matplotlib, openpyxl, joblib, rasterio
