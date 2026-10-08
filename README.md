# Satellite-Based SPI Forecasting with Machine and Deep Learning

Code and results for the paper **"Satellite-Based Drought Forecasting Across Multiple Horizons: A Comparative Analysis of Machine and Deep Learning"**.

The framework compares a ConvLSTM3D with Random Forest and XGBoost for forecasting SPI-3 at a target instant `Q` months after a `P`-month input window, in two hydroclimatically distinct areas of Brazil (Amazon and Pantanal), using CHIRPS precipitation.

---

## Related repositories

| Repository | Paper |
|---|---|
| [drought_forecast_binary](https://github.com/andreaeoluz/drought_forecast_binary) | Binary rare-drought forecasting with TL |
| [spi-forecast-continuous](https://github.com/andreaeoluz/spi-forecast-continuous) | Continuous SPI forecasting with TL |
| [drought-datasets](https://github.com/andreaeoluz/drought-datasets) | The two datasets used by the TL frameworks |
| [spi-forecast-ml-dl](https://github.com/andreaeoluz/spi-forecast-ml-dl) (this one) | ML vs. DL comparison for SPI forecasting |

---

## Method in brief

- **Data.** Monthly CHIRPS precipitation (1994–2025) for Area 1 and Area 2, exported per pixel from Google Earth Engine.
- **Inputs.** Precipitation, SPI-3 and ΔSPI over the `P`-month window. SPI is fitted on the training period only.
- **Split.** Training up to 2018, validation 2019–2024, test 2025 (`config.TRAIN_END_YEAR`, `config.REF_DATE`).
- **Models.** ConvLSTM3D (attention-free, last observed SPI + predicted change), Random Forest and XGBoost on per-pixel flattened windows. All three are tuned by random search on validation WI and predict the single target instant directly.
- **Grid.** P ∈ {3, 6, 9, 12} × Q ∈ {1, 3, 6, 9, 12} × 3 models × 2 areas.
- **Metrics.** Willmott's index of agreement (WI), RMSE and MAE, computed over valid pixels only.

---

## Repository structure

```
├── main.py                 # Full experiment grid for one area (ConvLSTM3D + RF + XGBoost, all P x Q)
├── config.py               # Paths, split dates, search spaces and hyperparameters
├── utils_data.py           # Data loading, SPI (Gamma fit on training), split indices, caching
├── dataset.py              # Spatiotemporal windows (SPIDataset)
├── data_preparation.py     # Flattened feature matrices for RF/XGBoost
├── model_convlstm3d.py     # ConvLSTM3D
├── model_classic.py        # Random Forest and XGBoost search, training and evaluation
├── train_model.py          # ConvLSTM3D training (masked loss) and random search
├── metrics.py              # WI, RMSE, MAE (mask-aware)
├── plots.py                # Shared figure style and training curves
├── visualization_spi.py    # Per-run prediction maps and GeoTIFFs (called by main.py)
├── generate_heatmaps.py    # WI heatmaps (P x Q) per model and area
├── generate_panel.py       # Prediction panel for all horizons (observed, RF, XGBoost, ConvLSTM3D)
├── data/
│   ├── export_data.ipynb   # CHIRPS export per area (Google Earth Engine)
│   ├── pr_Area1.xlsx       # Monthly precipitation, Area 1
│   └── pr_Area2.xlsx       # Monthly precipitation, Area 2
└── notebook/
    └── export_CHIRPS.ipynb # CHIRPS extraction for a new area or period
```

---

## Installation and data

```bash
git clone https://github.com/andreaeoluz/spi-forecast-ml-dl.git
cd spi-forecast-ml-dl
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.9+; a CUDA GPU is recommended for the ConvLSTM3D. The precipitation of both areas is included in `data/`; the notebooks regenerate it from CHIRPS (a Google Earth Engine account is required).

---

## Reproducing the paper

The default configuration is the one in the paper (`config.py`). Each area is run separately by setting `DATA_PATH` and `BASE_DIR` in `config.py`:

| Area | `DATA_PATH` | `BASE_DIR` |
|---|---|---|
| Area 1 | `data/pr_Area1.xlsx` | `EXPERIMENTS_AREA1` |
| Area 2 | `data/pr_Area2.xlsx` | `EXPERIMENTS_AREA2` |

```bash
python main.py              # full P x Q grid for the configured area
python generate_panel.py    # prediction panel (P = 12, Q = 1, 3, 6, 9, 12) for the configured area
python generate_heatmaps.py # WI heatmaps for both areas
```

`main.py` saves each (P, Q, model) run under `<BASE_DIR>/P{P}_Q{Q}/<model>/` and aggregates all runs into `<BASE_DIR>/metrics/test_results_all_models.xlsx` and `best_params_all_models.xlsx`. Rows with `test_source = "validation_as_test"` (too few test windows for that (P, Q)) are not held-out results.

---

## Paper figures and tables

| Paper element | Produced by |
|---|---|
| Test WI/RMSE/MAE tables per model, area and (P, Q) | `main.py` → `<BASE_DIR>/metrics/test_results_all_models.xlsx` |
| Selected hyperparameters | `main.py` → `<BASE_DIR>/metrics/best_params_all_models.xlsx` |
| WI heatmaps (`heatmap_wi_Area{1,2}_{ConvLSTM3D,RF,XGBoost}`) | `generate_heatmaps.py` |
| Prediction panels (`Panel_AREA1`, `Panel_AREA2`) | `generate_panel.py` |

---

## Included results

| File | Content |
|---|---|
| `EXPERIMENTS_AREA1/metrics/test_results_all_models.xlsx` | Test metrics for every model and (P, Q), Area 1 |
| `EXPERIMENTS_AREA1/metrics/best_params_all_models.xlsx` | Selected hyperparameters, Area 1 |
| `EXPERIMENTS_AREA2/metrics/test_results_all_models.xlsx` | Test metrics for every model and (P, Q), Area 2 |
| `EXPERIMENTS_AREA2/metrics/best_params_all_models.xlsx` | Selected hyperparameters, Area 2 |

---

## License

MIT — see [LICENSE](LICENSE).
