# model_classic.py - Classical ML models for drought forecasting (RF and XGBoost)

import warnings

import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor

from config import CLASSIC_PARAMS, RANDOM_SEED, RF_SPACE, XGB_SPACE
from metrics import mae, rmse, wi

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', message='A column-vector y was passed')


def _wi_numpy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Willmott's Index of Agreement, as a plain numpy function (not torch), so
    it can be wrapped as a scikit-learn scorer without per-call tensor
    conversion overhead. Matches how the final WI reported for model
    comparison is computed (metrics.wi over the flattened prediction set).
    """
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[valid], y_pred[valid]

    if y_true.size == 0:
        return np.nan

    ybar = y_true.mean()
    sse = np.sum((y_true - y_pred) ** 2)
    denom = np.sum((np.abs(y_pred - ybar) + np.abs(y_true - ybar)) ** 2)

    if denom == 0:
        return np.nan

    return float(1.0 - sse / denom)


# Used as the RandomizedSearchCV scoring function for both RF and XGBoost so
# that hyperparameter selection optimizes the same metric (WI) used to
# compare all models in the final results table, instead of RMSE/MSE.
wi_scorer = make_scorer(_wi_numpy, greater_is_better=True)


def randomized_search(model_name: str, X_train: np.ndarray, y_train: np.ndarray,
                      n_iter: int = None, cv: int = None):
    """
    Perform randomized hyperparameter search with time series cross-validation.

    Args:
        model_name: "RF" or "XGBoost"
        X_train: Training features (n_samples, n_features)
        y_train: Training targets (n_samples, n_outputs)
        n_iter: Number of parameter combinations to try
        cv: Number of cross-validation folds

    Returns:
        tuple: (best_estimator, best_params, cv_results)
    """
    n_iter = n_iter or CLASSIC_PARAMS["n_iter"]
    cv = cv or CLASSIC_PARAMS["cv"]

    if len(X_train) == 0:
        raise ValueError("X_train is empty. Cannot run randomized_search.")

    # Adjust CV splits when data is scarce
    n_splits = max(2, min(cv, len(X_train) - 1))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    if model_name == "RF":
        model = RandomForestRegressor(
            n_jobs=-1,
            random_state=RANDOM_SEED
        )
        param_dist = RF_SPACE

        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=min(n_iter, 50),
            cv=tscv,
            scoring=wi_scorer,
            n_jobs=1,
            random_state=RANDOM_SEED,
            verbose=0,
            error_score="raise",
            refit=True
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            search.fit(X_train, y_train)

        return search.best_estimator_, search.best_params_, search.cv_results_

    elif model_name == "XGBoost":
        model = XGBRegressor(
            tree_method="hist",
            n_jobs=-1,
            random_state=RANDOM_SEED,
            verbosity=0
        )

        param_dist = XGB_SPACE

        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=min(n_iter, 50),
            cv=tscv,
            scoring=wi_scorer,
            n_jobs=1,
            random_state=RANDOM_SEED,
            verbose=0,
            error_score="raise",
            refit=True
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            search.fit(X_train, y_train)

        return search.best_estimator_, search.best_params_, search.cv_results_

    else:
        raise ValueError(f"Invalid model_name: {model_name}. Use 'RF' or 'XGBoost'.")


def evaluate(model, X: np.ndarray, Y: np.ndarray) -> dict:
    """
    Evaluate model predictions using WI, RMSE, and MAE.

    Args:
        model: Trained sklearn model
        X: Input features
        Y: Ground truth targets (n_samples,) - single instant, Q steps ahead

    Returns:
        dict: Metrics including WI, RMSE, MAE
    """
    if len(X) == 0:
        return {"wi": np.nan, "rmse": np.nan, "mae": np.nan}

    Y_pred = model.predict(X)

    mask = ~np.isnan(Y) & ~np.isnan(Y_pred)
    yt_t = torch.tensor(Y[mask], dtype=torch.float32)
    yp_t = torch.tensor(Y_pred[mask], dtype=torch.float32)

    return {
        "wi": float(wi(yt_t, yp_t)),
        "rmse": float(rmse(yt_t, yp_t)),
        "mae": float(mae(yt_t, yp_t)),
    }


def evaluate_with_fallback(model, X_test: np.ndarray, Y_test: np.ndarray,
                           X_val: np.ndarray, Y_val: np.ndarray,
                           min_samples: int = 1, test_source: str = "test") -> dict:
    """
    Evaluate model using test data if available, otherwise fall back to validation.

    Args:
        model: Trained model
        X_test, Y_test: Test data
        X_val, Y_val: Validation data
        min_samples: Minimum required samples for evaluation
        test_source: Source identifier for logging

    Returns:
        dict: Evaluation metrics with metadata
    """
    if len(X_test) >= min_samples and test_source == "test":
        metrics = evaluate(model, X_test, Y_test)
        metrics["test_source"] = test_source
        metrics["n_test_samples"] = len(X_test)
        return metrics
    elif len(X_val) >= min_samples:
        metrics = evaluate(model, X_val, Y_val)
        metrics["test_source"] = "validation_as_test"
        metrics["n_test_samples"] = len(X_val)
        return metrics
    else:
        return {
            "wi": np.nan, "rmse": np.nan, "mae": np.nan,
            "test_source": "none", "n_test_samples": 0
        }


def run_classic(model_name: str, X_train: np.ndarray, Y_train: np.ndarray,
                X_val: np.ndarray, Y_val: np.ndarray,
                P: int, Q: int, n_iter: int = None, cv: int = None) -> dict:
    """
    Train and evaluate a classical ML model for a single (P, Q) combination.

    Args:
        model_name: "RF" or "XGBoost"
        X_train, Y_train: Training data (Y_train is a single instant, Q
            steps ahead - see SPIDataset)
        X_val, Y_val: Validation data
        P: Input sequence length (for metadata)
        Q: Forecast horizon, in instants ahead (for metadata)
        n_iter: Hyperparameter search iterations
        cv: Cross-validation folds

    Returns:
        dict: Results containing model, metrics, and best parameters
    """
    if len(X_train) == 0:
        return {"model_name": model_name, "P": P, "Q": Q, "val_metrics": {}, "model": None}

    model, best_params, cv_results = randomized_search(model_name, X_train, Y_train, n_iter, cv)

    return {
        "model_name": model_name,
        "P": P,
        "Q": Q,
        "val_metrics": evaluate(model, X_val, Y_val),
        "model": model,
        "best_params": best_params,
        "cv_results": cv_results
    }