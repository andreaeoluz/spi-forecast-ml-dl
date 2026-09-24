# train_model.py - Training loop for the (attention-free) ConvLSTM3D model

import gc
import random as _random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import CONVLSTM3D_PARAMS, CONVLSTM3D_SPACE, CONVLSTM3D_SEARCH_ITER, RANDOM_SEED, DEVICE
from metrics import wi, rmse
from plots import plot_training_curves
from model_convlstm3d import ConvLSTM3D


def masked_smooth_l1_loss(pred: torch.Tensor, target: torch.Tensor,
                           mask: torch.Tensor, beta: float = 0.5) -> torch.Tensor:
    """
    Smooth L1 (Huber) loss restricted to valid target entries.

    `target` may contain zero-filled placeholders for pixels/months where
    SPI could not be computed (see SPIDataset docstring); `mask` (1.0 =
    valid, 0.0 = placeholder) excludes those entries from the loss so the
    model is never trained to reproduce fabricated zeros.
    """
    diff = (pred - target).abs()
    elementwise = torch.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    denom = mask.sum().clamp_min(1.0)
    return (elementwise * mask).sum() / denom


def train_model(model, dataset_train, dataset_val, P, Q,
                epochs=None, lr=None, batch_size=None, device=None,
                patience=None):
    """
    Train the ConvLSTM3D model for a single (P, Q) combination.

    One model predicts one target instant (Q steps after the P-step input
    window - see SPIDataset); there is no teacher forcing or autoregressive
    rollout, since there is nothing to roll out over - a single forward
    pass per sample, exactly like drought_forecast_binary/regression's
    predictors.

    Args:
        model: ConvLSTM3D instance
        dataset_train: Training dataset
        dataset_val: Validation dataset
        P: Input sequence length
        Q: Forecast horizon (instants ahead)
        epochs: Maximum number of epochs
        lr: Learning rate
        batch_size: Batch size
        device: torch device
        patience: Early stopping patience

    Returns:
        Trained model (loaded with best weights)
    """
    epochs = epochs or CONVLSTM3D_PARAMS["epochs"]
    lr = lr or CONVLSTM3D_PARAMS["lr"]
    batch_size = batch_size or CONVLSTM3D_PARAMS["batch_size"]
    device = device or DEVICE
    patience = patience or CONVLSTM3D_PARAMS["patience"]

    if len(dataset_train) == 0:
        print(f"  ⚠ No training data for P={P}, Q={Q}")
        return model

    print(f"\n  📊 Training: P={P}, Q={Q} | epochs={epochs} | lr={lr} | batch={batch_size}")
    print(f"     Train samples: {len(dataset_train)} | Val samples: {len(dataset_val)}")

    sample_x, sample_y, _ = dataset_train[0]
    P_seq, C, H, W = sample_x.shape
    print(f"     Input shape: [P={P_seq}, C={C}, H={H}, W={W}]")
    print(f"     Target shape: [H={H}, W={W}] (single instant, Q={Q})")

    # Windows compatibility: num_workers must be 0
    loader_train = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
        num_workers=0,
    )

    loader_val = DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
        num_workers=0,
    ) if len(dataset_val) > 0 else None

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-5,
        betas=(0.9, 0.999)
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=patience // 2,
        min_lr=1e-6
    )

    best_wi = -float("inf")
    best_state = None
    patience_counter = 0
    history_loss, history_wi, history_rmse = [], [], []

    print("     Epoch | Loss    | Val WI  | Val RMSE | LR")
    print(f"     {'-' * 48}")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for x, y, y_mask in loader_train:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            y_mask = y_mask.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            pred = model(x)
            loss = masked_smooth_l1_loss(pred, y, y_mask, beta=0.5)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        history_loss.append(avg_loss)

        # Periodic cache cleanup
        if device.type == "cuda" and epoch % 5 == 0:
            torch.cuda.empty_cache()
            gc.collect()

        # === VALIDATION ===
        if loader_val is not None and len(dataset_val) > 0:
            model.eval()
            yt_all, yp_all, mask_all = [], [], []

            with torch.no_grad():
                for x, y, y_mask in loader_val:
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    y_mask = y_mask.to(device, non_blocking=True)

                    pred = model(x)

                    yt_all.append(y.flatten())
                    yp_all.append(pred.flatten())
                    mask_all.append(y_mask.flatten())

            yt_all = torch.cat(yt_all)
            yp_all = torch.cat(yp_all)
            mask_all = torch.cat(mask_all)

            val_wi = float(wi(yt_all, yp_all, mask=mask_all).cpu())
            val_rmse = float(rmse(yt_all, yp_all, mask=mask_all).cpu())

            history_wi.append(val_wi)
            history_rmse.append(val_rmse)

            scheduler.step(val_wi)
            current_lr = optimizer.param_groups[0]['lr']

            print(f"     {epoch:4d} | {avg_loss:6.4f} | {val_wi:7.4f} | {val_rmse:7.4f} | {current_lr:.6f}")

            # Early stopping logic
            if val_wi - best_wi > 1e-6:
                best_wi = val_wi
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"     ⏹ Early stopping at epoch {epoch}")
                break
        else:
            print(f"     {epoch:4d} | {avg_loss:6.4f} | {'N/A':7} | {'N/A':7} | {optimizer.param_groups[0]['lr']:.6f}")

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    model.best_wi = best_wi

    print(f"     ✅ Best WI: {best_wi:.4f}")

    plot_training_curves(history_loss, history_wi, history_rmse, P, Q)

    # Clean up
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()

    return model


def random_search_convlstm3d(dataset_train, dataset_val, P, Q,
                              n_iter=None, device=None):
    """
    Random hyperparameter search for the simple (attention-free) ConvLSTM3D,
    analogous to model_classic.randomized_search for RF/XGBoost: samples
    `n_iter` random configurations from CONVLSTM3D_SPACE, trains each one on
    `dataset_train` with early stopping against `dataset_val`, and keeps the
    configuration with the best validation WI.

    Unlike RandomizedSearchCV (RF/XGBoost), this does not use k-fold cross-
    validation - each candidate gets a single train/val split, since
    retraining a ConvLSTM k times per candidate would be too expensive.
    Returns the same result shape as model_classic.run_classic so main.py
    can treat every model type uniformly.
    """
    n_iter = n_iter or CONVLSTM3D_SEARCH_ITER
    device = device or DEVICE

    if len(dataset_train) == 0:
        return {
            "model_name": "ConvLSTM3D", "P": P, "Q": Q,
            "val_metrics": {}, "model": None,
            "best_params": None, "cv_results": None,
        }

    # Seeded per (P, Q) so the search is reproducible but not identical
    # across every combination.
    rng = _random.Random(RANDOM_SEED + P * 100 + Q)
    keys = list(CONVLSTM3D_SPACE.keys())

    best_model = None
    best_wi = -float("inf")
    best_params = None
    all_results = []

    for i in range(n_iter):
        params = {k: rng.choice(CONVLSTM3D_SPACE[k]) for k in keys}
        print(f"\n    [ConvLSTM3D search {i + 1}/{n_iter}] {params}")

        model = ConvLSTM3D(
            params["hidden"],
            dropout_p=params["dropout"],
            use_checkpoint=CONVLSTM3D_PARAMS.get("use_checkpoint", False),
        ).to(device)

        model = train_model(
            model, dataset_train, dataset_val, P, Q,
            epochs=CONVLSTM3D_PARAMS["epochs"],
            lr=params["lr"],
            batch_size=params["batch_size"],
            device=device,
            patience=CONVLSTM3D_PARAMS["patience"],
        )

        all_results.append({
            "hidden": params["hidden"], "dropout": params["dropout"],
            "lr": params["lr"], "batch_size": params["batch_size"],
            "val_wi": model.best_wi,
        })

        if model.best_wi > best_wi:
            best_wi = model.best_wi
            best_model = model
            best_params = params

    print(f"\n    [ConvLSTM3D search] best val WI = {best_wi:.4f} with {best_params}")

    return {
        "model_name": "ConvLSTM3D", "P": P, "Q": Q,
        "val_metrics": {"wi": best_wi},
        "model": best_model,
        "best_params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in best_params.items()},
        "cv_results": all_results,
    }
