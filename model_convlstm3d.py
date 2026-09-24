# model_convlstm3d.py - ConvLSTM3D for SPI forecasting with delta prediction

import torch
import torch.nn as nn
import torch.utils.checkpoint


class ConvLSTM3DCell(nn.Module):
    """ConvLSTM cell for 3D spatiotemporal processing."""

    def __init__(self, in_channels: int, hidden_channels: int,
                 kernel_size: int = 3, dropout: float = 0.3):
        super().__init__()
        padding = kernel_size // 2
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.dropout = nn.Dropout2d(p=dropout)

        self.conv = nn.Conv3d(
            in_channels + hidden_channels, 4 * hidden_channels,
            kernel_size=(1, kernel_size, kernel_size),
            padding=(0, padding, padding), bias=False
        )
        self.bn = nn.BatchNorm3d(4 * hidden_channels)

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor):
        """
        Args:
            x: [B, C, H, W] - input at current time step
            h_prev: [B, hidden, H, W] - previous hidden state
            c_prev: [B, hidden, H, W] - previous cell state

        Returns:
            h_next, c_next: updated states
        """
        combined = torch.cat([x.unsqueeze(2), h_prev.unsqueeze(2)], dim=1)
        gates = self.bn(self.conv(combined)).squeeze(2)
        i, f, o, g = torch.chunk(gates, 4, dim=1)

        c_next = torch.sigmoid(f) * c_prev + torch.sigmoid(i) * torch.tanh(g)
        h_next = torch.sigmoid(o) * torch.tanh(c_next)
        h_next = self.dropout(h_next)

        return h_next, c_next


class ConvLSTM3DEncoder(nn.Module):
    """Encoder: processes input sequence and extracts spatiotemporal features."""

    def __init__(self, input_dim: int = 3, hidden_dims: tuple = (32, 16, 8),
                 dropout: float = None, use_checkpoint: bool = False):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.input_dim = input_dim
        self.dropout = dropout
        self.use_checkpoint = use_checkpoint

        self.layers = nn.ModuleList()
        prev_dim = input_dim
        for h in hidden_dims:
            self.layers.append(ConvLSTM3DCell(prev_dim, h, dropout=dropout))
            prev_dim = h

        self.skip_proj = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dims[-1], 1),
            nn.BatchNorm2d(hidden_dims[-1])
        )
        self.layer_norm = nn.LayerNorm(hidden_dims[-1])
        self.bn_skip = nn.BatchNorm2d(hidden_dims[-1])

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        B, P, C, H, W = x.shape

        # Downsample if resolution is too high
        original_size = (H, W)
        need_resize = H * W > 2048

        if need_resize:
            scale_factor = min(1.0, (2048 / (H * W)) ** 0.5)
            new_h, new_w = int(H * scale_factor), int(W * scale_factor)
            x_reshaped = x.view(B * P, C, H, W)
            x_reshaped = nn.functional.interpolate(
                x_reshaped, size=(new_h, new_w), mode='bilinear', align_corners=False
            )
            x = x_reshaped.view(B, P, C, new_h, new_w)
            H, W = new_h, new_w

        # Initialize hidden and cell states
        h = [torch.zeros(B, hd, H, W, device=x.device) for hd in self.hidden_dims]
        c = [torch.zeros(B, hd, H, W, device=x.device) for hd in self.hidden_dims]

        # Process sequence - simple architecture, no attention: only the
        # final ConvLSTM hidden state is used (a plain seq-to-one encoder).
        for t in range(P):
            inp = x[:, t]
            for i, cell in enumerate(self.layers):
                h[i], c[i] = cell(inp, h[i], c[i])
                inp = h[i]
        last_hidden = h[-1]

        # Skip connection
        skip = self.skip_proj(x[:, 0])

        if last_hidden.shape[2:] != skip.shape[2:]:
            last_hidden = nn.functional.interpolate(
                last_hidden, size=skip.shape[2:], mode='bilinear', align_corners=False
            )

        out = self.bn_skip(last_hidden + skip)

        if need_resize:
            out = nn.functional.interpolate(
                out, size=original_size, mode='bilinear', align_corners=False
            )

        out = out.permute(0, 2, 3, 1)
        out = self.layer_norm(out)
        out = out.permute(0, 3, 1, 2)

        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training and hasattr(torch.utils, 'checkpoint'):
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl, x, use_reentrant=False
            )
        return self._forward_impl(x)


class ConvLSTM3D(nn.Module):
    """
    Simple ConvLSTM3D model for SPI forecasting using delta prediction.

    Strategy: Predict SPI variation (delta) instead of absolute values.
    SPI_predicted = last_observed_SPI + delta_predicted

    Deliberately simple (no channel/spatial/temporal attention): one model
    is trained per (P, Q) combination, directly mapping the P-step input
    window to the single target instant Q steps ahead - the same
    single-target-per-horizon strategy used by
    drought_forecast_binary/regression, rather than an autoregressive
    multi-step rollout. This answers whether a plain, hyperparameter-tuned
    ConvLSTM is competitive with the classical (RF/XGBoost) baselines and
    with the more elaborate attention-based architectures used elsewhere in
    this thesis, without the confound of a fancier architecture.

    Args:
        hidden: Tuple of hidden channels per layer
        dropout_p: Dropout rate (default: 0.3)
        use_checkpoint: Enable gradient checkpointing for memory efficiency
    """

    def __init__(self, hidden: tuple, dropout_p: float = 0.3, use_checkpoint: bool = False):
        super().__init__()
        if not isinstance(hidden, (tuple, list)):
            raise ValueError(f"hidden must be a tuple/list, got {type(hidden)}")

        self.hidden_dims = hidden
        self.dropout_p = dropout_p
        self.use_checkpoint = use_checkpoint

        self.encoder = ConvLSTM3DEncoder(
            input_dim=3,
            hidden_dims=hidden,
            dropout=dropout_p,
            use_checkpoint=use_checkpoint
        )

        # Refinement block with residual connection
        self.refine = nn.Sequential(
            nn.Conv2d(hidden[-1], hidden[-1], 3, padding=1),
            nn.BatchNorm2d(hidden[-1]),
            nn.Conv2d(hidden[-1], hidden[-1], 3, padding=1),
            nn.BatchNorm2d(hidden[-1])
        )

        self.dropout = nn.Dropout2d(p=dropout_p)

        # Prediction head for delta SPI (variation)
        self.head = nn.Sequential(
            nn.Conv2d(hidden[-1], 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.Tanh(),
            nn.Dropout2d(p=dropout_p),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.Tanh(),
            nn.Conv2d(64, 1, 1)
        )

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Predict delta SPI for the target instant."""
        h = self.encoder(x)

        residual = h
        h = self.refine(h)
        h = h + residual
        h = self.dropout(h)

        delta_pred = self.head(h).squeeze(1)
        return delta_pred

    def forward(self, x: torch.Tensor, use_checkpoint: bool = None) -> torch.Tensor:
        """
        Predict absolute SPI for the target instant (Q steps after the
        input window - see SPIDataset).

        Args:
            x: Input tensor [B, P, C, H, W]
            use_checkpoint: Override model's checkpoint setting

        Returns:
            spi_pred: Absolute SPI prediction [B, H, W]
        """
        if use_checkpoint is None:
            use_checkpoint = self.use_checkpoint

        if use_checkpoint and self.training and hasattr(torch.utils, 'checkpoint'):
            delta_pred = torch.utils.checkpoint.checkpoint(
                self._forward_impl, x, use_reentrant=False
            )
        else:
            delta_pred = self._forward_impl(x)

        # SPI_pred = last_observed_SPI + delta. Channel 1 is SPI, last time
        # step is the most recent observed month (persistence anchor).
        last_spi = x[:, -1, 1]
        spi_pred = last_spi + delta_pred

        return spi_pred

    def get_config(self) -> dict:
        """Return model configuration for logging."""
        return {
            "hidden_dims": self.hidden_dims,
            "dropout_p": self.dropout_p,
            "use_checkpoint": self.use_checkpoint,
            "num_layers": len(self.hidden_dims)
        }