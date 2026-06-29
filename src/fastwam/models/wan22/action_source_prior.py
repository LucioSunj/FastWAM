"""VLSP: Video-Latent Source Prior for the action flow source (FastWAM port).

FastWAM's action expert is a rectified-flow denoiser whose *source* endpoint is a
pure Gaussian (``noise_action = torch.randn_like(action)``).  VLSP instead derives
that source from the (first-frame / partially-denoised) video latent via a small
learned prior, so the action flow starts from an informed point:

    baseline:  source ~ N(0, I)
    VLSP:      source ~ q_phi(s | video_latent, proprio)

The source lives in the SAME normalized action space and shape ``[B, T, A]`` as the
action, so it is a drop-in replacement for the Gaussian draw (``scheduler.add_noise``
and ``scheduler.training_target`` are left untouched).

This module is self-contained (only torch) so it can be unit-tested on CPU and added
to ``MoT`` without touching the trainer / checkpoint code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

VIDEO_PRIOR_MODES = frozenset(
    {
        "video_prior_sample",
        "video_prior_mean",
        "video_prior_residual",
        "video_prior_blend",
        "video_prior_dropout",
        "shuffled_video_prior",
    }
)
SOURCE_MODE_IDS = {
    "gaussian": 0,
    "video_prior_sample": 1,
    "video_prior_mean": 2,
    "video_prior_residual": 3,
    "video_prior_blend": 4,
    "video_prior_dropout": 5,
    "shuffled_video_prior": 6,
}


@dataclass
class ActionSourcePriorConfig:
    """Config for the VLSP source prior. Defaults reproduce the Gaussian baseline."""

    enabled: bool = False
    mode: str = "gaussian"
    # which video tensor feeds the prior; only "first_frame" is wired today, but
    # the prior itself is agnostic and consumes whatever latent it is given.
    source_feature: str = "first_frame"

    pool_type: str = "mean"  # mean | attention | perceiver
    hidden_dim: int = 512
    max_action_horizon: int = 64
    num_attention_heads: int = 8
    num_perceiver_latents: int = 8
    mlp_depth: int = 2

    logstd_min: float = -5.0
    logstd_max: float = 1.0
    init_logstd: float = -1.0

    sampling_temperature: float = 1.0
    blend_alpha: float = 1.0
    residual_scale: float = 1.0
    source_dropout_prob: float = 0.0
    dropout_granularity: str = "sample"  # sample | trajectory | element

    detach_video_latents: bool = True
    use_proprio: bool = False

    kl_weight: float = 0.0
    mean_l2_weight: float = 0.0
    std_reg_weight: float = 0.0

    @classmethod
    def from_dict(cls, cfg: Optional[dict]) -> "ActionSourcePriorConfig":
        if cfg is None:
            return cls()
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in dict(cfg).items() if k in valid})


# --------------------------------------------------------------------------- #
#  Building blocks (CPU friendly)                                             #
# --------------------------------------------------------------------------- #
class _MLP(nn.Module):
    def __init__(self, dim: int, depth: int) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.LayerNorm(dim)]
        for _ in range(max(1, depth)):
            layers += [nn.Linear(dim, dim), nn.GELU()]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class _MeanPool(nn.Module):
    def __init__(self, dim_in: int, hidden: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim_in, hidden)

    def forward(self, x_B_N_D: torch.Tensor) -> torch.Tensor:
        return self.proj(x_B_N_D.mean(dim=1))


class _AttentionPool(nn.Module):
    def __init__(self, dim_in: int, hidden: int, num_heads: int) -> None:
        super().__init__()
        self.kv_proj = nn.Linear(dim_in, hidden)
        self.query = nn.Parameter(torch.zeros(1, 1, hidden))
        self.attn = nn.MultiheadAttention(hidden, num_heads, batch_first=True)

    def forward(self, x_B_N_D: torch.Tensor) -> torch.Tensor:
        kv = self.kv_proj(x_B_N_D)
        q = self.query.expand(x_B_N_D.shape[0], -1, -1).to(kv.dtype)
        out, _ = self.attn(q, kv, kv, need_weights=False)
        return out[:, 0]


class _PerceiverPool(nn.Module):
    def __init__(self, dim_in: int, hidden: int, num_latents: int, num_heads: int) -> None:
        super().__init__()
        self.kv_proj = nn.Linear(dim_in, hidden)
        self.latents = nn.Parameter(torch.zeros(1, num_latents, hidden))
        self.attn = nn.MultiheadAttention(hidden, num_heads, batch_first=True)
        self.ff = _MLP(hidden, depth=1)

    def forward(self, x_B_N_D: torch.Tensor) -> torch.Tensor:
        kv = self.kv_proj(x_B_N_D)
        lat = self.latents.expand(x_B_N_D.shape[0], -1, -1).to(kv.dtype)
        out, _ = self.attn(lat, kv, kv, need_weights=False)
        out = self.ff(lat + out)
        return out.mean(dim=1)


class VideoLatentSourcePrior(nn.Module):
    """Maps a video latent (+ optional proprio) to a horizon-aware diagonal Gaussian
    over the flow source. ``mu`` / ``logstd`` have shape ``[B, T, action_dim]``."""

    def __init__(
        self,
        cfg: ActionSourcePriorConfig,
        *,
        action_dim: int,
        video_emb_dim: int,
        proprio_dim: Optional[int],
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.action_dim = action_dim
        hidden = cfg.hidden_dim

        self.ctx_norm = nn.LayerNorm(video_emb_dim)
        if cfg.pool_type == "mean":
            self.pool: nn.Module = _MeanPool(video_emb_dim, hidden)
        elif cfg.pool_type == "attention":
            self.pool = _AttentionPool(video_emb_dim, hidden, cfg.num_attention_heads)
        elif cfg.pool_type == "perceiver":
            self.pool = _PerceiverPool(video_emb_dim, hidden, cfg.num_perceiver_latents, cfg.num_attention_heads)
        else:
            raise ValueError(f"unknown pool_type: {cfg.pool_type!r}")

        self.use_proprio = cfg.use_proprio and proprio_dim is not None
        if self.use_proprio:
            self.proprio_proj = nn.Linear(int(proprio_dim), hidden)

        self.horizon_queries = nn.Parameter(torch.zeros(1, cfg.max_action_horizon, hidden))
        self.trunk = _MLP(hidden, depth=cfg.mlp_depth)
        self.mu_head = nn.Linear(hidden, action_dim)
        self.logstd_head = nn.Linear(hidden, action_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.horizon_queries, std=0.02)
        nn.init.zeros_(self.mu_head.weight)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.zeros_(self.logstd_head.weight)
        nn.init.constant_(self.logstd_head.bias, self.cfg.init_logstd)

    @property
    def _pdtype(self) -> torch.dtype:
        return self.mu_head.weight.dtype

    def forward(
        self,
        video_tokens_B_N_D: torch.Tensor,
        proprio: Optional[torch.Tensor],
        horizon: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = video_tokens_B_N_D.to(self._pdtype)
        h = self.pool(self.ctx_norm(x))  # [B, hidden]
        if self.use_proprio:
            if proprio is not None:
                h = h + self.proprio_proj(proprio.to(h.dtype))
            else:
                # keep proprio_proj in the autograd graph for DDP/ZeRO
                h = h + 0.0 * self.proprio_proj(h.new_zeros((h.shape[0], self.proprio_proj.in_features)))
        q = self.horizon_queries[:, :horizon, :].to(h.dtype)
        per_step = self.trunk(q + h.unsqueeze(1))  # [B, T, hidden]
        mu = self.mu_head(per_step)
        logstd = self.logstd_head(per_step).clamp(min=self.cfg.logstd_min, max=self.cfg.logstd_max)
        return mu, logstd


# --------------------------------------------------------------------------- #
#  Top-level dispatcher                                                        #
# --------------------------------------------------------------------------- #
class ActionSourcePrior(nn.Module):
    """Drop-in replacement for the Gaussian action source.

    ``forward(gaussian, video_latent, ...)`` returns ``(source, metrics)`` where
    ``source`` has the same shape/space/dtype as ``gaussian`` (the original
    ``torch.randn_like(action)``). The passed-in ``gaussian`` is reused as the
    reparameterization noise, so determinism / seeding is inherited from the call
    site and the disabled path is bit-identical to the baseline.
    """

    def __init__(
        self,
        cfg: ActionSourcePriorConfig,
        *,
        action_dim: int,
        video_emb_dim: int,
        proprio_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        # Master switch: only build (trainable) params for actual video modes.
        self.enabled = bool(cfg.enabled) and cfg.mode in VIDEO_PRIOR_MODES
        if self.enabled:
            self.net: Optional[VideoLatentSourcePrior] = VideoLatentSourcePrior(
                cfg, action_dim=action_dim, video_emb_dim=video_emb_dim, proprio_dim=proprio_dim
            )
        else:
            self.net = None

    @staticmethod
    def _flatten_video_latent(video_latent: torch.Tensor) -> torch.Tensor:
        # [B, C, T, H, W] -> [B, T*H*W, C] ; [B, N, C] passes through.
        if video_latent.dim() == 5:
            b, c, t, h, w = video_latent.shape
            return video_latent.permute(0, 2, 3, 4, 1).reshape(b, t * h * w, c)
        if video_latent.dim() == 4:  # [B, C, H, W]
            b, c, h, w = video_latent.shape
            return video_latent.permute(0, 2, 3, 1).reshape(b, h * w, c)
        return video_latent

    def forward(
        self,
        gaussian: torch.Tensor,
        *,
        video_latent: torch.Tensor,
        proprio: Optional[torch.Tensor] = None,
        training: bool = False,
        seed: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        x0: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict]:
        if not self.enabled or self.net is None:
            return gaussian, {}

        B, T, A = gaussian.shape
        eps = gaussian.float()
        device = gaussian.device

        def randn_like_eps() -> torch.Tensor:
            return torch.randn(eps.shape, device=device, dtype=torch.float32, generator=generator)

        cond = video_latent
        shuffle_enabled = 0.0
        if self.cfg.mode == "shuffled_video_prior" and B > 1:
            perm = torch.randperm(B, device=device, generator=generator)
            cond = cond[perm]
            shuffle_enabled = 1.0
        if self.cfg.detach_video_latents:
            cond = cond.detach()

        tokens = self._flatten_video_latent(cond)
        mu, logstd = self.net(tokens, proprio, T)
        mu = mu.float()
        logstd = logstd.float()
        std = logstd.exp()
        temp = float(self.cfg.sampling_temperature)

        mode = self.cfg.mode
        source_video = mu + temp * std * eps
        dropout_rate = torch.zeros((), device=device)
        if mode in ("video_prior_sample", "shuffled_video_prior"):
            source = source_video
        elif mode == "video_prior_mean":
            source = mu
        elif mode == "video_prior_residual":
            source = eps + float(self.cfg.residual_scale) * mu
        elif mode == "video_prior_blend":
            alpha = float(self.cfg.blend_alpha)
            source = alpha * source_video + math.sqrt(max(1.0 - alpha * alpha, 0.0)) * randn_like_eps()
        elif mode == "video_prior_dropout":
            keep = self._dropout_keep_mask((B, T, A), 1.0 - float(self.cfg.source_dropout_prob), device, generator)
            source = keep * source_video + (1.0 - keep) * randn_like_eps()
            dropout_rate = (1.0 - keep).mean()
        else:  # pragma: no cover
            raise ValueError(f"unknown source_mode: {mode!r}")

        # Keep both heads in the autograd graph for every mode (DDP/ZeRO safe).
        source = source + 0.0 * (mu.mean() + logstd.mean())

        metrics = {"mu": mu, "logstd": logstd}
        with torch.no_grad():
            metrics["source/mu_mean"] = mu.mean()
            metrics["source/mu_std"] = mu.std()
            metrics["source/logstd_mean"] = logstd.mean()
            metrics["source/std_mean"] = std.mean()
            metrics["source/source_std"] = source.std()
            metrics["source/shuffle_enabled"] = torch.as_tensor(shuffle_enabled, device=device)
            metrics["source/dropout_rate_actual"] = dropout_rate.detach()
            metrics["source/source_mode_id"] = torch.as_tensor(float(SOURCE_MODE_IDS.get(mode, -1)), device=device)
            if x0 is not None:
                metrics["source/source_vs_x0_mse"] = F.mse_loss(source, x0.float())
            metrics["source/source_vs_gaussian_mse"] = F.mse_loss(source, eps)
        if not torch.isfinite(source).all():
            raise FloatingPointError(f"non-finite VLSP source (mode={mode!r})")
        return source.reshape(B, T, A), metrics

    def _dropout_keep_mask(self, shape, keep_prob, device, generator) -> torch.Tensor:
        b, t, a = shape
        if self.cfg.dropout_granularity in ("sample", "trajectory"):
            mshape = (b, 1, 1)
        elif self.cfg.dropout_granularity == "element":
            mshape = (b, t, a)
        else:
            raise ValueError(f"unknown dropout_granularity: {self.cfg.dropout_granularity!r}")
        u = torch.rand(mshape, device=device, dtype=torch.float32, generator=generator)
        return u.lt(keep_prob).float()


def compute_prior_regularization(metrics: dict, sp: "ActionSourcePrior"):
    """KL / mean-L2 / std regularizers on q(s | video). Returns a tensor or 0.0."""
    if metrics is None or "mu" not in metrics:
        return 0.0
    cfg = sp.cfg
    mu, logstd = metrics["mu"], metrics["logstd"]
    total = 0.0
    if cfg.kl_weight != 0.0:
        var = (2.0 * logstd).exp()
        kl = 0.5 * (mu.pow(2) + var - 1.0 - 2.0 * logstd).mean()
        total = total + cfg.kl_weight * kl
    if cfg.mean_l2_weight != 0.0:
        total = total + cfg.mean_l2_weight * mu.pow(2).mean()
    if cfg.std_reg_weight != 0.0:
        total = total + cfg.std_reg_weight * F.relu(-logstd).mean()
    return total
