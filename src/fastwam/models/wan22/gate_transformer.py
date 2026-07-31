from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastwam.adapters import PolicyRegime

from .kv_tap import GateKVSnapshot, GateLayerKV, KVSource, KeyValueBank


@dataclass(frozen=True)
class LayerTapConfig:
    """Select which MoT layers are visible to the Gate."""

    mode: str = "all"
    last_n: int | None = None
    indices: tuple[int, ...] | None = None

    def resolve(self, num_layers: int) -> tuple[int, ...]:
        if num_layers < 1:
            raise ValueError(f"`num_layers` must be positive, got {num_layers}.")
        if self.mode == "all":
            if self.last_n is not None or self.indices is not None:
                raise ValueError("`all` layer selection cannot also specify `last_n` or `indices`.")
            return tuple(range(num_layers))
        if self.mode == "last_n":
            if self.last_n is None or self.last_n < 1 or self.last_n > num_layers:
                raise ValueError(
                    f"`last_n` must be in [1, {num_layers}], got {self.last_n}."
                )
            if self.indices is not None:
                raise ValueError("`last_n` layer selection cannot also specify `indices`.")
            return tuple(range(num_layers - self.last_n, num_layers))
        if self.mode == "indices":
            if self.indices is None or not self.indices:
                raise ValueError("`indices` layer selection requires at least one index.")
            if self.last_n is not None:
                raise ValueError("`indices` layer selection cannot also specify `last_n`.")
            if len(set(self.indices)) != len(self.indices):
                raise ValueError(f"`indices` contains duplicates: {self.indices}.")
            if tuple(sorted(self.indices)) != self.indices:
                raise ValueError("`indices` must be strictly increasing.")
            invalid = [index for index in self.indices if index < 0 or index >= num_layers]
            if invalid:
                raise ValueError(
                    f"Layer indices {invalid} are outside a {num_layers}-layer MoT."
                )
            return self.indices
        raise ValueError(
            f"Unknown layer tap mode {self.mode!r}; expected 'all', 'last_n', or 'indices'."
        )


@dataclass(frozen=True)
class GateTransformerConfig:
    """Architecture and tap aggregation settings for the Gate sidecar."""

    num_mot_layers: int
    source_num_heads: int
    source_head_dim: int
    hidden_dim: int = 256
    num_query_tokens: int = 4
    share_blocks: bool = False
    ffn_multiplier: int = 4
    denoise_last_n: int = 1
    layer_taps: LayerTapConfig = field(default_factory=LayerTapConfig)
    current_mode_embedding: bool = True
    layer_index_embedding: bool = True
    denoise_timestep_embedding: bool = True

    def __post_init__(self) -> None:
        integer_fields = {
            "num_mot_layers": self.num_mot_layers,
            "source_num_heads": self.source_num_heads,
            "source_head_dim": self.source_head_dim,
            "hidden_dim": self.hidden_dim,
            "num_query_tokens": self.num_query_tokens,
            "ffn_multiplier": self.ffn_multiplier,
            "denoise_last_n": self.denoise_last_n,
        }
        invalid = {name: value for name, value in integer_fields.items() if value < 1}
        if invalid:
            raise ValueError(f"Gate dimensions/counts must be positive, got {invalid}.")
        self.layer_taps.resolve(self.num_mot_layers)

    @property
    def source_dim(self) -> int:
        return self.source_num_heads * self.source_head_dim


class DirectKVAttention(nn.Module):
    """Cross-attention that consumes already-projected source K/V directly."""

    def __init__(self, hidden_dim: int, source_num_heads: int, source_head_dim: int) -> None:
        super().__init__()
        self.source_num_heads = source_num_heads
        self.source_head_dim = source_head_dim
        self.source_dim = source_num_heads * source_head_dim
        self.query = nn.Linear(hidden_dim, self.source_dim)
        self.output = nn.Linear(self.source_dim, hidden_dim)

    def forward(self, query: torch.Tensor, bank: KeyValueBank) -> torch.Tensor:
        bank = bank.detached()
        if bank.feature_dim != self.source_dim:
            raise ValueError(
                f"Gate expected source K/V dim {self.source_dim}, got {bank.feature_dim}."
            )
        if query.shape[0] != bank.batch_size:
            raise ValueError(
                f"Gate query batch {query.shape[0]} does not match K/V batch {bank.batch_size}."
            )

        batch_size, query_len = query.shape[:2]
        source_len = bank.sequence_length
        q = self.query(query).view(
            batch_size, query_len, self.source_num_heads, self.source_head_dim
        )
        k = bank.key.to(dtype=query.dtype).view(
            batch_size, source_len, self.source_num_heads, self.source_head_dim
        )
        v = bank.value.to(dtype=query.dtype).view(
            batch_size, source_len, self.source_num_heads, self.source_head_dim
        )

        scores = torch.einsum("bqhd,bshd->bhqs", q, k)
        scores = scores / math.sqrt(self.source_head_dim)
        valid = bank.valid_mask[:, None, None, :]
        scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores.float(), dim=-1).to(dtype=scores.dtype)
        weights = weights * valid.to(dtype=weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(weights.dtype).tiny
        )
        attended = torch.einsum("bhqs,bshd->bqhd", weights, v)
        return self.output(attended.reshape(batch_size, query_len, self.source_dim))


def _combine_source_banks(video: KeyValueBank, action: KeyValueBank) -> KeyValueBank:
    if video.batch_size != action.batch_size or video.feature_dim != action.feature_dim:
        raise ValueError("Current-frame video and action K/V banks are incompatible.")
    return KeyValueBank(
        source=KVSource.CURRENT_VIDEO_AND_ACTION,
        key=torch.cat([video.key.detach(), action.key.detach()], dim=1),
        value=torch.cat([video.value.detach(), action.value.detach()], dim=1),
        valid_mask=torch.cat([video.valid_mask.detach(), action.valid_mask.detach()], dim=1),
    )


class GateTransformerBlock(nn.Module):
    """One read-only Gate block for one selected MoT layer."""

    def __init__(self, config: GateTransformerConfig) -> None:
        super().__init__()
        self.source_norm = nn.LayerNorm(config.hidden_dim)
        self.source_attention = DirectKVAttention(
            config.hidden_dim,
            config.source_num_heads,
            config.source_head_dim,
        )
        self.context_norm = nn.LayerNorm(config.hidden_dim)
        self.context_attention = DirectKVAttention(
            config.hidden_dim,
            config.source_num_heads,
            config.source_head_dim,
        )
        self.ffn_norm = nn.LayerNorm(config.hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, config.ffn_multiplier * config.hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.ffn_multiplier * config.hidden_dim, config.hidden_dim),
        )

    def forward(self, query: torch.Tensor, layer: GateLayerKV) -> torch.Tensor:
        layer = layer.detached()
        source = _combine_source_banks(layer.current_frame_video, layer.action)
        query = query + self.source_attention(self.source_norm(query), source)
        query = query + self.context_attention(self.context_norm(query), layer.context)
        return query + self.ffn(self.ffn_norm(query))


@dataclass(frozen=True)
class GateBehavior:
    """Exact epsilon-mixture Bernoulli behavior policy."""

    base_idm_probability: torch.Tensor
    behavior_idm_probability: torch.Tensor

    @property
    def distribution(self) -> torch.distributions.Bernoulli:
        return torch.distributions.Bernoulli(probs=self.behavior_idm_probability)

    def sample(self, *, generator: torch.Generator | None = None) -> torch.Tensor:
        """Sample a route tensor where one means IDM and zero means UNCOND."""

        return torch.bernoulli(
            self.behavior_idm_probability,
            generator=generator,
        ).to(dtype=torch.long)

    def log_prob(self, route: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(route.to(dtype=self.behavior_idm_probability.dtype))


def _policy_parameter(
    value: float | torch.Tensor,
    *,
    name: str,
    logits: torch.Tensor,
) -> torch.Tensor:
    parameter = torch.as_tensor(value, dtype=logits.dtype, device=logits.device)
    try:
        _broadcast_logits, parameter = torch.broadcast_tensors(logits, parameter)
    except RuntimeError as error:
        raise ValueError(
            f"`{name}` with shape {tuple(parameter.shape)} is not broadcastable "
            f"to logits {tuple(logits.shape)}."
        ) from error
    if parameter.shape != logits.shape:
        raise ValueError(
            f"`{name}` would expand logits from {tuple(logits.shape)} to "
            f"{tuple(parameter.shape)}."
        )
    return parameter


def epsilon_mixture_bernoulli(
    logits: torch.Tensor,
    *,
    temperature: float | torch.Tensor,
    epsilon: float | torch.Tensor,
) -> GateBehavior:
    """Construct the exact behavior policy used for Gate exploration."""

    temperature_tensor = _policy_parameter(
        temperature,
        name="temperature",
        logits=logits,
    )
    if bool(((~torch.isfinite(temperature_tensor)) | (temperature_tensor <= 0)).any()):
        raise ValueError(f"`temperature` must be finite and positive, got {temperature}.")
    epsilon_tensor = _policy_parameter(epsilon, name="epsilon", logits=logits)
    if bool(((~torch.isfinite(epsilon_tensor)) | (epsilon_tensor < 0) | (epsilon_tensor > 1)).any()):
        raise ValueError("`epsilon` must be finite and lie in [0, 1].")
    base_probability = torch.sigmoid(logits / temperature_tensor)
    behavior_probability = (1 - epsilon_tensor) * base_probability + epsilon_tensor / 2
    return GateBehavior(
        base_idm_probability=base_probability,
        behavior_idm_probability=behavior_probability,
    )


def deterministic_idm_route(
    logits: torch.Tensor,
    *,
    temperature: float | torch.Tensor = 1.0,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Return a deterministic route tensor where one means IDM."""

    temperature_tensor = _policy_parameter(
        temperature,
        name="temperature",
        logits=logits,
    )
    if bool(((~torch.isfinite(temperature_tensor)) | (temperature_tensor <= 0)).any()):
        raise ValueError(f"`temperature` must be finite and positive, got {temperature}.")
    if not 0 <= threshold <= 1:
        raise ValueError(f"`threshold` must lie in [0, 1], got {threshold}.")
    return (torch.sigmoid(logits / temperature_tensor) >= threshold).to(dtype=torch.long)


class GateTransformer(nn.Module):
    """Small read-only Transformer that predicts whether the next chunk uses IDM."""

    def __init__(self, config: GateTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.selected_layers = config.layer_taps.resolve(config.num_mot_layers)
        self.query_tokens = nn.Parameter(
            torch.empty(1, config.num_query_tokens, config.hidden_dim)
        )
        nn.init.normal_(self.query_tokens, std=config.hidden_dim**-0.5)

        block_count = 1 if config.share_blocks else len(self.selected_layers)
        self.blocks = nn.ModuleList(
            GateTransformerBlock(config) for _ in range(block_count)
        )
        self.mode_embedding = (
            nn.Embedding(2, config.hidden_dim) if config.current_mode_embedding else None
        )
        self.layer_embedding = (
            nn.Embedding(config.num_mot_layers, config.hidden_dim)
            if config.layer_index_embedding
            else None
        )
        self.timestep_projection = (
            nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.SiLU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
            )
            if config.denoise_timestep_embedding
            else None
        )
        self.output_norm = nn.LayerNorm(config.hidden_dim)
        self.output = nn.Linear(config.hidden_dim, 1)

    def _timestep_embedding(self, timestep: torch.Tensor) -> torch.Tensor:
        half_dim = self.config.hidden_dim // 2
        if half_dim == 0:
            return timestep[:, None]
        exponent = -math.log(10000.0) * torch.arange(
            half_dim,
            dtype=torch.float32,
            device=timestep.device,
        ) / max(half_dim - 1, 1)
        angles = timestep.float()[:, None] * exponent.exp()[None, :]
        embedding = torch.cat([angles.cos(), angles.sin()], dim=-1)
        if embedding.shape[-1] < self.config.hidden_dim:
            embedding = F.pad(embedding, (0, self.config.hidden_dim - embedding.shape[-1]))
        return embedding.to(dtype=timestep.dtype)

    def _conditioning(self, layer: GateLayerKV, dtype: torch.dtype) -> torch.Tensor:
        device = layer.action.key.device
        conditioning = torch.zeros(
            (layer.batch_size, self.config.hidden_dim),
            device=device,
            dtype=dtype,
        )
        if self.mode_embedding is not None:
            mode_ids = torch.tensor(
                [1 if mode is PolicyRegime.IDM else 0 for mode in layer.current_mode],
                device=device,
                dtype=torch.long,
            )
            conditioning = conditioning + self.mode_embedding(mode_ids).to(dtype=dtype)
        if self.layer_embedding is not None:
            layer_ids = torch.full(
                (layer.batch_size,),
                layer.layer_index,
                device=device,
                dtype=torch.long,
            )
            conditioning = conditioning + self.layer_embedding(layer_ids).to(dtype=dtype)
        if self.timestep_projection is not None:
            time_embedding = self._timestep_embedding(layer.denoise_timestep).to(dtype=dtype)
            conditioning = conditioning + self.timestep_projection(time_embedding).to(dtype=dtype)
        return conditioning

    def forward_snapshot(self, snapshot: GateKVSnapshot) -> torch.Tensor:
        """Compute pre-sigmoid IDM logits for one denoising snapshot."""

        missing = [index for index in self.selected_layers if index not in snapshot.layer_indices]
        if missing:
            raise ValueError(
                f"Gate snapshot is missing configured MoT layers {missing}; "
                f"available={snapshot.layer_indices}."
            )
        first = snapshot.layer(self.selected_layers[0])
        if first.feature_dim != self.config.source_dim:
            raise ValueError(
                f"Gate configured for source dim {self.config.source_dim}, "
                f"snapshot provides {first.feature_dim}."
            )

        if first.action.key.device != self.query_tokens.device:
            raise ValueError(
                "Gate K/V and Gate parameters must be on the same device; materialize "
                "the stored snapshot on the Gate compute device first."
            )
        query = self.query_tokens.expand(first.batch_size, -1, -1)
        for block_offset, layer_index in enumerate(self.selected_layers):
            layer = snapshot.layer(layer_index).detached()
            query = query + self._conditioning(layer, query.dtype).unsqueeze(1)
            block = self.blocks[0] if self.config.share_blocks else self.blocks[block_offset]
            query = block(query, layer)
        pooled = self.output_norm(query).mean(dim=1)
        return self.output(pooled).squeeze(-1)

    @staticmethod
    def aggregate_denoise_logits(logits: torch.Tensor) -> torch.Tensor:
        """Aggregate denoising taps before sigmoid, never after it."""

        if logits.ndim < 2 or logits.shape[0] < 1:
            raise ValueError(
                "`logits` must be [num_denoise_taps, ...] with at least one tap."
            )
        return logits.mean(dim=0)

    def forward(self, snapshots: list[GateKVSnapshot] | tuple[GateKVSnapshot, ...]) -> torch.Tensor:
        if len(snapshots) < self.config.denoise_last_n:
            raise ValueError(
                f"Gate requires the last {self.config.denoise_last_n} denoising snapshots, "
                f"got {len(snapshots)}."
            )
        selected = snapshots[-self.config.denoise_last_n :]
        logits = torch.stack([self.forward_snapshot(snapshot) for snapshot in selected], dim=0)
        return self.aggregate_denoise_logits(logits)
