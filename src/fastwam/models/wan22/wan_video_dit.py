import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Any, Dict, Tuple, Optional
from .helpers.gradient import gradient_checkpoint_forward

from fastwam.utils.logging_config import get_logger

logger = get_logger(__name__)

    
def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, ctx_mask: Optional[torch.Tensor] = None, compatibility_mode=True):
    if compatibility_mode:
        B, S_q, _ = q.shape
        S_k = k.shape[1]
        head_dim = q.shape[2] // num_heads
        q = q.view(B, S_q, num_heads, head_dim).transpose(1, 2)
        k = k.view(B, S_k, num_heads, head_dim).transpose(1, 2)
        v = v.view(B, S_k, num_heads, head_dim).transpose(1, 2)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=ctx_mask)
        x = x.transpose(1, 2).reshape(B, S_q, -1)
        return x
    else:
        raise NotImplementedError("Only compatibility mode is implemented for flash attention. Please set compatibility_mode=True.")



def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x * (1 + scale) + shift)


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def rope_apply(x, freqs, num_heads):
    B, S, D = x.shape
    x = x.view(B, S, num_heads, -1)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(B, S, x.shape[2], -1, 2))
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


def create_group_causal_attn_mask(
    num_temporal_groups: int, num_query_per_group: int, num_key_per_group: int, mode: str = "causal"
) -> torch.Tensor:
    """
    Creates a group-based attention mask for scaled dot-product attention with two modes:
    'causal' and 'group_diagonal'.

    Parameters:
    - num_temporal_groups (int): The number of temporal groups (e.g., frames in a video sequence).
    - num_query_per_group (int): The number of query tokens per temporal group. (e.g., latent tokens in a frame, H x W).
    - num_key_per_group (int): The number of key tokens per temporal group. (e.g., action tokens per frame).
    - mode (str): The mode of the attention mask. Options are:
        - 'causal': Query tokens can attend to key tokens from the same or previous temporal groups.
        - 'group_diagonal': Query tokens can attend only to key tokens from the same temporal group.

    Returns:
    - attn_mask (torch.Tensor): A boolean tensor of shape (L, S), where:
        - L = num_temporal_groups * num_query_per_group (total number of query tokens)
        - S = num_temporal_groups * num_key_per_group (total number of key tokens)
      The mask indicates where attention is allowed (True) and disallowed (False).

    Example:
    Input:
        num_temporal_groups = 3
        num_query_per_group = 4
        num_key_per_group = 2
    Output:
        Causal Mask Shape: torch.Size([12, 6])
        Group Diagonal Mask Shape: torch.Size([12, 6])
        if mode='causal':
        tensor([[ True,  True, False, False, False, False],
                [ True,  True, False, False, False, False],
                [ True,  True, False, False, False, False],
                [ True,  True, False, False, False, False],
                [ True,  True,  True,  True, False, False],
                [ True,  True,  True,  True, False, False],
                [ True,  True,  True,  True, False, False],
                [ True,  True,  True,  True, False, False],
                [ True,  True,  True,  True,  True,  True],
                [ True,  True,  True,  True,  True,  True],
                [ True,  True,  True,  True,  True,  True],
                [ True,  True,  True,  True,  True,  True]])

        if mode='group_diagonal':
        tensor([[ True,  True, False, False, False, False],
                [ True,  True, False, False, False, False],
                [ True,  True, False, False, False, False],
                [ True,  True, False, False, False, False],
                [False, False,  True,  True, False, False],
                [False, False,  True,  True, False, False],
                [False, False,  True,  True, False, False],
                [False, False,  True,  True, False, False],
                [False, False, False, False,  True,  True],
                [False, False, False, False,  True,  True],
                [False, False, False, False,  True,  True],
                [False, False, False, False,  True,  True]])

    """
    assert mode in ["causal", "group_diagonal"], f"Mode {mode} must be 'causal' or 'group_diagonal'"

    # Total number of query and key tokens
    total_num_query_tokens = num_temporal_groups * num_query_per_group  # Total number of query tokens (L)
    total_num_key_tokens = num_temporal_groups * num_key_per_group  # Total number of key tokens (S)

    # Generate time indices for query and key tokens (shape: [L] and [S])
    query_time_indices = torch.arange(num_temporal_groups).repeat_interleave(num_query_per_group)  # Shape: [L]
    key_time_indices = torch.arange(num_temporal_groups).repeat_interleave(num_key_per_group)  # Shape: [S]

    # Expand dimensions to compute outer comparison
    query_time_indices = query_time_indices.unsqueeze(1)  # Shape: [L, 1]
    key_time_indices = key_time_indices.unsqueeze(0)  # Shape: [1, S]

    if mode == "causal":
        # Causal Mode: Query can attend to keys where key_time <= query_time
        attn_mask = query_time_indices >= key_time_indices  # Shape: [L, S]
    elif mode == "group_diagonal":
        # Group Diagonal Mode: Query can attend only to keys where key_time == query_time
        attn_mask = query_time_indices == key_time_indices  # Shape: [L, S]

    assert attn_mask.shape == (total_num_query_tokens, total_num_key_tokens), "Attention mask shape mismatch"
    return attn_mask


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight


class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        
    def forward(self, q, k, v, ctx_mask=None):
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, ctx_mask=ctx_mask)
        return x


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, attn_head_dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.attn_hidden_dim = self.num_heads * self.attn_head_dim

        self.q = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.k = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.v = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.o = nn.Linear(self.attn_hidden_dim, hidden_dim)
        self.norm_q = RMSNorm(self.attn_hidden_dim, eps=eps)
        self.norm_k = RMSNorm(self.attn_hidden_dim, eps=eps)
        
        # self.attn = AttentionModule(self.num_heads)

    def forward(self, x, freqs, self_attn_mask: Optional[torch.Tensor] = None):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, ctx_mask=self_attn_mask)
        return self.o(x)


class CrossAttention(nn.Module):
    def __init__(self, hidden_dim: int, attn_head_dim: int, num_heads: int, eps: float = 1e-6,):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.attn_hidden_dim = self.num_heads * self.attn_head_dim

        self.q = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.k = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.v = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.o = nn.Linear(self.attn_hidden_dim, hidden_dim)
        self.norm_q = RMSNorm(self.attn_hidden_dim, eps=eps)
        self.norm_k = RMSNorm(self.attn_hidden_dim, eps=eps)
            
        # self.attn = AttentionModule(self.num_heads)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor, ctx_mask: Optional[torch.Tensor] = None):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, ctx_mask=ctx_mask)
        return self.o(x)


class GateModule(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x, gate, residual):
        return x + gate * residual

class DiTBlock(nn.Module):
    def __init__(self,  hidden_dim: int, attn_head_dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attn_head_dim = attn_head_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(hidden_dim, attn_head_dim, num_heads, eps)
        self.cross_attn = CrossAttention(
            hidden_dim, attn_head_dim, num_heads, eps)
        self.norm1 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(hidden_dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, hidden_dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, hidden_dim) / hidden_dim**0.5)
        self.gate = GateModule()

    def forward(self, x, context, t_mod, freqs, context_mask=None, self_attn_mask: Optional[torch.Tensor] = None):
        if context_mask is not None and context_mask.dim() == 3:
            context_mask = context_mask.unsqueeze(1) # (B, 1, seq_len, context_len), 1 for heads
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=chunk_dim)
        if has_seq:
            # means t_mod has separate modulation for each token, otherwise same modulation for all tokens in the block
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2),
                shift_mlp.squeeze(2), scale_mlp.squeeze(2), gate_mlp.squeeze(2),
            )
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = self.gate(x, gate_msa, self.self_attn(input_x, freqs, self_attn_mask=self_attn_mask))
        x = x + self.cross_attn(self.norm3(x), context, ctx_mask=context_mask)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        return x


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, has_pos_emb=False):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        if len(t_mod.shape) == 3:
            shift, scale = (self.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device) + t_mod.unsqueeze(2)).chunk(2, dim=2)
            x = (self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2)))
        else:
            shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + scale) + shift))
        return x


class WanVideoDiT(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        attn_head_dim: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
        seperated_timestep: bool = False,
        require_vae_embedding: bool = False,
        require_clip_embedding: bool = False,
        fuse_vae_embedding_in_latents: bool = True,
        action_conditioned: bool = False,
        action_dim: int = 7,
        action_group_causal_mask_mode = "causal",
        video_attention_mask_mode: str = "bidirectional",
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.in_dim = in_dim
        self.freq_dim = freq_dim
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.seperated_timestep = seperated_timestep
        self.require_vae_embedding = require_vae_embedding
        self.require_clip_embedding = require_clip_embedding
        self.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents
        self.video_attention_mask_mode = str(video_attention_mask_mode)

        if num_heads <= 0:
            raise ValueError(f"`num_heads` must be > 0, got {num_heads}")
        if attn_head_dim <= 0:
            raise ValueError(f"`attn_head_dim` must be > 0, got {attn_head_dim}")
        if attn_head_dim % 2 != 0:
            raise ValueError(
                f"`attn_head_dim` must be even for RoPE, got {attn_head_dim}"
            )
        
        self.action_conditioned = action_conditioned
        self.action_dim = action_dim
        assert has_image_input == False
        assert require_clip_embedding == False
        assert require_vae_embedding == False and fuse_vae_embedding_in_latents == True, "Only support fusing vae embedding in latents"

        self.patch_embedding = nn.Conv3d(
            in_dim, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, attn_head_dim, num_heads, ffn_dim, eps)
            for _ in range(num_layers)
        ])
        self.head = Head(hidden_dim, out_dim, patch_size, eps)
        self.freqs = precompute_freqs_cis_3d(attn_head_dim)
        if has_ref_conv:
            self.ref_conv = nn.Conv2d(16, hidden_dim, kernel_size=(2, 2), stride=(2, 2))
        self.has_image_pos_emb = has_image_pos_emb
        self.has_ref_conv = has_ref_conv
        self.control_adapter = None

        if self.action_conditioned:
            self.action_embedding = nn.Linear(action_dim, hidden_dim)
            self.action_group_causal_mask_mode = action_group_causal_mask_mode
        
        self.use_gradient_checkpointing = use_gradient_checkpointing
        if self.use_gradient_checkpointing:
            logger.info("Using gradient checkpointing for DiT blocks. This will save memory but use more computation.")
            

    def patchify(self, x: torch.Tensor, control_camera_latents_input: Optional[torch.Tensor] = None):
        x = self.patch_embedding(x)
        if self.control_adapter is not None and control_camera_latents_input is not None:
            y_camera = self.control_adapter(control_camera_latents_input)
            x = [u + v for u, v in zip(x, y_camera)]
            x = x[0].unsqueeze(0)
        return x

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        f, h, w = grid_size[0], grid_size[1], grid_size[2]
        px, py, pz = self.patch_size[0], self.patch_size[1], self.patch_size[2]
        B = x.shape[0]
        c = x.shape[-1] // (px * py * pz)
        x = x.view(B, f, h, w, px, py, pz, c)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
        x = x.view(B, c, f * px, h * py, w * pz)
        return x

    def _validate_forward_inputs(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor],
        action: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 5:
            raise ValueError(f"`latents` must be 5D [B, C, T, H, W], got shape {tuple(x.shape)}")
        num_latent_frames = x.shape[2]
        if context.ndim != 3:
            raise ValueError(f"`context` must be 3D [B, L, D], got shape {tuple(context.shape)}")
        if timestep.ndim != 1:
            raise ValueError(f"`timestep` must be 1D [B] or [1], got shape {tuple(timestep.shape)}")
        if self.action_conditioned:
            allow_text_only_single_frame = (num_latent_frames == 1 and action is None)
            if not allow_text_only_single_frame:
                assert action is not None, "Action input is required for action-conditioned model."
                if action.ndim != 3:
                    raise ValueError(f"`action` must be 3D [B, action_horizon, action_dim], got shape {tuple(action.shape)}")
                if action.shape[2] != self.action_dim:
                    raise ValueError(f"`action` last dimension must be {self.action_dim}, got {action.shape[2]}")
                if num_latent_frames <= 1:
                    raise ValueError(f"video length must be > 1 for action-conditioned model, got {num_latent_frames}")
                if action.shape[1] % (num_latent_frames - 1) != 0:
                    raise ValueError(
                        f"action horizon must be divisible by (num_latent_frames - 1), got action_horizon={action.shape[1]}"
                    )
        if context_mask is None:
            context_mask = torch.ones((context.shape[0], context.shape[1]), dtype=torch.bool, device=context.device)
        else:
            if context_mask.ndim != 2:
                raise ValueError(f"`context_mask` must be 2D [B, L], got shape {tuple(context_mask.shape)}")
            if context_mask.shape[0] != context.shape[0] or context_mask.shape[1] != context.shape[1]:
                raise ValueError(f"`context_mask` shape must match `context` shape [B, L], got {tuple(context_mask.shape)} vs {tuple(context.shape)}")

        batch_size = x.shape[0]
        if batch_size != context.shape[0]:
            if not self.training and batch_size == 1:
                x = x.expand(context.shape[0], -1, -1, -1, -1)
                batch_size = context.shape[0]
            else:
                raise ValueError(
                    f"Batch mismatch between latents and context: {batch_size} vs {context.shape[0]}."
                )

        if timestep.shape[0] not in (1, batch_size):
            raise ValueError(
                f"`timestep` length must be 1 or batch_size({batch_size}), got {timestep.shape[0]}"
            )
        if timestep.shape[0] == 1 and batch_size > 1:
            assert not self.training, "During training, timestep length must match batch_size."
            timestep = timestep.expand(batch_size)
        return x, timestep, context_mask

    def build_video_to_video_mask(
        self,
        video_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        if video_seq_len <= 0:
            raise ValueError(f"`video_seq_len` must be positive, got {video_seq_len}")
        if video_tokens_per_frame <= 0:
            raise ValueError(f"`video_tokens_per_frame` must be positive, got {video_tokens_per_frame}")

        if self.video_attention_mask_mode == "bidirectional":
            return torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)

        if self.video_attention_mask_mode == "per_frame_causal":
            if video_seq_len % video_tokens_per_frame != 0:
                raise ValueError(
                    "`video_seq_len` must be divisible by `video_tokens_per_frame` in `per_frame_causal` mode, "
                    f"got {video_seq_len} and {video_tokens_per_frame}"
                )
            num_video_frames = video_seq_len // video_tokens_per_frame
            frame_causal = torch.tril(
                torch.ones((num_video_frames, num_video_frames), dtype=torch.bool, device=device)
            )
            return frame_causal.repeat_interleave(video_tokens_per_frame, dim=0).repeat_interleave(
                video_tokens_per_frame, dim=1
            )

        if self.video_attention_mask_mode == "first_frame_causal":
            video_mask = torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)
            first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
            video_mask[:first_frame_tokens, first_frame_tokens:] = False
            return video_mask

        raise ValueError(f"Unsupported video attention mask mode: {self.video_attention_mask_mode}")

    def pre_dit(
        self,
        x: torch.Tensor,                              # 输入潜空间视频张量,形状 (B, C, T, H, W)
        timestep: torch.Tensor,                       # 扩散时间步,形状 (B,),每样本一个标量
        context: torch.Tensor,                        # 文本条件嵌入,形状 (B, L, D_text)
        context_mask: Optional[torch.Tensor] = None,  # 文本 padding/有效 token 的 mask,形状 (B, L)
        action: Optional[torch.Tensor] = None,       # 动作序列,形状 (B, action_len, D_action);可选
        fuse_vae_embedding_in_latents: bool = False,  # 是否把首帧 VAE 嵌入 fuse 到 latents(条件化模式)
        control_camera_latents_input: Optional[torch.Tensor] = None,  # 相机控制信号(controlnet 风格)
    ) -> Dict[str, Any]:                             # 返回一个 dict,供主 forward 循环使用
        # ---------- 输入校验 ----------
        # _validate_forward_inputs 会做以下检查:
        #   - x 必须 5D [B, C, T, H, W]
        #   - timestep 必须 1D [B]
        #   - context 必须 3D [B, L, D]
        #   - 当 action_conditioned=True 时,action 必须有合法形状
        # 同时会把 context_mask 规范化为 (B, L) 的 bool 张量
        x, timestep, context_mask = self._validate_forward_inputs(
            x=x,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            action=action,
        )

        # ---------- 1) 计算每帧的 token 数 ----------
        batch_size = x.shape[0]                        # 批大小 B
        patch_h = int(self.patch_size[1])              # H 方向每个 patch 占据多少潜空间像素
        patch_w = int(self.patch_size[2])              # W 方向每个 patch 占据多少潜空间像素
        # 强约束: 潜空间 H/W 必须能被 patch 整除,否则 patchify 后出现非整尾
        if x.shape[3] % patch_h != 0 or x.shape[4] % patch_w != 0:
            raise ValueError(
                "Latent spatial shape must be divisible by DiT patch size, "
                f"got HxW=({x.shape[3]}, {x.shape[4]}), patch=({patch_h}, {patch_w})"
            )
        # tokens_per_frame = 一帧切出的 patch 总数 = (H/ph) * (W/pw)
        # 例: H=30, W=52, patch=(1,2,2) -> 15*26 = 390
        # 后续所有 mask 形状推导都依赖这个中间量
        tokens_per_frame = (x.shape[3] // patch_h) * (x.shape[4] // patch_w)

        # ---------- 2) 逐 token 的 Separated Timestep 嵌入 ----------
        # 这是 FastWAM 的核心特性: 不同帧可以有不同的 timestep
        # 进入该分支的两个条件:
        #   - seperated_timestep=True: 构造 DiT 时显式开启
        #   - fuse_vae_embedding_in_latents=True: 首帧是干净条件,需"从 t=0 起步"做扩散
        if self.seperated_timestep and fuse_vae_embedding_in_latents:
            # 防御性检查 patch_size 形状是否正确
            if not hasattr(self, "patch_size") or len(self.patch_size) < 3:
                raise ValueError(f"Invalid dit.patch_size: {getattr(self, 'patch_size', None)}")

            # 构造一个逐 token 的 timestep 张量,形状 (B, T, tokens_per_frame)
            # 全部初始化为该样本的 timestep 标量 (broadcast)
            token_timesteps = torch.ones( 
                (batch_size, x.shape[2], tokens_per_frame),
                dtype=timestep.dtype,
                device=timestep.device,
            ) * timestep.view(batch_size, 1, 1)
            # 关键一行: 第 0 帧(通常是 fuse 进去的干净 VAE 条件帧)的 timestep 强制置 0
            # 设计意图: 首帧作为"条件",其扩散前向/反向都从 t=0 出发,不参与加噪去噪,作为静态视觉锚点
            # 后续 T-1 帧仍按 timestep 去噪,实现"以首帧为条件生成后续帧"
            token_timesteps[:, 0, :] = 0
            # 拉平为 (B, T * tokens_per_frame),与接下来进入主循环的视频 token 序列一一对应
            token_timesteps = token_timesteps.reshape(batch_size, -1)
            # 对每个 token 的 timestep 单独做正弦位置式编码
            # 输入 (B*T*tokens_per_frame,), 输出 (B*T*tokens_per_frame, freq_dim)
            # 这样不同位置的 token 拿到不同的时间嵌入,实现 per-token 的 adaLN 调制
            token_t_emb = sinusoidal_embedding_1d(self.freq_dim, token_timesteps.reshape(-1))
            # 经 MLP 投影到 hidden_dim,再 reshape 为 (B, T*tokens_per_frame, hidden_dim)
            t = self.time_embedding(token_t_emb).reshape(batch_size, -1, self.hidden_dim)
            # 投影出 6 组调制向量,形状 (B, seq_len, 6, hidden_dim)
            # 对应 DiT 中 adaln-zero 的 (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
            # 在每个 block 中按需拆出来对 Q/K/V 与 MLP 做调制
            t_mod = self.time_projection(t).unflatten(2, (6, self.hidden_dim))
        else:
            # 旧逻辑(整段视频共享 1 个 timestep)被显式禁用
            # 即使走到 else,raise 已终止;下面两行是死代码,保留作历史参考
            raise NotImplementedError("Only support seperated_timestep with fuse_vae_embedding_in_latents for now.")
            t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
            t_mod = self.time_projection(t).unflatten(1, (6, self.hidden_dim))

        # ---------- 3) Patchify: 把潜空间切成 token 序列 ----------
        # self.patch_embedding 是 Conv3d(in_C, hidden_dim, kernel=patch_size, stride=patch_size)
        # 输出形状 (B, hidden_dim, T, H/ph, W/pw)
        # 若有 control_adapter 且给了相机控制信号,会注入控制条件(controlnet 风格),形状不变
        x = self.patchify(x, control_camera_latents_input=control_camera_latents_input)
        # 解构出 f, h, w: patch 化之后的"时空网格三个维度"
        # 注意此处的 f 已是 patch 网格的"帧数"(通常等于 T),不是原始像素帧数
        f, h, w = x.shape[2:]

        # ---------- 4) Context 构造: 文本 + 可选动作 + 分组因果 mask ----------
        # 把原始文本特征(通常来自 T5 / UMT5)投影到 hidden_dim
        context = self.text_embedding(context) # (B, L, dim)
        context_len = context.shape[1]         # 文本 token 数(已过 text embed,不是原始字符长度)

        # ---- 情形 A: 有动作输入(action-conditioned + action 不为空) ----
        if self.action_conditioned and action is not None:
            # 取动作序列长度,经过 self.action_embedding(通常是一个 MLP)把动作投影到 hidden_dim
            action_len = action.shape[1]
            action_emb = self.action_embedding(action) # (B, action_len, dim)
            # 给每个动作 token 加 1D 正弦位置编码
            # 让模型区分"第几个动作"以及捕捉动作序列内部的时序关系
            # unsqueeze(0) 把 (action_len, dim) -> (1, action_len, dim) 以便 broadcast 加到 batch 维
            action_pos_embed = sinusoidal_embedding_1d(self.hidden_dim,
                torch.arange(action_len, device=action_emb.device)) # (action_len, dim)
            action_emb = action_emb + action_pos_embed.unsqueeze(0) # (B, action_len, dim)
            # 把动作 token 拼到文本 token 之后
            # 后续跨注意力的 key/value 序列就是"前 L 个文本 + 后 action_len 个动作"
            context = torch.cat([context, action_emb], dim=1) # (B, context_len + action_len, dim)

            # ---- 构造"分组因果 mask" ----
            # 关键设计: 把 patch 化后的视频分成 f 段,对应原始 T 个 latent 帧
            # 第 0 帧是干净条件帧(由 token_timesteps[:, 0, :] = 0 标记),不参与动作交叉注意力
            # 因此"需要看动作"的时间组数 = f - 1
            num_temporal_groups = f - 1 # first latent frame do not attend to actions
            # 必须至少有 2 个 latent 帧(1 条件 + 至少 1 生成),否则提供 action 没有意义
            if num_temporal_groups <= 0:
                raise ValueError(
                    "Action-conditioned context mask requires at least 2 latent frames when `action` is provided."
                )
            # 约束: 总动作 token 数必须能被"生成帧数 f-1"整除
            # 即每个生成帧平均分配到相同数量的动作 token
            # 例: f=4(1 条件 + 3 生成), action_len=30 -> 每生成帧 10 个动作 token
            assert action_emb.shape[1] % num_temporal_groups == 0, \
                f"Action embedding length {action_emb.shape[1]} must be divisible by number of temporal groups {num_temporal_groups}"
            # 调用工厂函数 create_group_causal_attn_mask(64-144 行)生成布尔 mask
            # 形状: (num_temporal_groups * tokens_per_frame, num_temporal_groups * key_per_group)
            #     = ((f-1)*tokens_per_frame, action_len)
            # 两种 mode:
            #   "causal": 第 i 帧所有 patch query 可 attend 第 0..i 帧对应组的所有动作 key(时间因果)
            #   "group_diagonal": 第 i 帧 query 只能 attend 第 i 帧对应组的那一小段动作 key(严格 1-to-1)
            action_group_mask = create_group_causal_attn_mask(
                num_temporal_groups=num_temporal_groups,
                num_query_per_group=tokens_per_frame,
                num_key_per_group=action_len // num_temporal_groups,
                mode=self.action_group_causal_mask_mode,
            ).to(context.device) # ((f-1)*tokens_per_frame, action_len)

            # ---- 拼装最终 mask ----
            # seq_len = patch 化后所有视频 token 数 = f * h * w
            seq_len = f * h * w # query length
            # 初始化全 False 的三维 bool mask,形状 (B, seq_len, L + action_len)
            #   维度 1 (query): 视频 token 序列
            #   维度 2 (key):   文本 + 动作 token 序列
            #   False = 不可 attend, True = 可 attend
            final_context_mask = torch.zeros((batch_size, seq_len, context.shape[1]), dtype=torch.bool, device=context.device) # (B, seq_len, L + action_len)
            # 文本部分: 每个视频 token(含第 0 条件帧)都可 attend 文本,沿用原 context_mask
            # unsqueeze(1) 把 (B, L) -> (B, 1, L), 再 expand 到 (B, seq_len, L)
            # 写入 mask 的前 context_len 列(即"文本段"对应的 key 列)
            final_context_mask[:, :, :context_len] = context_mask.unsqueeze(1).expand(-1, seq_len, -1) # (B, seq_len, L)
            # 动作部分: 只让第 1..f-1 帧(query 索引 tokens_per_frame 到 seq_len)attend 动作
            # [:, tokens_per_frame:, :] 切片跳过第 0 帧的 tokens_per_frame 个 query token
            # 写入 mask 的后 action_len 列(即"动作段"对应的 key 列)
            # 第 0 帧 query 在动作段上保持 False(由全 0 初始化),符合"条件帧不与动作耦合"的设计
            final_context_mask[:, tokens_per_frame:, context_len:] = action_group_mask.unsqueeze(0).expand(batch_size, -1, -1) # (B, seq_len, action_len)
            # 用新 mask 替换原 context_mask,供后续 CrossAttention 使用
            context_mask = final_context_mask

        # ---- 情形 B: action-conditioned 但没给 action(单帧文本模式) ----
        elif self.action_conditioned and action is None:
            # 仅在单 latent 帧模式下允许"无动作"推理(等价于纯文本 -> 单帧图像生成)
            # f != 1 时拒绝: 多帧生成无动作无意义(模型没学过该分布)
            if f != 1:
                raise ValueError(
                    "Action-conditioned model requires `action` unless running single-frame text-only mode with num_latent_frames=1."
                )
            # mask 形状 (B, 1, L) -> expand 成 (B, seq_len, L),让所有视频 token attend 文本
            context_mask = context_mask.unsqueeze(1).expand(-1, f * h * w, -1) # (B, seq_len, L)

        # ---- 情形 C: 非 action-conditioned(纯文生视频) ----
        else:
            # 标准文生视频路径,直接把 (B, L) 广播成 (B, seq_len, L),无动作参与
            context_mask = context_mask.unsqueeze(1).expand(-1, f * h * w, -1) # (B, seq_len, L)

        # ---------- 5) 重排 video tokens 为 Transformer 期望的序列形式 ----------
        # patchify 后的 x 形状 (B, hidden_dim, f, h, w)
        # 重排为 (B, f*h*w, hidden_dim),即 Transformer 标准的 (B, seq_len, dim)
        # .contiguous() 保证内存连续,后续 attention/linear 算子不会因 stride 异常回退到慢路径
        x_tokens = x.permute(0, 2, 3, 4, 1).reshape(x.shape[0], -1, x.shape[1]).contiguous()

        # ---------- 6) 构造 3D RoPE 频率表 ----------
        # self.freqs 是在 DiT 构造时预计算好的 3 组 1D RoPE 频率(complex 张量)
        # 分别对应 T/H/W 三轴(由 precompute_freqs_cis_3d 生成,见 38-52 行)
        # 取各自的前 f/h/w 个位置(因为当前 batch 的网格可能比预计算的最大长度小)
        freqs = torch.cat([
            # T 轴: (f, 1, 1, d_t) -> (f, h, w, d_t),T 轴频率在 (h, w) 上复制
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            # H 轴: (1, h, 1, d_h) -> (f, h, w, d_h),H 轴频率在 (f, w) 上复制
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            # W 轴: (1, 1, w, d_w) -> (f, h, w, d_w),W 轴频率在 (f, h) 上复制
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x_tokens.device)
        # cat 后形状 (f, h, w, d_t + d_h + d_w),即"全网格 RoPE 频率表"
        # reshape(f*h*w, 1, -1) 拉平为 token 序列形式
        # 前缀维度 1 用于后续 rope_apply 的 broadcast 到多头
        # .to(x_tokens.device) 把频率张量搬到和 video tokens 同一设备
        # (初始化时可能落在 CPU 上)

        # ---------- 7) 返回 pre_state ----------
        # 字典中的所有 key 都会在 forward 的主循环(628-666 行)里被消费:
        #   tokens         -> self-attention 的 QKV 输入
        #   freqs          -> SelfAttention 中通过 rope_apply 应用到 Q/K
        #   t_mod          -> 在每个 WanAttentionBlock 里被拆成 6 份,做 adaLN 调制
        #   context + mask -> 喂给 CrossAttention
        #   meta.grid_size -> 用于 post_dit 的 unpatchify
        #   meta.tokens_per_frame -> 用于 build_video_to_video_mask(self-attn 的时序 mask)
        return {
            "tokens": x_tokens,            # (B, f*h*w, hidden_dim)         patch 化后的视频序列
            "freqs": freqs,                # (f*h*w, 1, rope_dim)            3D RoPE 频率
            "t": t,                        # (B, f*h*w, hidden_dim)          逐 token 的时间嵌入
            "t_mod": t_mod,                # (B, f*h*w, 6, hidden_dim)       adaln-zero 6 组调制向量
            "context": context,            # (B, L+action_len, hidden_dim)   文本(+动作)条件序列
            "context_mask": context_mask,  # (B, f*h*w, L+action_len)        跨注意力 mask
            "meta": {
                "grid_size": (f, h, w),    # patch 网格,供 post_dit unpatchify 使用
                "tokens_per_frame": tokens_per_frame,  # 每帧 patch 数,供 video self-attn mask 使用
                "batch_size": batch_size,  # 批大小,便于 debug / reshape
            },
        }

    def post_dit(self, x_tokens: torch.Tensor, pre_state: Dict[str, Any]) -> torch.Tensor:
        f, h, w = pre_state["meta"]["grid_size"]
        x = self.head(x_tokens, pre_state["t"])
        x = self.unpatchify(x, (f, h, w))
        return x

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        fuse_vae_embedding_in_latents: bool = False,
    ):
        pre_state = self.pre_dit(
            x=x,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        x_tokens = pre_state["tokens"]
        context_emb = pre_state["context"]
        t_mod = pre_state["t_mod"]
        freqs = pre_state["freqs"]
        context_attn_mask = pre_state["context_mask"]
        self_attn_mask = self.build_video_to_video_mask(
            video_seq_len=x_tokens.shape[1],
            video_tokens_per_frame=int(pre_state["meta"]["tokens_per_frame"]),
            device=x_tokens.device,
        ) if self.video_attention_mask_mode != "bidirectional" else None # special rule for faster speed

        for block in self.blocks:
            if self.use_gradient_checkpointing:
                x_tokens = gradient_checkpoint_forward(
                    block,
                    self.use_gradient_checkpointing,
                    x_tokens, context_emb, t_mod, freqs, context_mask=context_attn_mask, self_attn_mask=self_attn_mask
                )
            else:
                x_tokens = block(x_tokens, context_emb, t_mod, freqs, context_mask=context_attn_mask, self_attn_mask=self_attn_mask)

        return self.post_dit(x_tokens, pre_state)
