import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# --- Norms & activations ---

# LayerNorm for channel-first (B, C, H, W) tensors
class LayerNorm2D(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = rearrange(x, "b c h w -> b h w c")
        x = self.norm(x)
        x = rearrange(x, "b h w c -> b c h w")

        return(x)


# Blocks adapted from MetaFormer
# https://github.com/sail-sg/metaformer/blob/main/metaformer_baselines.py

class StarReLU(nn.Module):
    """
    s * relu(x)**2 + b
    """
    def __init__(self, scale_value=1.0, bias_value=0.0,
        scale_learnable=True, bias_learnable=True,
        mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
            requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
            requires_grad=bias_learnable)
    def forward(self, x):
        return self.scale * self.relu(x)**2 + self.bias


# --- Basic blocks ---

class Scale(nn.Module):
    """
    Learnable per-channel scale.
    """
    def __init__(self, dim, init_value=1.0, trainable=True):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        return x * self.scale


class MLP(nn.Module):
    def __init__(self,
                 embed_dim: int = 64,
                 expand_factor: int = 2,
                 dropout: float = 0.,
                 act: torch.nn.Module = nn.GELU
                 ):
        super().__init__()

        self.embed_dim = embed_dim
        self.inner_dim = embed_dim * expand_factor
        self.dropout = dropout

        self.fc1 = nn.Linear(self.embed_dim, self.inner_dim)
        self.act = act()
        self.drop1 = nn.Dropout(p=self.dropout)
        self.fc2 = nn.Linear(self.inner_dim, self.embed_dim)
        self.drop2 = nn.Dropout(p=self.dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)

        return x


class SepConv(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2.
    Optional inverted residual via `residual=True`.
    """
    def __init__(self, dim, expansion_ratio=2,
        act1_layer=StarReLU, act2_layer=nn.Identity,
        bias=False, kernel_size=7, padding=3, residual = False,
        **kwargs,):
        super().__init__()
        self.residual = residual
        med_channels = int(expansion_ratio * dim)
        self.pwconv1 = nn.Linear(dim, med_channels, bias=bias)
        self.act1 = act1_layer()
        self.dwconv = nn.Conv2d(
            med_channels, med_channels, kernel_size=kernel_size,
            padding=padding, groups=med_channels, bias=bias) # depthwise conv
        self.act2 = act2_layer()
        self.pwconv2 = nn.Linear(med_channels, dim, bias=bias)

    def forward(self, x):
        shortcut = x

        x = self.pwconv1(x)
        x = self.act1(x)

        x = x.permute(0, 3, 1, 2) # (B, H, W, C) -> (B, C, H, W)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (B, C, H, W) -> (B, H, W, C)

        x = self.act2(x)
        x = self.pwconv2(x)
        x = x + shortcut if self.residual else x
        return x


class Downsampling(nn.Module):
    """
    Conv + norm downsampling layer. @torch.compiler.disable applied due to dynamic shapes.
    """
    def __init__(self, in_channels, out_channels,
        kernel_size, stride=1, padding=0,
        pre_norm=None, post_norm =None):
        super().__init__()
        self.pre_norm = pre_norm(in_channels) if pre_norm else nn.Identity()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.post_norm = post_norm(out_channels) if post_norm else nn.Identity()

    @torch.compiler.disable
    def forward(self, x):
        x = self.pre_norm(x)
        x = self.conv(x)
        x = self.post_norm(x)
        return x


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.view(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def rope_cos_sin_1d(pos: torch.Tensor, dim: int, base: float, out_dtype: torch.dtype, device: torch.device):
    """
    pos: [N] (int or float)
    returns cos,sin: [N, dim] in out_dtype
    """
    assert dim % 2 == 0
    # compute in fp32 for stability, cast down at end
    pos_f = pos.to(device=device, dtype=torch.float32)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim // 2, device=device, dtype=torch.float32) / (dim // 2)))
    freqs = pos_f[:, None] * inv_freq[None, :]  # [N, dim/2]
    cos = freqs.cos().repeat_interleave(2, dim=-1).to(out_dtype)
    sin = freqs.sin().repeat_interleave(2, dim=-1).to(out_dtype)
    return cos, sin


def apply_rope_1d(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    x: [B, heads, N, d]
    cos/sin: [N, d]
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1,1,N,d]
    sin = sin.unsqueeze(0).unsqueeze(0)
    return x * cos + rotate_half(x) * sin


class RoPEAttention2D(nn.Module):
    """
    Self-attention with 2D RoPE. Expects x as [B, H, W, C].
    If has_cls=True, prepends a CLS token dimension before flattening.
    """

    def __init__(
        self,
        dim: int,
        head_dim: int = 32,
        num_heads: int | None = None,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        proj_bias: bool = False,
        rope_base: float = 10000.0,
        rope_ratio: float = 1.0,   # fraction of per-head dims used for RoPE
        has_cls: bool = False,
        use_sdpa: bool = True,
        is_causal: bool = False,
    ):
        super().__init__()

        self.head_dim = head_dim
        self.num_heads = num_heads if num_heads is not None else dim // head_dim
        if self.num_heads == 0:
            self.num_heads = 1

        self.attention_dim = self.num_heads * self.head_dim
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, 3 * self.attention_dim, bias=qkv_bias)
        self.proj = nn.Linear(self.attention_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.attn_drop = attn_drop  # passed as float to SDPA
        self.use_sdpa = use_sdpa
        self.is_causal = is_causal

        self.rope_base = rope_base
        self.has_cls = has_cls

        # Per-head rope chunk sizes (even)
        rope_dim_total = int(self.head_dim * rope_ratio)
        half = rope_dim_total // 2
        self.h_dim = (half // 2) * 2
        self.w_dim = (half // 2) * 2
        self.rope_dim = self.h_dim + self.w_dim
        if self.rope_dim == 0:
            raise ValueError(
                f"rope_ratio={rope_ratio} too small for head_dim={self.head_dim}. Need >=2 dims per axis."
            )
        if self.rope_dim > self.head_dim:
            raise ValueError("rope_dim exceeds head_dim")

        # qk-norm per head dim (applied after RoPE)
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)

    @staticmethod
    def _hw_positions(H_patches: int, W_patches: int, device: torch.device):
        h = torch.arange(H_patches, device=device).repeat_interleave(W_patches)  # [H*W]
        w = torch.arange(W_patches, device=device).repeat(H_patches)             # [H*W]
        return h, w

    def forward(self, x: torch.Tensor, attn_mask=None):
        """
        x: [B, H, W, C]
        """
        B, H, W, C = x.shape

        N = H * W
        n_patches = H * W
        if self.has_cls:
            assert N == 1 + n_patches, f"Expected N=1+H*W={1+n_patches}, got N={N}"
        else:
            assert N == n_patches, f"Expected N=H*W={n_patches}, got N={N}"

        # qkv: [B, N, 3*attn_dim] -> [3, B, heads, N, head_dim]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # [B, heads, N, head_dim]

        # split CLS vs patches
        if self.has_cls:
            q_cls, q_p = q[:, :, :1, :], q[:, :, 1:, :]
            k_cls, k_p = k[:, :, :1, :], k[:, :, 1:, :]
        else:
            q_cls = k_cls = None
            q_p, k_p = q, k

        # positions for patches
        h_pos, w_pos = self._hw_positions(H, W, device=x.device)  # [n_patches]

        # apply RoPE on patch tokens only
        s = 0
        qh = q_p[..., s : s + self.h_dim]
        kh = k_p[..., s : s + self.h_dim]
        cos_h, sin_h = rope_cos_sin_1d(h_pos, self.h_dim, self.rope_base, qh.dtype, qh.device)
        qh = apply_rope_1d(qh, cos_h, sin_h)
        kh = apply_rope_1d(kh, cos_h, sin_h)
        s += self.h_dim

        qw = q_p[..., s : s + self.w_dim]
        kw = k_p[..., s : s + self.w_dim]
        cos_w, sin_w = rope_cos_sin_1d(w_pos, self.w_dim, self.rope_base, qw.dtype, qw.device)
        qw = apply_rope_1d(qw, cos_w, sin_w)
        kw = apply_rope_1d(kw, cos_w, sin_w)
        s += self.w_dim

        if s < self.head_dim:
            q_p = torch.cat([qh, qw, q_p[..., s:]], dim=-1)
            k_p = torch.cat([kh, kw, k_p[..., s:]], dim=-1)
        else:
            q_p = torch.cat([qh, qw], dim=-1)
            k_p = torch.cat([kh, kw], dim=-1)

        # recombine
        if self.has_cls:
            q = torch.cat([q_cls, q_p], dim=2)
            k = torch.cat([k_cls, k_p], dim=2)
        else:
            q, k = q_p, k_p

        # qk-norm after RoPE (LayerNorm)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # attention (SDPA recommended)
        if self.use_sdpa:
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop if self.training else 0.0,
                attn_mask=attn_mask,
                is_causal=self.is_causal,
                scale=self.scale,
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = F.dropout(attn, p=self.attn_drop, training=self.training)
            out = attn @ v

        out = out.transpose(1, 2).reshape(B, H, W, self.attention_dim)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


# --- Head ---

class MlpHead(nn.Module):
    """ MLP classification head
    """
    def __init__(self,
                 dim: int,
                 n_classes: int = 1000,
                 mlp_ratio: int = 4,
                 head_dropout: float = 0.,
                 bias: bool =True):
        super().__init__()
        hidden_features = int(mlp_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden_features, bias=bias)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(hidden_features)
        self.fc2 = nn.Linear(hidden_features, n_classes, bias=bias)
        self.head_dropout = nn.Dropout(head_dropout)


    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.head_dropout(x)
        x = self.fc2(x)
        return x
