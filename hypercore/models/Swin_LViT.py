from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from geoopt import ManifoldParameter
from .. import nn as hnn


def window_partition(x: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """
    Partition input into non-overlapping windows.
    
    Args:
        x: Input tensor of shape (B, H*W, C) on Lorentz manifold
        window_size: Size of the window
        H: Height of the feature map
        W: Width of the feature map
        
    Returns:
        Windows tensor of shape (num_windows * B, window_size*window_size, C)
    """
    B, N, C = x.shape
    assert N == H * W, "Token count must be H*W"
    x = x.view(B, H, W, C)
    # [B, H//Ws, Ws, W//Ws, Ws, C] -> windows as batch
    x = x.view(B,
               H // window_size, window_size,
               W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int, B: int) -> torch.Tensor:
    """
    Reverse window partition to restore original shape.
    
    Args:
        windows: Windows tensor of shape (num_windows*B, window_size*window_size, C)
        window_size: Size of the window
        H: Height of the feature map
        W: Width of the feature map
        B: Batch size
        
    Returns:
        Tensor of shape (B, H*W, C)
    """
    nW = (H // window_size) * (W // window_size)
    _, L, C = windows.shape
    assert L == window_size * window_size
    x = windows.view(B,
                     H // window_size, W // window_size,
                     window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C)
    return x.view(B, H * W, C)


class HyperbolicMLP(nn.Module):
    """Hyperbolic MLP with two linear layers, activation, and dropout."""
    def __init__(self, manifold, in_channel: int, hidden_channel: int, dropout: float = 0.0):
        super().__init__()
        self.manifold = manifold
        self.dense_1 = hnn.LorentzLinear(self.manifold, in_channel, hidden_channel - 1)
        self.dense_2 = hnn.LorentzLinear(self.manifold, hidden_channel, in_channel - 1)
        self.activation = hnn.LorentzActivation(manifold, activation=nn.GELU())
        self.dropout = hnn.LorentzDropout(self.manifold, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dense_2(x)
        x = self.dropout(x)
        return x

def build_rope_2d(window_size: int, head_spatial_dim: int, device: torch.device) -> torch.Tensor:
    """
    Build 2D Rotary Position Embeddings (RoPE) for a window.
    
    Creates complex-valued frequency bases for 2D position encoding:
    - Splits frequency channels between y and x dimensions
    - Uses inverse frequency ladder (1/10000^(i/d)) for each dimension
    
    Args:
        window_size: Width/height of the attention window
        head_spatial_dim: Spatial dimension per attention head (must be even)
        device: torch device for tensor allocation
        
    Returns:
        Complex tensor of shape [L=window_size^2, head_spatial_dim//2]
    """
    assert head_spatial_dim % 2 == 0, "per-head spatial dim (out_channels-1) must be even"
    W = window_size
    L = W * W

    # number of complex pairs (each rotates a 2-d real pair)
    n_pairs = head_spatial_dim // 2
    # split those pairs between y and x
    n_y = n_pairs // 2
    n_x = n_pairs - n_y

    ys, xs = torch.meshgrid(
        torch.arange(W, device=device), torch.arange(W, device=device), indexing='ij'
    )
    posy = ys.reshape(L, 1).float()
    posx = xs.reshape(L, 1).float()

    # classic RoPE frequency ladders
    # guard against n_y or n_x == 0 (tiny heads)
    def inv_freq(n):
        if n <= 0:
            return torch.empty(0, device=device)
        # like transformer-style 1/10000^(i/d)
        return 1.0 / (10000 ** (torch.arange(0, n, device=device).float() / max(1, n)))

    fy = inv_freq(n_y)  # [n_y]
    fx = inv_freq(n_x)  # [n_x]

    ang_y = posy * fy  # [L, n_y]
    ang_x = posx * fx  # [L, n_x]

    # stack y then x to reach total n_pairs channels
    cos = torch.cat([torch.cos(ang_y), torch.cos(ang_x)], dim=1)  # [L, n_pairs]
    sin = torch.cat([torch.sin(ang_y), torch.sin(ang_x)], dim=1)  # [L, n_pairs]

    freqs = torch.complex(cos, sin)  # [L, n_pairs] complex
    return freqs


class LorentzWindowAttention(nn.Module):
    """
    Lorentz Window Attention with RoPE-based 2D positional encoding.
    
    Args:
        manifold: Lorentz manifold for hyperbolic operations
        per_head_dim: Per-head Lorentz dimension (head_dim_spatial + 1)
        num_heads: Number of attention heads
        window_size: Size of the attention window (WxW)
    """
    # Scale parameter for residual connection
    RESIDUAL_SCALE = 27.5
    
    def __init__(self, manifold, per_head_dim, num_heads, window_size):
        super().__init__()
        self.manifold = manifold
        self.per_head_dim = per_head_dim
        self.num_heads = num_heads
        self.window_size = window_size

        self.attn = hnn.LorentzMultiheadAttention(
            manifold, per_head_dim, per_head_dim, num_heads,
            attention_type='full',
            trans_heads_concat=True
        )
        # LayerNorm on the total spatial dimensions
        self.ln = hnn.LorentzLayerNorm(manifold, num_heads * per_head_dim - 1)
        self.res = hnn.LResNet(manifold, use_scale=True, scale=self.RESIDUAL_SCALE)

        self.register_buffer('_rope_cache_complex', None, persistent=False)
        self._rope_window_size = None
        self._rope_head_dim = None

    def _get_rope(self, device: torch.device) -> torch.Tensor:
        """
        Build or retrieve cached RoPE frequencies.
        
        Args:
            device: Device for tensor allocation
            
        Returns:
            Complex tensor of RoPE frequencies [L, Dsp//2]
        """
        # per-head spatial dim expected by attention after Wq/Wk: (out_channels - 1)
        Dsp = int(self.attn.out_channels - 1)
        assert Dsp % 2 == 0, "Require even (out_channels-1) for RoPE."
        L = self.window_size * self.window_size

        if (self._rope_cache_complex is None or
            self._rope_window_size != self.window_size or
            self._rope_head_dim != Dsp or
            self._rope_cache_complex.device != device):
            self._rope_cache_complex = build_rope_2d(self.window_size, Dsp, device)
            self._rope_window_size = self.window_size
            self._rope_head_dim = Dsp
        return self._rope_cache_complex  # [L, Dsp//2] complex

    def forward(self, x: torch.Tensor, H: int, W: int, output_attentions: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, N, C = x.shape
        h = self.ln(x)

        win_tokens = window_partition(h, self.window_size, H, W) # (nWB, L, C)

        rope = self._get_rope(win_tokens.device) # [L, Dsp//2] complex

        attn_out = self.attn(win_tokens, win_tokens, output_attentions=output_attentions, rot_pos=rope)
        if output_attentions:
            attn_out, attn_weights = attn_out

        y = window_reverse(attn_out, self.window_size, H, W, B)
        y = self.res(x, y)
        return (y, attn_weights) if output_attentions else y


class LorentzSwinBlock(nn.Module):
    """
    A single Swin block with either W-MSA or SW-MSA.
    
    Args:
        manifold: Lorentz manifold for hyperbolic operations
        per_head_dim: Per-head Lorentz dimension (head_dim_spatial + 1)
        hidden_mlp_dim: Hidden dimension for MLP layer
        num_heads: Number of attention heads
        window_size: Size of attention window
        shift_size: Shift size for SW-MSA (0 for W-MSA)
        dropout: Dropout rate
    """
    # Scale parameter for residual connection
    RESIDUAL_SCALE = 27.5
    
    def __init__(self, manifold, per_head_dim, hidden_mlp_dim, num_heads, window_size, shift_size=0, dropout=0.0):
        super().__init__()
        self.manifold = manifold
        self.per_head_dim = per_head_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.attn = LorentzWindowAttention(manifold, per_head_dim, num_heads, window_size)
        self.ln2 = hnn.LorentzLayerNorm(manifold, num_heads * per_head_dim - 1)
        self.mlp = HyperbolicMLP(manifold, num_heads * per_head_dim, hidden_mlp_dim, dropout=dropout)
        self.res2 = hnn.LResNet(manifold, use_scale=True, scale=self.RESIDUAL_SCALE)

    def forward(self, x: torch.Tensor, H: int, W: int, output_attentions: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Input tensor of shape (B, H*W, C)
            H: Height of feature map
            W: Width of feature map
            output_attentions: Whether to return attention weights
            
        Returns:
            If output_attentions is False: output tensor (B, H*W, C)
            If output_attentions is True: tuple of (output tensor, attention weights)
        """
        B, N, C = x.shape
        if self.shift_size > 0:
            x_grid = x.view(B, H, W, C)
            x_grid = torch.roll(x_grid, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            x = x_grid.view(B, H * W, C)

        if output_attentions:
            x, attn_w = self.attn(x, H, W, output_attentions=True)
        else:
            x = self.attn(x, H, W, output_attentions=False)

        if self.shift_size > 0:
            x_grid = x.view(B, H, W, C)
            x_grid = torch.roll(x_grid, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            x = x_grid.view(B, H * W, C)

        y = self.mlp(self.ln2(x))
        x = self.res2(x, y)

        if output_attentions:
            return x, attn_w
        return x



class LorentzPatchMerging(nn.Module):
    """
    2x downsample in H/W; concatenates 2x2 neighbors and projects.
    Doubles the spatial dimensions: num_heads * head_dim_spatial -> 2 * num_heads * head_dim_spatial
    Output total dimension = 2 * num_heads * head_dim_spatial + 1 (one shared time coordinate)
    
    Args:
        manifold: Lorentz manifold for hyperbolic operations
        in_dim: Input Lorentz dimension (including time coordinate)
        out_dim: Output Lorentz dimension (including time coordinate)
    """
    def __init__(self, manifold, in_dim, out_dim):
        super().__init__()
        self.manifold = manifold
        self.in_dim = in_dim
        self.out_dim = out_dim
        # Project from 4*in_dim spatial to out_dim spatial
        # in_dim includes 1 time, so spatial = in_dim - 1
        # After concat: 4 * (in_dim - 1) + 4 = 4*in_dim - 4 + 4 = 4*in_dim spatial (4 time coords concatenated)
        # Actually, each patch has 1 time coord, concat gives 4 patches -> need to handle properly
        # The projection takes concatenated Lorentz points and outputs new Lorentz point
        self.proj = hnn.LorentzLinear(self.manifold, 4 * in_dim, out_dim - 1)

    def forward(self, x: torch.Tensor, H: int, W: int) -> Tuple[torch.Tensor, int, int]:
        """
        Merge 2x2 patches into a single patch with doubled channels.
        
        Args:
            x: Input tensor of shape (B, H*W, C)
            H: Height of the feature map (must be even)
            W: Width of the feature map (must be even)
            
        Returns:
            Tuple of (merged tensor, new_H, new_W)
        """
        B, N, C = x.shape
        assert N == H * W, f"Token count {N} must equal H*W={H*W}"
        assert H % 2 == 0 and W % 2 == 0, f"H={H} and W={W} must be even for patch merging"

        X = x.view(B, H, W, C)
        x00 = X[:, 0::2, 0::2, :] # (B, H/2, W/2, C)
        x01 = X[:, 0::2, 1::2, :]
        x10 = X[:, 1::2, 0::2, :]
        x11 = X[:, 1::2, 1::2, :]
        merged = torch.cat([x00, x01, x10, x11], dim=-1).contiguous() # (B, H/2, W/2, 4C)
        merged = merged.view(B, -1, 4 * C)
        merged = self.proj(merged) # (B, (H/2)*(W/2), out_dim)
        return merged, H // 2, W // 2


class LorentzSwinLayer(nn.Module):
    """
    A single Swin stage containing multiple blocks and optional downsampling.
    
    Args:
        manifold: Lorentz manifold for hyperbolic operations
        per_head_dim: Per-head Lorentz dimension (head_dim_spatial + 1)
        depth: Number of blocks in this stage
        num_heads: Number of attention heads
        window_size: Size of attention window
        mlp_ratio: Ratio of mlp hidden dim to embedding dim
        dropout: Dropout rate
        downsample: Whether to downsample at the end of this stage
    """
    def __init__(self, manifold, per_head_dim, depth, num_heads, window_size, mlp_ratio=4.0, dropout=0.0, downsample=True):
        super().__init__()
        self.blocks = nn.ModuleList()
        hidden_mlp_dim = int(mlp_ratio * num_heads * per_head_dim)
        for i in range(depth):
            shift = 0 if (i % 2 == 0) else window_size // 2
            self.blocks.append(
                LorentzSwinBlock(
                    manifold, per_head_dim, hidden_mlp_dim,
                    num_heads=num_heads, window_size=window_size,
                    shift_size=shift, dropout=dropout
                )
            )
        self.downsample = downsample
        if downsample:
            # Double per-head structure: num_heads * per_head_dim -> 2*num_heads * per_head_dim
            self.merger = LorentzPatchMerging(manifold, num_heads * per_head_dim, 2 * num_heads * per_head_dim)
        else:
            self.merger = None

    def forward(self, x: torch.Tensor, H: int, W: int, output_attentions: bool = False) -> Union[Tuple[torch.Tensor, int, int], Tuple[torch.Tensor, int, int, list]]:
        attns = []
        for blk in self.blocks:
            if output_attentions:
                x, aw = blk(x, H, W, output_attentions=True)
                attns.append(aw)
            else:
                x = blk(x, H, W, output_attentions=False)
        if self.downsample:
            x, H, W = self.merger(x, H, W)
        if output_attentions:
            # stack per-block attention maps if needed by caller
            return x, H, W, attns
        return x, H, W



class LSwinViT(nn.Module):
    """
    Lorentz Swin Vision Transformer (hierarchical).
    
    A hierarchical vision transformer operating on the Lorentz manifold.
    Uses window-based attention with shifted windows (Swin) and hyperbolic geometry.
    
    Note on dimensions:
        - head_dim parameter: Spatial dimensions per attention head (e.g., 64)
        - Total Lorentz dimension: num_heads * head_dim + 1 (shared time coordinate)
        - When trans_heads_concat=True, attention heads share a single time coordinate
    
    Args:
        manifold_in: Manifold for input patch embedding
        manifold_hidden: Manifold for hidden layer operations
        manifold_out: Manifold for output embeddings and classification
        image_size: Input image size (assumed square)
        patch_size: Patch size for initial embedding
        in_channel: Number of input image channels (RGB=3)
        depths: Number of blocks in each stage
        num_heads: Number of attention heads in each stage
        head_dim: Spatial dimension per attention head (NOT including time coordinate)
        window_size: Size of attention window
        mlp_ratio: Ratio of MLP hidden dim to embedding dim
        dropout: Dropout rate
        num_classes: Number of output classes (0 for feature extraction only)
        embed_dim: Final embedding dimension (None to use num_heads[-1] * head_dim + 1)
    """
    def __init__(
        self,
        manifold_in,
        manifold_hidden,
        manifold_out,
        image_size=224,
        patch_size=4,
        in_channel=3,
        depths=(2, 2, 6, 2),
        num_heads=(2, 4, 8, 16),
        head_dim=64,
        window_size=7,
        mlp_ratio=4.0,
        dropout=0.0,
        num_classes=0,
        embed_dim=None
    ):
        super().__init__()
        
        # Validate inputs
        assert len(depths) == len(num_heads), "depths and num_heads must have the same length"
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        
        self.manifold_in = manifold_in
        self.manifold_hidden = manifold_hidden
        self.manifold_out = manifold_out
        self.manifold = manifold_out 
        self.num_classes = num_classes
        self.image_size = image_size
        self.patch_size = patch_size
        self.window_size = window_size
        self.depths = depths
        self.num_heads = num_heads

        # head_dim is the SPATIAL dimension per head (user-specified)
        # Each head operates in its own Lorentz space with dim = head_dim + 1
        # When concatenated with trans_heads_concat=True, they share one time coordinate
        # Total dimension = num_heads * head_dim + 1
        self.head_dim_spatial = head_dim
        self.head_dim_lorentz = head_dim + 1  # Per-head Lorentz dimension (spatial + time)

        self.width = num_heads[-1] * self.head_dim_lorentz  # Final width
        self.embed_dim = embed_dim if embed_dim is not None else self.width
        self.in_channel = in_channel + 1  # add time-like channel for Lorentz

        self.H0 = self.W0 = image_size // patch_size
        self.num_patches = self.H0 * self.W0

        # Patch embedding to (B, H0*W0, C0) where C0 = num_heads[0] * head_dim_lorentz
        self.width0 = num_heads[0] * self.head_dim_lorentz
        self.patch_embed = hnn.LorentzPatchEmbedding(
            manifold_in, image_size, patch_size, self.in_channel, self.width0 - 1
        )

        # Optional absolute position embedding (Swin typically omits; kept here as a learnable offset)
        self.pe = ManifoldParameter(
            self.manifold_in.random_normal((1, self.num_patches, self.width0)),
            manifold=self.manifold_in, requires_grad=True
        )
        self.add_pos = hnn.LResNet(manifold_in, use_scale=True, scale=1.0)

        # Build hierarchical stages
        self.layers = nn.ModuleList()
        H, W = self.H0, self.W0
        for i, depth in enumerate(depths):
            heads = num_heads[i]
            downsample = i < len(depths) - 1
            # Clamp window size if feature map is smaller than window
            stage_window_size = min(window_size, H, W)
            layer = LorentzSwinLayer(
                manifold_hidden, self.head_dim_lorentz, depth,
                num_heads=heads,
                window_size=stage_window_size,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                downsample=downsample
            )
            self.layers.append(layer)
            if downsample:
                # After merging, spatial dims halve
                H, W = H // 2, W // 2

        self.final_width = num_heads[-1] * self.head_dim_lorentz
        self.width = self.final_width

        if self.embed_dim and self.embed_dim != self.width:
            self.final_proj = hnn.LorentzLinear(self.manifold_out, self.width, self.embed_dim)
        else:
            self.final_proj = None

        # Classifier on Lorentzian centroid pooled token set
        if num_classes > 0:
            projection_dim = self.embed_dim if self.final_proj else self.final_width
            self.classifier = hnn.LorentzMLR(self.manifold_out, projection_dim, num_classes)
        else:
            self.classifier = None

    def forward(self, x, return_embeddings=False, return_both=False, output_attentions=False):
        B = x.size(0)
        x = x.permute(0, 2, 3, 1)
        x_hyp = self.manifold_in.projx(F.pad(x, pad=(1, 0)))
        x = self.patch_embed(x_hyp)
        x = self.add_pos(x, self.pe)

        H, W = self.H0, self.W0
        collected = []
        for layer in self.layers:
            if output_attentions:
                x, H, W, attns = layer(x, H, W, output_attentions=True)
                collected.append(attns)
            else:
                x, H, W = layer(x, H, W, output_attentions=False)

        emb = self.manifold_out.lorentzian_centroid(x)

        if self.final_proj is not None:
            emb = self.final_proj(emb)

        if self.num_classes == 0:
            return emb
        if return_embeddings:
            return emb
        logits = self.classifier(emb)
        if return_both:
            if output_attentions:
                return (logits, emb), collected
            return logits, emb
        if output_attentions:
            return logits, collected
        return logits


def LSwin_tiny(manifold_in, manifold_hidden, manifold_out,
               image_size=224, patch_size=4, num_classes=0,
               depths=(2,2,6,2), num_heads=(2,4,8,16),
               head_dim=32, window_size=7, mlp_ratio=4.0, dropout=0.0, **kwargs):
    return LSwinViT(
        manifold_in, manifold_hidden, manifold_out,
        image_size=image_size, patch_size=patch_size,
        depths=depths, num_heads=num_heads, head_dim=head_dim,
        window_size=window_size, mlp_ratio=mlp_ratio,
        dropout=dropout, num_classes=num_classes, **kwargs
    )


def LSwin_small(manifold_in, manifold_hidden, manifold_out,
                image_size=224, patch_size=4, num_classes=0,
                depths=(2,2,18,2), num_heads=(3,6,12,24),
                head_dim=32, window_size=7, mlp_ratio=4.0, dropout=0.0, **kwargs):
    return LSwinViT(
        manifold_in, manifold_hidden, manifold_out,
        image_size=image_size, patch_size=patch_size,
        depths=depths, num_heads=num_heads, head_dim=head_dim,
        window_size=window_size, mlp_ratio=mlp_ratio,
        dropout=dropout, num_classes=num_classes, **kwargs
    )


def LSwin_base(manifold_in, manifold_hidden, manifold_out,
               image_size=224, patch_size=4, num_classes=0,
               depths=(2,2,18,2), num_heads=(4,8,16,32),
               head_dim=48, window_size=7, mlp_ratio=4.0, dropout=0.0, **kwargs):
    return LSwinViT(
        manifold_in, manifold_hidden, manifold_out,
        image_size=image_size, patch_size=patch_size,
        depths=depths, num_heads=num_heads, head_dim=head_dim,
        window_size=window_size, mlp_ratio=mlp_ratio,
        dropout=dropout, num_classes=num_classes, **kwargs
    )
