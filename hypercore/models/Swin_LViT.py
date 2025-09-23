import torch
import torch.nn as nn
import torch.nn.functional as F
from geoopt import ManifoldParameter
from .. import nn as hnn


def window_partition(x, window_size, H, W):
    """
    x: (B, H*W, C) on Lorentz manifold (C includes the time-like coord)
    returns: (num_windows * B, window_size*window_size, C)
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


def window_reverse(windows, window_size, H, W, B):
    """
    windows: (num_windows*B, window_size*window_size, C)
    returns: (B, H*W, C)
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
    def __init__(self, manifold, in_channel, hidden_channel, dropout=0):
        super().__init__()
        self.manifold = manifold
        self.dense_1 = hnn.LorentzLinear(self.manifold, in_channel, hidden_channel - 1)
        self.dense_2 = hnn.LorentzLinear(self.manifold, hidden_channel, in_channel - 1)
        self.activation = hnn.LorentzActivation(manifold, activation=nn.GELU())
        self.dropout = hnn.LorentzDropout(self.manifold, dropout)

    def forward(self, x):
        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dense_2(x)
        x = self.dropout(x)
        return x

def build_rope_2d(window_size: int, head_spatial_dim: int, device):
    """
    Make complex RoPE frequencies for a WxW window.
    Return shape: [L=W*W, head_spatial_dim//2] (complex).
    We split the complex channels between y and x (roughly half-half).
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
    def __init__(self, manifold, dim, num_heads, window_size):
        super().__init__()
        self.manifold = manifold
        self.dim = dim # per-head Lorentz dim (incl. time)
        self.num_heads = num_heads
        self.window_size = window_size

        self.attn = hnn.LorentzMultiheadAttention(
            manifold, dim, dim, num_heads,
            attention_type='full',
            trans_heads_concat=True
        )
        self.ln = hnn.LorentzLayerNorm(manifold, num_heads * dim - 1)
        self.res = hnn.LResNet(manifold, use_scale=True, scale=27.5)

        self.register_buffer('_rope_cache_complex', None, persistent=False)

    def _get_rope(self, device):
        # per-head spatial dim expected by attention after Wq/Wk: (out_channels - 1)
        Dsp = int(self.attn.out_channels - 1)
        assert Dsp % 2 == 0, "Require even (out_channels-1) for RoPE."
        L = self.window_size * self.window_size

        if (self._rope_cache_complex is None or
            self._rope_cache_complex.shape[0] != L or
            self._rope_cache_complex.shape[1] != Dsp // 2 or
            self._rope_cache_complex.device != device):
            self._rope_cache_complex = build_rope_2d(self.window_size, Dsp, device)
        return self._rope_cache_complex  # [L, Dsp//2] complex

    def forward(self, x, H, W, output_attentions=False):
        B, N, C = x.shape
        h = self.ln(x)

        win_tokens = window_partition(h, self.window_size, H, W) # (nWB, L, C)

        rope = self._get_rope(win_tokens.device) # [L, Dsp//2] complex
        # sanity check just once per run (cheap)
        with torch.no_grad():
            # after Wq: [Bwin, L, H, Dsp] -> view_as_complex -> [..., Dsp/2]
            Dsp = int(self.attn.out_channels - 1)
            assert rope.shape == (self.window_size * self.window_size, Dsp // 2), \
                f"RoPE shape {rope.shape} != ({self.window_size*self.window_size}, {Dsp//2})"

        out = self.attn(win_tokens, win_tokens, output_attentions=output_attentions, rot_pos=rope)
        if output_attentions:
            out, attn_w = out

        y = window_reverse(out if not output_attentions else out, self.window_size, H, W, B)
        y = self.res(x, y)
        return (y, attn_w) if output_attentions else y


class LorentzSwinBlock(nn.Module):
    """
    A single Swin block with either W-MSA or SW-MSA, now passing output_attentions down.
    """
    def __init__(self, manifold, dim, hidden_mlp_dim, num_heads, window_size, shift_size=0, dropout=0.0):
        super().__init__()
        self.manifold = manifold
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.attn = LorentzWindowAttention(manifold, dim, num_heads, window_size)
        self.ln2 = hnn.LorentzLayerNorm(manifold, num_heads * dim - 1)
        self.mlp = HyperbolicMLP(manifold, num_heads * dim, hidden_mlp_dim, dropout=dropout)
        self.res2 = hnn.LResNet(manifold, use_scale=True, scale=27.5)

    def forward(self, x, H, W, output_attentions=False):
        """
        x: (B, H*W, C)
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
    2x downsample in H/W; 4*C -> 2*C (typical Swin doubling).
    Works by concatenating 2x2 neighbors along channel, projecting with LorentzLinear.
    """
    def __init__(self, manifold, in_dim, out_dim):
        super().__init__()
        self.manifold = manifold
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.proj = hnn.LorentzLinear(self.manifold, 4 * in_dim, out_dim - 1)

    def forward(self, x, H, W):
        """
        x: (B, H*W, C)
        returns: (B, (H/2)*(W/2), C_out), H/2, W/2
        """
        B, N, C = x.shape
        assert N == H * W
        assert H % 2 == 0 and W % 2 == 0, "H and W must be even for patch merging"

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
    def __init__(self, manifold, dim, depth, num_heads, window_size, mlp_ratio=4.0, dropout=0.0, downsample=True):
        super().__init__()
        self.blocks = nn.ModuleList()
        hidden_mlp_dim = int(mlp_ratio * num_heads * dim)
        for i in range(depth):
            shift = 0 if (i % 2 == 0) else window_size // 2
            self.blocks.append(
                LorentzSwinBlock(
                    manifold, dim, hidden_mlp_dim,
                    num_heads=num_heads, window_size=window_size,
                    shift_size=shift, dropout=dropout
                )
            )
        self.downsample = downsample
        if downsample:
            self.merger = LorentzPatchMerging(manifold, num_heads * dim, 2 * num_heads * dim)

    def forward(self, x, H, W, output_attentions=False):
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
        embed_dim=64, # per-head dim in your setup (hidden_channel)
        window_size=7,
        mlp_ratio=4.0,
        dropout=0.0,
        num_classes=0,
    ):
        super().__init__()
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
        self.embed_dim = embed_dim
        self.in_channel = in_channel + 1  # add time-like channel for Lorentz

        assert image_size % patch_size == 0, "Image size must be divisible by patch size"
        self.H0 = self.W0 = image_size // patch_size
        self.num_patches = self.H0 * self.W0

        # Patch embedding to (B, H0*W0, C0) where C0 = num_heads[0]*embed_dim
        self.width0 = num_heads[0] * embed_dim
        self.patch_embed = hnn.LorentzPatchEmbedding(
            manifold_in, image_size, patch_size, self.in_channel, self.width0 - 1
        )

        # optional absolute pos (Swin typically omits; kept here as a learnable offset)
        self.pe = ManifoldParameter(
            self.manifold_in.random_normal((1, self.num_patches, self.width0)),
            manifold=self.manifold_in, requires_grad=True
        )
        self.add_pos = hnn.LResNet(manifold_in, use_scale=True, scale=1.0)

        # Stages
        self.layers = nn.ModuleList()
        H, W = self.H0, self.W0
        dim = embed_dim
        for i, depth in enumerate(depths):
            heads = num_heads[i]
            downsample = i < len(depths) - 1
            layer = LorentzSwinLayer(
                manifold_hidden, dim, depth,
                num_heads=heads,
                window_size=min(window_size, H, W),  # clamp if feature map is small
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                downsample=downsample
            )
            self.layers.append(layer)
            if downsample:
                # after merging, spatial dims halve; channel per-head stays `dim`, but width doubles
                H, W = H // 2, W // 2
                dim = dim  # per-head scalar stays; width grows via num_heads progression
                # next stage heads already set by num_heads[i+1]

        self.final_width = num_heads[-1] * embed_dim
        self.width = self.final_width
        # classifier on Lorentzian centroid pooled token set
        self.classifier = hnn.LorentzMLR(self.manifold_out, self.final_width, num_classes)

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
               embed_dim=32, window_size=7, mlp_ratio=4.0, dropout=0.0):
    return LSwinViT(
        manifold_in, manifold_hidden, manifold_out,
        image_size=image_size, patch_size=patch_size,
        depths=depths, num_heads=num_heads, embed_dim=embed_dim,
        window_size=window_size, mlp_ratio=mlp_ratio,
        dropout=dropout, num_classes=num_classes
    )


def LSwin_small(manifold_in, manifold_hidden, manifold_out,
                image_size=224, patch_size=4, num_classes=0,
                depths=(2,2,18,2), num_heads=(3,6,12,24),
                embed_dim=32, window_size=7, mlp_ratio=4.0, dropout=0.0):
    return LSwinViT(
        manifold_in, manifold_hidden, manifold_out,
        image_size=image_size, patch_size=patch_size,
        depths=depths, num_heads=num_heads, embed_dim=embed_dim,
        window_size=window_size, mlp_ratio=mlp_ratio,
        dropout=dropout, num_classes=num_classes
    )


def LSwin_base(manifold_in, manifold_hidden, manifold_out,
               image_size=224, patch_size=4, num_classes=0,
               depths=(2,2,18,2), num_heads=(4,8,16,32),
               embed_dim=48, window_size=7, mlp_ratio=4.0, dropout=0.0):
    return LSwinViT(
        manifold_in, manifold_hidden, manifold_out,
        image_size=image_size, patch_size=patch_size,
        depths=depths, num_heads=num_heads, embed_dim=embed_dim,
        window_size=window_size, mlp_ratio=mlp_ratio,
        dropout=dropout, num_classes=num_classes
    )
