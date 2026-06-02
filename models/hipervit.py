"""
hipervit.py
-----------
HiPerViT: Hybrid Multi-Scale Perception Vision Transformer with SkewMod.

Architecture overview:
    - EfficientNet-B0 backbone (pretrained, partial fine-tuning) extracts
      multi-scale feature maps at three resolutions.
    - Each resolution stream is independently patchified and projected into a
      shared embedding space, then fed through a shared Vision Transformer.
    - A SkewMod module injects a class-imbalance-aware bias into the early
      Transformer layers to improve minority-class gradient flow.  The bias
      signal is the pre-computed imbalance ratio rho = log(pos / neg), passed
      in at construction time from the training script.
    - Four CLS tokens (all-scales concat, 112-scale, 56-scale, 28-scale) are
      fused by a dual MLP head to produce the final logits.

Dependencies:
    torch, timm, einops
"""

import torch
import torch.nn as nn
import timm
import numpy as np
from torch import einsum
from einops import rearrange


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def exists(val):
    """Return True if *val* is not None."""
    return val is not None


def default(val, d):
    """Return *val* if it is not None, otherwise return the default *d*."""
    return val if exists(val) else d


def pair(t):
    """Convert a scalar *t* to a (t, t) tuple; leave tuples unchanged."""
    return t if isinstance(t, tuple) else (t, t)


# ---------------------------------------------------------------------------
# Activation
# ---------------------------------------------------------------------------

_sigmoid = nn.Sigmoid()


class Swish(torch.autograd.Function):
    """
    Swish activation function: f(x) = x * sigmoid(x).

    Implemented as a custom autograd Function so that the sigmoid result
    computed in the forward pass is reused during backpropagation, avoiding
    redundant computation.
    """

    @staticmethod
    def forward(ctx, i):
        result = i * _sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        (i,) = ctx.saved_tensors
        sigmoid_i = _sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class SwishModule(nn.Module):
    """Drop-in ``nn.Module`` wrapper around the :class:`Swish` autograd function."""

    def forward(self, x):
        return Swish.apply(x)


# ---------------------------------------------------------------------------
# Transformer building blocks
# ---------------------------------------------------------------------------

class Residual(nn.Module):
    """
    Generic residual (skip-connection) wrapper.

    Wraps any callable *fn* so that its output is added to its input:
    ``output = fn(x) + x``.
    """

    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    """
    Apply Layer Normalisation before a sub-layer *fn*.

    This follows the "pre-norm" convention used in most modern ViT variants.
    """

    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    """
    Two-layer position-wise feed-forward network (FFN) with GELU activation.

    Args:
        dim:        Input and output feature dimension.
        hidden_dim: Width of the hidden layer (typically ``dim * 4``).
        dropout:    Dropout probability applied after each linear layer.
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """
    Multi-head dot-product self-attention.

    Args:
        dim:      Token embedding dimension.
        heads:    Number of attention heads.
        dim_head: Per-head key/query/value dimension.
        dropout:  Dropout probability on the output projection.
    """

    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        """
        Args:
            x: Token sequence of shape ``(B, N, dim)``.

        Returns:
            Attended output of shape ``(B, N, dim)``.
        """
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv)

        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        attn = self.attend(dots)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


# ---------------------------------------------------------------------------
# SkewMod: class-imbalance-aware embedding modulation
# ---------------------------------------------------------------------------

class SkewMod(nn.Module):
    """
    Skew Modulation (SkewMod) module.

    Injects a learned, imbalance-aware bias into the token stream so that
    minority-class samples receive stronger gradient signal during training.

    At each targeted Transformer layer, a global-average-pooled summary of
    the current token sequence is concatenated with a pre-computed scalar
    imbalance signal ``rho = log(pos / neg)`` (supplied by the training
    script via :class:`Transformer`) and passed through a small MLP.
    The resulting bias vector is broadcast back to all tokens.

    The key difference from a fixed hard-coded ratio is that ``rho`` is
    computed from the *actual* class distribution of the training split, so
    the same model code works correctly across datasets with different
    imbalance levels without any manual tuning.

    Args:
        width: Token embedding dimension *D* (equals ``dim`` of the
               Transformer).
        rho:   Imbalance signal ``log(pos / neg)`` pre-computed by
               :func:`train.calculate_imbalance_ratio_rho`.  A negative
               value (pos < neg) encodes minority-class scarcity; a value
               of 0.0 disables the signal (balanced dataset).
    """

    def __init__(self, width: int, rho: float = 0.0):
        super().__init__()
        # Store rho as a non-trainable buffer so it is moved to the correct
        # device automatically and saved/restored with the model state-dict.
        self.register_buffer("rho", torch.tensor(rho, dtype=torch.float32))
        self.embed_modulator = nn.Sequential(
            nn.Linear(width + 1, width),
            nn.ReLU(),
            nn.Linear(width, width),
        )

    def forward(self, x):
        """
        Inject the imbalance-aware bias into the token sequence.

        Args:
            x: Token tensor of shape ``(B, N, D)`` where *B* is batch size,
               *N* is sequence length, and *D* is the embedding dimension.

        Returns:
            Modulated token tensor of the same shape ``(B, N, D)``.
        """
        B = x.shape[0]

        # Broadcast the scalar rho signal across the batch: (B, 1)
        imbalance_signal = self.rho.expand(B).unsqueeze(1)  # (B, 1)

        # Global average pool across the token / sequence dimension
        gap_summary = x.mean(dim=1)                                    # (B, D)
        gap_input   = torch.cat([gap_summary, imbalance_signal], dim=1) # (B, D+1)

        # Compute and inject the modulation bias back to all tokens
        d_embed = self.embed_modulator(gap_input).unsqueeze(1)  # (B, 1, D)
        return x + d_embed


# ---------------------------------------------------------------------------
# GradProbe: gradient diagnostic hook for SkewMod output
# ---------------------------------------------------------------------------

class GradProbe:
    """
    Backward-hook utility that logs per-class gradient norms at the output of
    a :class:`SkewMod` layer.

    Attach this probe during training to verify that SkewMod is successfully
    amplifying gradients for the minority class relative to the majority class.

    Usage::

        probe = GradProbe()
        model.transformer.skew_mod.register_full_backward_hook(probe.hook)
        ...
        probe.set_labels(batch_labels)   # call before each forward pass
        loss.backward()
        print(probe.logs[-1])            # {"g_min": float, "g_maj": float}

    Attributes:
        logs: List of dicts with keys ``"g_min"`` (minority gradient norm)
              and ``"g_maj"`` (majority gradient norm) recorded per backward
              pass.
    """

    def __init__(self):
        self.logs: list = []
        self._last_labels = None

    def set_labels(self, y: torch.Tensor) -> None:
        """
        Cache the batch labels so that the hook can split gradients by class.

        Must be called before each forward pass whose backward will be probed.

        Args:
            y: Integer class labels of shape ``(B,)``.
        """
        self._last_labels = y.detach().cpu()

    def hook(self, module, grad_input, grad_output) -> None:
        """
        Backward hook registered on a :class:`SkewMod` module.

        Computes the per-sample L2 gradient norm and averages separately
        for minority (label == 1) and majority (label != 1) samples.

        Args:
            module:      The hooked module (unused).
            grad_input:  Tuple of input gradients (unused).
            grad_output: Tuple of output gradients; ``grad_output[0]`` has
                         shape ``(B, N, D)``.
        """
        g_tokens = grad_output[0].detach()

        if self._last_labels is None:
            return

        y = self._last_labels.detach().cpu()

        # Guard against size mismatch from uneven batch samplers
        B = min(g_tokens.shape[0], y.shape[0])
        if B == 0:
            return

        g_tokens = g_tokens[:B]  # (B, N, D)
        y = y[:B]                # (B,)

        g = g_tokens.norm(dim=(1, 2))  # per-sample gradient norm: (B,)

        minority = y == 1
        g_min = g[minority].mean().item() if minority.any() else float("nan")
        g_maj = g[~minority].mean().item() if (~minority).any() else float("nan")

        self.logs.append({"g_min": g_min, "g_maj": g_maj})


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------

class Transformer(nn.Module):
    """
    Standard ViT Transformer encoder with optional SkewMod injection.

    The SkewMod bias is applied after every attention + FFN block in the
    first third of the network (layers ``0 … floor(depth/3) - 1``), giving
    the imbalance signal the widest possible influence on learned
    representations.

    Args:
        dim:            Token embedding dimension.
        depth:          Number of Transformer encoder layers.
        heads:          Number of attention heads.
        dim_head:       Per-head dimension.
        mlp_dim:        Hidden dimension of the FFN.
        dropout:        Dropout probability for attention.
        do_skew_mod:    Whether to enable SkewMod injection.
        skew_mod_layer: Layer index at which SkewMod is centred
                        (defaults to ``depth // 2``).
        rho:            Imbalance ratio ``log(pos / neg)`` forwarded to
                        :class:`SkewMod`.  Computed by
                        :func:`train.calculate_imbalance_ratio_rho` and
                        passed through from :class:`HiPerViT`.
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
        do_skew_mod: bool = True,
        skew_mod_layer: int = None,
        rho: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.do_skew_mod = do_skew_mod

        if self.do_skew_mod:
            self.skew_mod_layer_index = default(skew_mod_layer, depth // 2) - 1
            self.skew_mod = SkewMod(width=dim, rho=rho)

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                    PreNorm(dim, FeedForward(dim=dim, hidden_dim=mlp_dim, dropout=0)),
                ])
            )

    def forward(self, x):
        """
        Args:
            x: Token sequence of shape ``(B, N, dim)``.

        Returns:
            Transformed token sequence of the same shape.
        """
        num_layers = len(self.layers)
        skew_range = range(0, int(num_layers / 3))

        for index, (attn, ff) in enumerate(self.layers):
            x = attn(x) + x
            x = ff(x) + x

            if self.do_skew_mod and index in skew_range:
                x = self.skew_mod(x)

        return x


# ---------------------------------------------------------------------------
# HiPerViT: main model
# ---------------------------------------------------------------------------

class HiPerViT(nn.Module):
    """
    HiPerViT — Hybrid Multi-Scale Perception Vision Transformer.

    Combines a pretrained EfficientNet-B0 backbone with a shared Vision
    Transformer encoder operating on three resolution streams simultaneously.
    A SkewMod layer biases early Transformer layers towards minority-class
    samples, making the model suited for class-imbalanced medical or
    fine-grained visual recognition tasks.

    The imbalance conditioning signal ``rho = log(pos / neg)`` is computed
    from the training split by :func:`train.calculate_imbalance_ratio_rho`
    and passed in at construction time via the *rho* argument.  This makes
    SkewMod dataset-aware without any hard-coded constants.

    Architecture
    ------------
    1. **EfficientNet-B0 backbone** (partial fine-tuning of the last 3 blocks)
       extracts feature maps at three scales:

       - Scale 112 → ``feature_channels[0]`` channels
       - Scale  56 → ``feature_channels[1]`` channels
       - Scale  28 → ``feature_channels[2]`` channels

    2. Each scale is patchified (patch size ``P × P``) and projected to a
       shared embedding dimension *dim* via a dedicated linear layer.

    3. **Four independent ViT forward passes** (shared weights) over:

       - All-scales concatenation: [112 ‖ 56 ‖ 28]
       - 112-scale only
       -  56-scale only
       -  28-scale only

    4. The four resulting CLS tokens are concatenated (``4 × dim``) and
       passed through ``mlp_head_con`` for the primary logits.
       The 28-scale CLS token is additionally passed through ``mlp_head_28``
       for an auxiliary prediction.  Final output is their **sum**.

    Args:
        config:         Configuration dict with a ``"model"`` sub-dict
                        containing keys: ``image-size``, ``patch-size``,
                        ``dim``, ``depth``, ``heads``, ``mlp-dim``,
                        ``emb-dim``, ``dim-head``, ``dropout``,
                        ``emb-dropout``.
        out_dim:        Number of output classes.
        channels:       EfficientNet output channels (1280 for B0, 2560 for
                        B7).  Used only for compatibility; actual patch dims
                        are inferred dynamically from EfficientNet feature
                        info.
        pretrained:     Whether to initialise EfficientNet with ImageNet
                        weights.
        attention_type: Reserved for alternative attention implementations
                        (currently unused).
        rho:            Imbalance ratio ``log(pos / neg)`` computed by
                        :func:`train.calculate_imbalance_ratio_rho` from the
                        training-split DataFrame.  Forwarded to
                        :class:`SkewMod` via :class:`Transformer`.
                        Defaults to ``0.0`` (no imbalance conditioning).
    """

    def __init__(
        self,
        config: dict,
        out_dim: int,
        pretrained: bool = True,
        rho: float = 0.0,
    ):
        super().__init__()

        # ------------------------------------------------------------------
        # Unpack configuration
        # ------------------------------------------------------------------
        model_cfg   = config["model"]
        image_size  = model_cfg["image-size"]
        patch_size  = model_cfg["patch-size"]
        num_classes = out_dim
        dim         = model_cfg["dim"]
        depth       = model_cfg["depth"]
        heads       = model_cfg["heads"]
        mlp_dim     = model_cfg["mlp-dim"]
        emb_dim     = model_cfg["emb-dim"]
        dim_head    = model_cfg["dim-head"]
        dropout     = model_cfg["dropout"]
        emb_dropout = model_cfg["emb-dropout"]

        assert image_size % patch_size == 0, (
            "Image dimensions must be divisible by the patch size. "
            f"Got image_size={image_size}, patch_size={patch_size}."
        )

        self.patch_size = patch_size

        # ------------------------------------------------------------------
        # EfficientNet-B0 backbone (multi-scale, features-only mode)
        # ------------------------------------------------------------------
        self.efficient_net = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            features_only=True,
            out_indices=[2, 3, 4],
        )

        # Freeze all backbone blocks; unfreeze the last 3 for fine-tuning
        for i, block in enumerate(self.efficient_net.blocks):
            requires_grad = i >= len(self.efficient_net.blocks) - 3
            for param in block.parameters():
                param.requires_grad = requires_grad

        # Dynamically read the channel counts for the three output scales
        self.feature_channels = self.efficient_net.feature_info.channels()

        print(
            f"[HiPerViT] backbone: efficientnet_b0 | pretrained: {pretrained} | rho: {rho:.4f}"
        )
        print(f"[HiPerViT] Transformer depth: {depth}")
        print(f"[HiPerViT] Feature channels from EfficientNet: {self.feature_channels}")

        # ------------------------------------------------------------------
        # Patch projection layers (one per scale)
        # ------------------------------------------------------------------
        self.num_patches = (image_size // patch_size) ** 2
        print(f"[HiPerViT] num_patches (per scale): {self.num_patches}")

        self.patch_to_embedding_112 = nn.Linear(
            self.feature_channels[0] * patch_size ** 2, dim
        )
        self.patch_to_embedding_56 = nn.Linear(
            self.feature_channels[1] * patch_size ** 2, dim
        )
        self.patch_to_embedding_28 = nn.Linear(
            self.feature_channels[2] * patch_size ** 2, dim
        )

        # ------------------------------------------------------------------
        # ViT components (shared across all scale streams)
        # ------------------------------------------------------------------
        # pos_embedding covers the longest possible sequence (CLS + all patches
        # from all three scales concatenated).  Shape: (1, emb_dim, dim) so it
        # broadcasts cleanly over the batch dimension.
        self.pos_embedding = nn.Parameter(torch.randn(1, emb_dim, dim))
        self.cls_token     = nn.Parameter(torch.randn(1, 1, dim))
        self.to_cls_token  = nn.Identity()
        self.dropout       = nn.Dropout(emb_dropout)

        # rho is forwarded to Transformer → SkewMod so the imbalance signal
        # is baked in at construction time and does not need to be passed at
        # every forward call.
        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim,
            dropout=dropout,
            rho=rho,
        )

        # ------------------------------------------------------------------
        # Classification heads
        # ------------------------------------------------------------------
        # Primary head: fuses CLS tokens from all four streams
        self.mlp_head_con = nn.Sequential(
            nn.Linear(dim * 4, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, num_classes),
        )

        # Auxiliary head: deep supervision from the 28-scale CLS token
        self.mlp_head_28 = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, num_classes),
        )

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _encode_scale(self, patch_tokens: torch.Tensor, cls_tokens: torch.Tensor) -> torch.Tensor:
        """
        Run one ViT forward pass for a single resolution scale stream.

        Prepends the CLS token, adds positional embeddings, applies dropout,
        runs the Transformer, and returns the CLS output token.

        Args:
            patch_tokens: Patch embeddings of shape ``(B, N, dim)``.
            cls_tokens:   Expanded CLS tokens of shape ``(B, 1, dim)``.

        Returns:
            CLS token output of shape ``(B, dim)``.
        """
        x = torch.cat((cls_tokens, patch_tokens), dim=1)   # (B, 1+N, dim)
        # Slice positional embeddings to the actual sequence length of this
        # stream (varies between all-scales concat and single-scale streams)
        # and broadcast over the batch dimension.
        x += self.pos_embedding[:, : x.shape[1], :]        # (1, 1+N, dim)
        x = self.dropout(x)
        x = self.transformer(x)
        return self.to_cls_token(x[:, 0])

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for HiPerViT.

        Args:
            x: Input image tensor of shape ``(B, C, H, W)``.

        Returns:
            Logits tensor of shape ``(B, num_classes)``, computed as the sum
            of the primary fused-head prediction and the auxiliary 28-scale
            head prediction (deep supervision).
        """
        p = self.patch_size

        # ---- 1. Multi-scale feature extraction ---------------------------
        features = self.efficient_net(x)  # list of 3 feature maps

        # ---- 2. Patchify and project each scale --------------------------
        y_112 = self.patch_to_embedding_112(
            rearrange(features[0], "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=p, p2=p)
        )
        y_56 = self.patch_to_embedding_56(
            rearrange(features[1], "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=p, p2=p)
        )
        y_28 = self.patch_to_embedding_28(
            rearrange(features[2], "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=p, p2=p)
        )

        # ---- 3. All-scales concatenated stream ---------------------------
        y_all = torch.cat((y_112, y_56, y_28), dim=1)  # (B, total_patches, dim)

        # ---- 4. CLS token expanded for the batch -------------------------
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)  # (B, 1, dim)

        # ---- 5. Independent ViT encoding per stream ----------------------
        x_all = self._encode_scale(y_all, cls_tokens)
        x_112 = self._encode_scale(y_112, cls_tokens)
        x_56  = self._encode_scale(y_56,  cls_tokens)
        x_28  = self._encode_scale(y_28,  cls_tokens)

        # ---- 6. Primary fused classification head ------------------------
        x_fused = torch.cat((x_all, x_112, x_56, x_28), dim=1)  # (B, 4*dim)
        logits  = self.mlp_head_con(x_fused)

        # ---- 7. Auxiliary 28-scale head (deep supervision) ---------------
        logits_28 = self.mlp_head_28(x_28)

        return logits + logits_28