"""STAM (Space-Time Attention Model) trainer — config and HuggingFace Trainer wrapper.

Ported from `Alibaba-MIIL/STAM <https://github.com/Alibaba-MIIL/STAM>`_.

STAM processes video by encoding each frame independently through a
spatial Vision Transformer (ViT) and then aggregating the per-frame
CLS tokens with a temporal TransformerEncoder.  It uses **only** 2-D
convolutions (patch embedding), linear layers, and standard multi-head
self-attention — no 3-D convolutions or depthwise ops — making it fully
compatible with Qualcomm AI Hub compilation.

Default configuration matches ViT-B/16 (patch 16, embed 768, depth 12,
12 heads) with 6 temporal transformer layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from loguru import logger

from lpcv.models.base import (
    BaseForClassification,
    BaseModelTrainer,
    BaseTrainerConfig,
    log_freeze_stats,
)

VIT_PRETRAINED_VARIANTS: dict[int, str] = {
    384: "vit_small_patch16_224",
    768: "vit_base_patch16_224",
    1024: "vit_large_patch16_224",
}


def _trunc_normal_(tensor: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    """Fill *tensor* with values drawn from a truncated normal distribution."""
    with torch.no_grad():
        tensor.normal_().fmod_(2).mul_(std)
    return tensor


class _DropPath(nn.Module):
    """Stochastic depth (drop path) per sample."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.rand(shape, dtype=x.dtype, device=x.device).add_(keep).floor_()
        return x.div(keep).mul_(mask)


class _Mlp(nn.Module):
    """Two-layer MLP with GELU activation."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


class _Attention(nn.Module):
    """Multi-head self-attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(x))


class _Block(nn.Module):
    """Transformer block: LayerNorm → Attention → DropPath → LayerNorm → MLP → DropPath."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = _DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            drop=drop,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class _PatchEmbed(nn.Module):
    """2D patch embedding via Conv2d."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class _TemporalAggregate(nn.Module):
    """Temporal aggregation via TransformerEncoder over per-frame embeddings.

    Parameters
    ----------
    clip_length
        Number of frames per clip.
    embed_dim
        Embedding dimension (must match spatial ViT output).
    n_layers
        Number of temporal transformer encoder layers.
    """

    def __init__(
        self,
        clip_length: int,
        embed_dim: int = 768,
        n_layers: int = 6,
    ) -> None:
        super().__init__()
        self.clip_length = clip_length
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            batch_first=False,
        )
        self.transformer_enc = nn.TransformerEncoder(
            enc_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(embed_dim),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, clip_length + 1, embed_dim))

        _trunc_normal_(self.pos_embed)
        _trunc_normal_(self.cls_token)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            _trunc_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Aggregate frame embeddings.

        Parameters
        ----------
        x
            Per-frame CLS embeddings of shape ``(B * T, D)``.

        Returns
        -------
        torch.Tensor
            Video-level embedding of shape ``(B, D)``.
        """
        nvids = x.shape[0] // self.clip_length
        x = x.view(nvids, self.clip_length, -1)

        cls_tokens = self.cls_token.expand(nvids, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        x = x.transpose(1, 0)
        o = self.transformer_enc(x)
        return o[0]


class SpatialViT(nn.Module):
    """Spatial Vision Transformer backbone.

    Processes individual frames through patch embedding, positional
    encoding, and a stack of transformer blocks.

    Parameters
    ----------
    img_size
        Spatial input size (square).
    patch_size
        Patch size for embedding.
    in_chans
        Number of input channels.
    embed_dim
        Transformer embedding dimension.
    depth
        Number of transformer blocks.
    num_heads
        Number of attention heads.
    mlp_ratio
        MLP expansion ratio.
    qkv_bias
        Use bias in QKV projection.
    drop_rate
        Dropout rate.
    attn_drop_rate
        Attention dropout rate.
    drop_path_rate
        Stochastic depth rate.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = _PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                _Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        _trunc_normal_(self.pos_embed)
        _trunc_normal_(self.cls_token)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            _trunc_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of single frames.

        Parameters
        ----------
        x
            Frame tensor of shape ``(B, C, H, W)``.

        Returns
        -------
        torch.Tensor
            CLS token embedding of shape ``(B, embed_dim)``.
        """
        b = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)[:, 0]
        return x


class STAM(nn.Module):
    """Space-Time Attention Model for video classification.

    Encodes each frame independently through a spatial ViT, then
    aggregates per-frame embeddings via a temporal TransformerEncoder.

    Parameters
    ----------
    num_classes
        Number of output classes.
    num_frames
        Number of frames per video clip.
    img_size
        Spatial input size (square).
    patch_size
        Patch size for the spatial ViT.
    embed_dim
        Transformer embedding dimension.
    spatial_depth
        Number of spatial transformer blocks.
    num_heads
        Number of spatial attention heads.
    temporal_layers
        Number of temporal transformer encoder layers.
    mlp_ratio
        MLP expansion ratio.
    qkv_bias
        Use bias in QKV projection.
    drop_rate
        Dropout rate.
    attn_drop_rate
        Attention dropout rate.
    drop_path_rate
        Stochastic depth rate.
    """

    def __init__(
        self,
        num_classes: int,
        num_frames: int = 16,
        img_size: int = 112,
        patch_size: int = 16,
        embed_dim: int = 768,
        spatial_depth: int = 12,
        num_heads: int = 12,
        temporal_layers: int = 6,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_frames = num_frames
        self.spatial = SpatialViT(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=spatial_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
        )
        self.temporal = _TemporalAggregate(
            clip_length=num_frames,
            embed_dim=embed_dim,
            n_layers=temporal_layers,
        )
        self.head = nn.Linear(embed_dim, num_classes)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x
            Video tensor of shape ``(B, C, T, H, W)``.

        Returns
        -------
        torch.Tensor
            Classification logits of shape ``(B, num_classes)``.
        """
        b, c, t, h, w = x.shape
        frames = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        frame_embeds = self.spatial(frames)
        video_embed = self.temporal(frame_embeds)
        return self.head(video_embed)


def _interpolate_pos_embed(
    pos_embed: torch.Tensor,
    orig_size: int,
    new_size: int,
    patch_size: int,
) -> torch.Tensor:
    """Interpolate positional embeddings to a different spatial resolution.

    Parameters
    ----------
    pos_embed
        Original positional embedding of shape ``(1, 1 + N, D)``
        where ``N = (orig_size // patch_size) ** 2``.
    orig_size
        Original spatial input size.
    new_size
        Target spatial input size.
    patch_size
        Patch size used by the model.

    Returns
    -------
    torch.Tensor
        Interpolated positional embedding for the new resolution.
    """
    if orig_size == new_size:
        return pos_embed

    cls_token = pos_embed[:, :1, :]
    patch_pos = pos_embed[:, 1:, :]

    orig_grid = orig_size // patch_size
    new_grid = new_size // patch_size
    dim = patch_pos.shape[-1]

    patch_pos = patch_pos.reshape(1, orig_grid, orig_grid, dim).permute(0, 3, 1, 2)
    patch_pos = nn.functional.interpolate(
        patch_pos.float(),
        size=(new_grid, new_grid),
        mode="bicubic",
        align_corners=False,
    )
    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, new_grid * new_grid, dim)
    return torch.cat([cls_token, patch_pos], dim=1)


def _build_stam(
    num_classes: int,
    num_frames: int = 16,
    img_size: int = 112,
    patch_size: int = 16,
    embed_dim: int = 768,
    spatial_depth: int = 12,
    num_heads: int = 12,
    temporal_layers: int = 6,
    pretrained: bool = False,
) -> STAM:
    """Build a STAM model.

    Parameters
    ----------
    num_classes
        Number of output classes.
    num_frames
        Number of frames per clip.
    img_size
        Spatial input size (square).
    patch_size
        Patch size for spatial ViT.
    embed_dim
        Transformer embedding dimension.
    spatial_depth
        Number of spatial transformer blocks.
    num_heads
        Number of spatial attention heads.
    temporal_layers
        Number of temporal encoder layers.
    pretrained
        Load ImageNet-21k ViT-B/16 pretrained weights for the spatial
        backbone (with positional embedding interpolation if ``img_size``
        differs from 224).

    Returns
    -------
    STAM
        Model instance ready for fine-tuning.
    """
    model = STAM(
        num_classes=num_classes,
        num_frames=num_frames,
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        spatial_depth=spatial_depth,
        num_heads=num_heads,
        temporal_layers=temporal_layers,
    )

    if pretrained:
        try:
            from timm.models import create_model  # type: ignore[import-untyped]

            variant = VIT_PRETRAINED_VARIANTS.get(embed_dim)
            if variant is None:
                logger.warning(
                    f"No pretrained ViT variant for embed_dim={embed_dim}, "
                    f"available: {sorted(VIT_PRETRAINED_VARIANTS)}"
                )
                return model
            vit = create_model(variant, pretrained=True)
            vit_sd = vit.state_dict()

            spatial_sd = model.spatial.state_dict()
            new_sd: dict[str, torch.Tensor] = {}
            for k, v in spatial_sd.items():
                if k in vit_sd and vit_sd[k].shape == v.shape:
                    new_sd[k] = vit_sd[k]
                elif k == "pos_embed" and k in vit_sd and vit_sd[k].shape[-1] == v.shape[-1]:
                    new_sd[k] = _interpolate_pos_embed(vit_sd[k], 224, img_size, patch_size)
                else:
                    new_sd[k] = v

            model.spatial.load_state_dict(new_sd, strict=False)
            logger.info("Loaded ViT-B/16 pretrained weights for STAM spatial backbone")
        except ImportError:
            logger.warning("timm not installed — cannot load pretrained ViT weights")
        except Exception as e:
            logger.warning(f"Failed to load pretrained weights: {e}")

    return model


@dataclass
class STAMTrainerConfig(BaseTrainerConfig):
    """Hyperparameters for a STAM training run.

    Attributes
    ----------
    num_classes
        Number of output classes.  When ``0``, inferred from the dataset.
    num_frames
        Number of frames sampled per video.
    crop_size
        Spatial crop size (square).
    patch_size
        Patch size for spatial ViT.
    embed_dim
        Transformer embedding dimension.
    spatial_depth
        Number of spatial transformer blocks.
    num_heads
        Number of spatial attention heads.
    temporal_layers
        Number of temporal transformer encoder layers.
    label_smoothing
        Label smoothing factor for cross-entropy loss.
    """

    num_classes: int = 0
    num_frames: int = 16
    crop_size: int = 112
    patch_size: int = 16
    embed_dim: int = 768
    spatial_depth: int = 12
    num_heads: int = 12
    temporal_layers: int = 6
    label_smoothing: float = 0.1
    learning_rate: float = 1e-4
    freeze_strategy: str = "partial"


class STAMForClassification(BaseForClassification):
    """STAM wrapper compatible with HuggingFace Trainer.

    Parameters
    ----------
    num_classes
        Number of output classes.
    num_frames
        Number of frames per clip.
    crop_size
        Spatial input size (square).
    patch_size
        Patch size for spatial ViT.
    embed_dim
        Transformer embedding dimension.
    spatial_depth
        Number of spatial transformer blocks.
    num_heads
        Number of spatial attention heads.
    temporal_layers
        Number of temporal encoder layers.
    pretrained
        Load pretrained ViT weights for spatial backbone.
    label_smoothing
        Label smoothing for cross-entropy loss.
    """

    def __init__(
        self,
        num_classes: int,
        num_frames: int = 16,
        crop_size: int = 112,
        patch_size: int = 16,
        embed_dim: int = 768,
        spatial_depth: int = 12,
        num_heads: int = 12,
        temporal_layers: int = 6,
        pretrained: bool = False,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone: nn.Module = _build_stam(
            num_classes=num_classes,
            num_frames=num_frames,
            img_size=crop_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            spatial_depth=spatial_depth,
            num_heads=num_heads,
            temporal_layers=temporal_layers,
            pretrained=pretrained,
        )
        self.num_classes = num_classes
        self.crop_size = crop_size
        self.num_frames = num_frames
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.spatial_depth = spatial_depth
        self.num_heads_val = num_heads
        self.temporal_layers = temporal_layers
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def _extra_save_meta(self) -> dict[str, Any]:
        return {
            "crop_size": self.crop_size,
            "num_frames": self.num_frames,
            "patch_size": self.patch_size,
            "embed_dim": self.embed_dim,
            "spatial_depth": self.spatial_depth,
            "num_heads": self.num_heads_val,
            "temporal_layers": self.temporal_layers,
        }

    @classmethod
    def load_pretrained(cls, path: str | Path) -> STAMForClassification:
        """Load a saved model from a directory.

        Parameters
        ----------
        path
            Directory containing ``model.pt``.

        Returns
        -------
        STAMForClassification
            Loaded model in eval mode.
        """
        path = Path(path)
        checkpoint = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
        model = cls(
            num_classes=checkpoint["num_classes"],
            num_frames=checkpoint.get("num_frames", 16),
            crop_size=checkpoint.get("crop_size", 112),
            patch_size=checkpoint.get("patch_size", 16),
            embed_dim=checkpoint.get("embed_dim", 768),
            spatial_depth=checkpoint.get("spatial_depth", 12),
            num_heads=checkpoint.get("num_heads", 12),
            temporal_layers=checkpoint.get("temporal_layers", 6),
            pretrained=False,
        )
        model.backbone.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model


class STAMModelTrainer(BaseModelTrainer):
    """High-level wrapper around HuggingFace ``Trainer`` for STAM fine-tuning.

    Parameters
    ----------
    config
        A :class:`STAMTrainerConfig` with all hyperparameters.
    train_dataset
        Training dataset.
    eval_dataset
        Evaluation dataset.
    val_transform_config
        Validation transform config to save alongside the model.
    """

    model_display_name = "STAM"
    model: STAMForClassification

    def _init_model(self) -> STAMForClassification:
        num_classes = self.config.num_classes if self.config.num_classes > 0 else self.num_labels
        logger.info(
            f"Initializing STAM trainer: classes={num_classes}, "
            f"crop={self.config.crop_size}, patch={self.config.patch_size}, "
            f"embed={self.config.embed_dim}, spatial_depth={self.config.spatial_depth}, "
            f"heads={self.config.num_heads}, temporal_layers={self.config.temporal_layers}, "
            f"epochs={self.config.num_train_epochs}, freeze={self.config.freeze_strategy}"
        )
        return STAMForClassification(
            num_classes=num_classes,
            num_frames=self.config.num_frames,
            crop_size=self.config.crop_size,
            patch_size=self.config.patch_size,
            embed_dim=self.config.embed_dim,
            spatial_depth=self.config.spatial_depth,
            num_heads=self.config.num_heads,
            temporal_layers=self.config.temporal_layers,
            pretrained=True,
            label_smoothing=self.config.label_smoothing,
        )

    def _apply_freeze_strategy(self, strategy: str) -> None:
        """Freeze model parameters according to *strategy*.

        Parameters
        ----------
        strategy
            One of:

            - ``"none"`` — all parameters trainable.
            - ``"backbone"`` — freeze the entire spatial ViT; only train
              the temporal aggregator and classification head.
            - ``"partial"`` — freeze the spatial ViT patch embedding and
              the first 8 transformer blocks; keep blocks 8–11, norm,
              temporal aggregator, and head trainable.
        """
        if strategy == "none":
            return

        backbone: STAM = self.model.backbone  # type: ignore[assignment]
        if strategy == "backbone":
            for param in backbone.spatial.parameters():
                param.requires_grad = False
        elif strategy == "partial":
            for name, param in backbone.spatial.named_parameters():
                if name.startswith(("blocks.8", "blocks.9", "blocks.10", "blocks.11", "norm")):
                    continue
                param.requires_grad = False
        else:
            logger.warning(f"Unknown freeze strategy '{strategy}', skipping")
            return

        log_freeze_stats(self.model, strategy)

    def _extra_training_args(self) -> dict[str, Any]:
        return {"ddp_find_unused_parameters": False}
