"""
model.py
========
Section 3: Novel Hybrid Architecture -- "Retinal-SwinNet".

Design:
    Fundus Image
        |--------------------------------------------|
        v                                              v
    EfficientNet-B3 (local CNN branch)        Swin-Transformer (global branch)
    -> spatial feature map F_CNN               -> spatial token grid F_Swin
        |                                              |
        v                                              v
      1x1 Conv projection                     Linear projection
        \___________________  fused token dim (d) __________________/
                              |
                       Bidirectional Cross-Attention
                  (CNN tokens <-> Swin tokens, multi-head)
                              |
                    Global-Average-Pool + Concat
                         -> F_Fused in R^(B x 512)
                              |
                 Dropout(0.4) -> Linear(512,128) -> ReLU
                              |
                 Dropout(0.2) -> Linear(128, 4)
                              |
                     4-class logits

Both backbones are used as feature extractors. Their spatial token maps
are passed into the cross-attention fusion layer, allowing CNN and
Transformer tokens to query each other.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


# --------------------------------------------------------------------- #
# Branch 1: CNN local feature extractor
# --------------------------------------------------------------------- #

class EfficientNetB3Branch(nn.Module):
    """
    Truncated EfficientNet-B3 backbone.

    Returns the last convolutional feature map before global pooling
    and classification.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        weights = (
            torchvision.models.EfficientNet_B3_Weights.DEFAULT
            if pretrained else None
        )

        backbone = torchvision.models.efficientnet_b3(
            weights=weights
        )

        # Full convolutional feature extractor
        self.features = backbone.features

        # EfficientNet-B3 final feature channels
        self.out_channels = 1536

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            Tensor of shape (B, 1536, H', W')
        """
        return self.features(x)


# --------------------------------------------------------------------- #
# Branch 2: Swin Transformer global contextual engine
# --------------------------------------------------------------------- #

class SwinTransformerBranch(nn.Module):
    """
    Truncated Swin Transformer backbone.

    Supported variants:
        - swin_t
        - swin_b

    Returns the final spatial token representation.
    """

    def __init__(
        self,
        variant: str = "swin_t",
        pretrained: bool = True
    ):
        super().__init__()

        if variant == "swin_t":

            weights = (
                torchvision.models.Swin_T_Weights.DEFAULT
                if pretrained else None
            )

            backbone = torchvision.models.swin_t(
                weights=weights
            )

            self.out_channels = 768

        elif variant == "swin_b":

            weights = (
                torchvision.models.Swin_B_Weights.DEFAULT
                if pretrained else None
            )

            backbone = torchvision.models.swin_b(
                weights=weights
            )

            self.out_channels = 1024

        else:
            raise ValueError(
                f"Unsupported swin variant: {variant}"
            )

        # Swin patch-merging feature stages
        self.features = backbone.features

        # Final LayerNorm
        self.norm = backbone.norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        tokens = self.features(x)

        tokens = self.norm(tokens)

        # (B, H, W, C) -> (B, C, H, W)
        tokens = tokens.permute(
            0, 3, 1, 2
        ).contiguous()

        return tokens


# --------------------------------------------------------------------- #
# Fusion: Projection + Bidirectional Cross-Attention
# --------------------------------------------------------------------- #

class CrossAttentionFusion(nn.Module):
    """
    Projects CNN and Swin features into a shared embedding dimension.

    CNN tokens attend to Swin tokens and Swin tokens attend to CNN tokens.

    The resulting representations are globally pooled and concatenated
    into the final fused representation:

        F_Fused ∈ R^(B × 512)
    """

    def __init__(
        self,
        cnn_channels: int,
        swin_channels: int,
        proj_dim: int = 256,
        fused_dim: int = 512,
        num_heads: int = 8,
    ):
        super().__init__()

        assert fused_dim == 2 * proj_dim, (
            "fused_dim must equal 2 * proj_dim since "
            "the two attended representations are concatenated."
        )

        # CNN projection
        self.cnn_proj = nn.Conv2d(
            cnn_channels,
            proj_dim,
            kernel_size=1
        )

        # Swin projection
        self.swin_proj = nn.Conv2d(
            swin_channels,
            proj_dim,
            kernel_size=1
        )

        # CNN -> Swin attention
        self.cnn_to_swin_attn = nn.MultiheadAttention(
            embed_dim=proj_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # Swin -> CNN attention
        self.swin_to_cnn_attn = nn.MultiheadAttention(
            embed_dim=proj_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # Normalization layers
        self.cnn_norm = nn.LayerNorm(proj_dim)
        self.swin_norm = nn.LayerNorm(proj_dim)

    @staticmethod
    def _flatten(x: torch.Tensor) -> torch.Tensor:
        """
        Convert:

            (B, C, H, W)

        into:

            (B, H*W, C)
        """

        b, c, h, w = x.shape

        return x.flatten(2).transpose(1, 2)

    def forward(
        self,
        f_cnn: torch.Tensor,
        f_swin: torch.Tensor
    ) -> torch.Tensor:

        # Project CNN feature map
        cnn_projected = self.cnn_proj(f_cnn)

        # Project Swin feature map
        swin_projected = self.swin_proj(f_swin)

        # Convert feature maps to token sequences
        cnn_tok = self._flatten(
            cnn_projected
        )

        swin_tok = self._flatten(
            swin_projected
        )

        # -------------------------------------------------------------- #
        # CNN tokens query Swin tokens
        # -------------------------------------------------------------- #

        cnn_attended, _ = self.cnn_to_swin_attn(
            query=cnn_tok,
            key=swin_tok,
            value=swin_tok
        )

        cnn_attended = self.cnn_norm(
            cnn_attended + cnn_tok
        )

        # -------------------------------------------------------------- #
        # Swin tokens query CNN tokens
        # -------------------------------------------------------------- #

        swin_attended, _ = self.swin_to_cnn_attn(
            query=swin_tok,
            key=cnn_tok,
            value=cnn_tok
        )

        swin_attended = self.swin_norm(
            swin_attended + swin_tok
        )

        # -------------------------------------------------------------- #
        # Global average pooling
        # -------------------------------------------------------------- #

        cnn_pooled = cnn_attended.mean(
            dim=1
        )

        swin_pooled = swin_attended.mean(
            dim=1
        )

        # -------------------------------------------------------------- #
        # Final fusion
        # -------------------------------------------------------------- #

        fused = torch.cat(
            [
                cnn_pooled,
                swin_pooled
            ],
            dim=1
        )

        return fused


# --------------------------------------------------------------------- #
# Classifier head
# --------------------------------------------------------------------- #

class ClassifierHead(nn.Module):
    """
    Classification head:

        Dropout(0.4)
        -> Linear(512, 128)
        -> ReLU
        -> Dropout(0.2)
        -> Linear(128, 4)

    Output:
        4-class logits
    """

    def __init__(
        self,
        in_dim: int = 512,
        hidden_dim: int = 128,
        num_classes: int = 4,
        p1: float = 0.4,
        p2: float = 0.2
    ):
        super().__init__()

        self.net = nn.Sequential(

            nn.Dropout(p1),

            nn.Linear(
                in_dim,
                hidden_dim
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Dropout(p2),

            nn.Linear(
                hidden_dim,
                num_classes
            )
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        return self.net(x)


# --------------------------------------------------------------------- #
# Full Retinal-SwinNet model
# --------------------------------------------------------------------- #

class RetinalSwinNet(nn.Module):
    """
    Hybrid CNN-Swin Transformer architecture for:

        1. Glaucoma
        2. Cataracts
        3. Diabetic Retinopathy
        4. Normal

    The underlying architecture remains unchanged:

        EfficientNet-B3
                +
        Swin Transformer
                ↓
        Bidirectional Cross-Attention
                ↓
        Feature Fusion
                ↓
        4-Class Classifier
    """

    def __init__(
        self,
        num_classes: int = 4,
        swin_variant: str = "swin_t",
        proj_dim: int = 256,
        fused_dim: int = 512,
        cross_attn_heads: int = 8,
        classifier_hidden_dim: int = 128,
        dropout_1: float = 0.4,
        dropout_2: float = 0.2,
        pretrained: bool = True,
    ):
        super().__init__()

        # -------------------------------------------------------------- #
        # EfficientNet-B3 local feature branch
        # -------------------------------------------------------------- #

        self.cnn_branch = EfficientNetB3Branch(
            pretrained=pretrained
        )

        # -------------------------------------------------------------- #
        # Swin Transformer global feature branch
        # -------------------------------------------------------------- #

        self.swin_branch = SwinTransformerBranch(
            variant=swin_variant,
            pretrained=pretrained
        )

        # -------------------------------------------------------------- #
        # Cross-attention fusion
        # -------------------------------------------------------------- #

        self.fusion = CrossAttentionFusion(

            cnn_channels=self.cnn_branch.out_channels,

            swin_channels=self.swin_branch.out_channels,

            proj_dim=proj_dim,

            fused_dim=fused_dim,

            num_heads=cross_attn_heads
        )

        # -------------------------------------------------------------- #
        # 4-class classifier
        # -------------------------------------------------------------- #

        self.classifier = ClassifierHead(

            in_dim=fused_dim,

            hidden_dim=classifier_hidden_dim,

            num_classes=num_classes,

            p1=dropout_1,

            p2=dropout_2
        )

        # Cached feature maps for XAI
        self._last_cnn_feats = None
        self._last_swin_feats = None

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ):

        # -------------------------------------------------------------- #
        # Extract local CNN features
        # -------------------------------------------------------------- #

        f_cnn = self.cnn_branch(x)

        # -------------------------------------------------------------- #
        # Extract global Transformer features
        # -------------------------------------------------------------- #

        f_swin = self.swin_branch(x)

        # Store features for XAI
        self._last_cnn_feats = f_cnn
        self._last_swin_feats = f_swin

        # -------------------------------------------------------------- #
        # Cross-attention feature fusion
        # -------------------------------------------------------------- #

        fused = self.fusion(
            f_cnn,
            f_swin
        )

        # -------------------------------------------------------------- #
        # 4-class prediction
        # -------------------------------------------------------------- #

        logits = self.classifier(
            fused
        )

        if return_features:

            return (
                logits,
                fused,
                f_cnn,
                f_swin
            )

        return logits

    # ------------------------------------------------------------------ #
    # Differential learning-rate parameter groups
    # ------------------------------------------------------------------ #

    def get_param_groups(
        self,
        lr_head: float,
        lr_backbone: float,
        weight_decay: float
    ):
        """
        Pretrained CNN + Swin backbones receive the smaller learning rate.

        Fusion + classifier receive the larger learning rate.
        """

        backbone_params = (
            list(self.cnn_branch.parameters())
            +
            list(self.swin_branch.parameters())
        )

        head_params = (
            list(self.fusion.parameters())
            +
            list(self.classifier.parameters())
        )

        return [

            {
                "params": backbone_params,
                "lr": lr_backbone,
                "weight_decay": weight_decay
            },

            {
                "params": head_params,
                "lr": lr_head,
                "weight_decay": weight_decay
            }
        ]


# --------------------------------------------------------------------- #
# Model factory
# --------------------------------------------------------------------- #

def build_model(cfg) -> RetinalSwinNet:
    """
    Factory function that builds Retinal-SwinNet using Config.
    """

    return RetinalSwinNet(

        num_classes=cfg.num_classes,

        swin_variant=cfg.swin_backbone_name,

        proj_dim=cfg.fused_dim // 2,

        fused_dim=cfg.fused_dim,

        cross_attn_heads=cfg.cross_attn_heads,

        classifier_hidden_dim=cfg.classifier_hidden_dim,

        dropout_1=cfg.dropout_1,

        dropout_2=cfg.dropout_2,

        pretrained=True
    )