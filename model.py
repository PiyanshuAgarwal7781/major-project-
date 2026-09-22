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
        \\___________________  fused token dim (d) __________________/
                              |
                    Bidirectional Cross-Attention
              (CNN tokens <-> Swin tokens, multi-head)
                              |
                    Global-Average-Pool + Concat -> F_Fused in R^(B x 512)
                              |
              Dropout(0.4) -> Linear(512,128) -> ReLU
                              |
              Dropout(0.2) -> Linear(128, 5)  [5-class logits]

Both backbones are used *as feature extractors* (their spatial token maps,
before global pooling / classification heads), which is what lets the
cross-attention fusion layer actually let CNN and Transformer tokens
"query" each other rather than just concatenating two pooled vectors.
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
    Truncated EfficientNet-B3 backbone. Returns the last convolutional
    feature map (before global pooling / classifier), shape
    (B, C_cnn, H, W), so fine-grained local lesion cues (microaneurysms,
    cotton-wool spots, cup/disc rim) survive into the fusion stage.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = torchvision.models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
        backbone = torchvision.models.efficientnet_b3(weights=weights)
        # `.features` is the full conv stack; EfficientNet-B3's last stage
        # outputs 1536 channels.
        self.features = backbone.features
        self.out_channels = 1536

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)  # (B, 1536, H', W')


# --------------------------------------------------------------------- #
# Branch 2: Swin-Transformer global contextual engine
# --------------------------------------------------------------------- #
class SwinTransformerBranch(nn.Module):
    """
    Truncated Swin-Transformer backbone (swin_t or swin_b from
    torchvision). Returns the last-stage token grid, shape
    (B, H', W', C_swin), reshaped to (B, C_swin, H', W') so it lines up
    with the CNN branch's convention for the fusion layer.
    """

    def __init__(self, variant: str = "swin_t", pretrained: bool = True):
        super().__init__()
        if variant == "swin_t":
            weights = torchvision.models.Swin_T_Weights.DEFAULT if pretrained else None
            backbone = torchvision.models.swin_t(weights=weights)
            self.out_channels = 768
        elif variant == "swin_b":
            weights = torchvision.models.Swin_B_Weights.DEFAULT if pretrained else None
            backbone = torchvision.models.swin_b(weights=weights)
            self.out_channels = 1024
        else:
            raise ValueError(f"Unsupported swin variant: {variant}")

        # `backbone.features` yields the patch-merging stages; the final
        # LayerNorm (backbone.norm) is applied after, per torchvision's
        # SwinTransformer implementation, on tokens shaped (B, H, W, C).
        self.features = backbone.features
        self.norm = backbone.norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.features(x)          # (B, H', W', C_swin)
        tokens = self.norm(tokens)
        tokens = tokens.permute(0, 3, 1, 2).contiguous()  # (B, C_swin, H', W')
        return tokens


# --------------------------------------------------------------------- #
# Fusion: 1x1 Conv/Linear projection + bidirectional cross-attention
# --------------------------------------------------------------------- #
class CrossAttentionFusion(nn.Module):
    """
    Projects CNN and Swin token maps into a shared embedding dimension,
    flattens each into a token sequence, and lets the two modalities
    attend to each other via two `nn.MultiheadAttention` passes:
        CNN tokens attend over Swin tokens (query=CNN, kv=Swin)
        Swin tokens attend over CNN tokens (query=Swin, kv=CNN)
    The attended sequences are then globally average-pooled and
    concatenated into a single fused vector F_Fused in R^(B x fused_dim).
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
            "fused_dim must equal 2 * proj_dim since we concatenate the "
            "pooled CNN-attended and Swin-attended vectors."
        )

        self.cnn_proj = nn.Conv2d(cnn_channels, proj_dim, kernel_size=1)
        self.swin_proj = nn.Conv2d(swin_channels, proj_dim, kernel_size=1)

        self.cnn_to_swin_attn = nn.MultiheadAttention(
            embed_dim=proj_dim, num_heads=num_heads, batch_first=True
        )
        self.swin_to_cnn_attn = nn.MultiheadAttention(
            embed_dim=proj_dim, num_heads=num_heads, batch_first=True
        )

        self.cnn_norm = nn.LayerNorm(proj_dim)
        self.swin_norm = nn.LayerNorm(proj_dim)

    @staticmethod
    def _flatten(x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, H*W, C)
        b, c, h, w = x.shape
        return x.flatten(2).transpose(1, 2)

    def forward(self, f_cnn: torch.Tensor, f_swin: torch.Tensor) -> torch.Tensor:
        cnn_tok = self._flatten(self.cnn_proj(f_cnn))      # (B, N_cnn, proj_dim)
        swin_tok = self._flatten(self.swin_proj(f_swin))   # (B, N_swin, proj_dim)

        # CNN tokens query the Swin (global-context) tokens.
        cnn_attended, _ = self.cnn_to_swin_attn(query=cnn_tok, key=swin_tok, value=swin_tok)
        cnn_attended = self.cnn_norm(cnn_attended + cnn_tok)  # residual

        # Swin tokens query the CNN (local-lesion) tokens.
        swin_attended, _ = self.swin_to_cnn_attn(query=swin_tok, key=cnn_tok, value=cnn_tok)
        swin_attended = self.swin_norm(swin_attended + swin_tok)  # residual

        cnn_pooled = cnn_attended.mean(dim=1)    # (B, proj_dim)
        swin_pooled = swin_attended.mean(dim=1)  # (B, proj_dim)

        fused = torch.cat([cnn_pooled, swin_pooled], dim=1)  # (B, fused_dim)
        return fused


# --------------------------------------------------------------------- #
# Classifier head
# --------------------------------------------------------------------- #
class ClassifierHead(nn.Module):
    """Dropout(0.4) -> Linear(512,128) -> ReLU -> Dropout(0.2) -> Linear(128, num_classes)."""

    def __init__(self, in_dim: int = 512, hidden_dim: int = 128,
                 num_classes: int = 5, p1: float = 0.4, p2: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(p1),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# --------------------------------------------------------------------- #
# Full model
# --------------------------------------------------------------------- #
class RetinalSwinNet(nn.Module):
    """
    Hybrid CNN-Swin Transformer architecture for 5-class ocular disease
    diagnosis from color fundus photographs.
    """

    def __init__(
        self,
        num_classes: int = 5,
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
        self.cnn_branch = EfficientNetB3Branch(pretrained=pretrained)
        self.swin_branch = SwinTransformerBranch(variant=swin_variant, pretrained=pretrained)

        self.fusion = CrossAttentionFusion(
            cnn_channels=self.cnn_branch.out_channels,
            swin_channels=self.swin_branch.out_channels,
            proj_dim=proj_dim,
            fused_dim=fused_dim,
            num_heads=cross_attn_heads,
        )

        self.classifier = ClassifierHead(
            in_dim=fused_dim,
            hidden_dim=classifier_hidden_dim,
            num_classes=num_classes,
            p1=dropout_1,
            p2=dropout_2,
        )

        # Cached from the last forward pass, for Grad-CAM / attention
        # visualization in xai.py.
        self._last_cnn_feats = None
        self._last_swin_feats = None

    def forward(self, x: torch.Tensor, return_features: bool = False):
        f_cnn = self.cnn_branch(x)     # (B, 1536, Hc, Wc)
        f_swin = self.swin_branch(x)   # (B, 768/1024, Hs, Ws)

        self._last_cnn_feats = f_cnn
        self._last_swin_feats = f_swin

        fused = self.fusion(f_cnn, f_swin)   # (B, fused_dim)
        logits = self.classifier(fused)      # (B, num_classes)

        if return_features:
            return logits, fused, f_cnn, f_swin
        return logits

    def get_param_groups(self, lr_head: float, lr_backbone: float, weight_decay: float):
        """
        Differential learning-rate parameter groups for AdamW, per
        Section 4: pre-trained backbones get a smaller LR than the
        newly-initialized fusion + classifier layers.
        """
        backbone_params = list(self.cnn_branch.parameters()) + list(self.swin_branch.parameters())
        head_params = list(self.fusion.parameters()) + list(self.classifier.parameters())
        return [
            {"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay},
            {"params": head_params, "lr": lr_head, "weight_decay": weight_decay},
        ]


def build_model(cfg) -> RetinalSwinNet:
    """Factory that reads architectural hyperparameters straight from Config."""
    return RetinalSwinNet(
        num_classes=cfg.num_classes,
        swin_variant=cfg.swin_backbone_name,
        proj_dim=cfg.fused_dim // 2,
        fused_dim=cfg.fused_dim,
        cross_attn_heads=cfg.cross_attn_heads,
        classifier_hidden_dim=cfg.classifier_hidden_dim,
        dropout_1=cfg.dropout_1,
        dropout_2=cfg.dropout_2,
        pretrained=True,
    )
