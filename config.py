"""
config.py
=========
Central configuration for Retinal-SwinNet.

Keeping every hyperparameter, path, and architectural constant in one place
means the training script, the model, and the paper-section generator all
read from a single source of truth -- important for reproducibility when
this project gets written up for the IEEE submission.
"""

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # ------------------------------------------------------------------ #
    # Data
    # ------------------------------------------------------------------ #
    # Combined Dataset A + B (used for training and validation)
    train_root: str = "./data/train"

    # Independent Dataset C (used only for final testing)
    test_root: str = "./data/test"

    # 4-class ocular disease classification
    class_names: List[str] = field(default_factory=lambda: [
        "Glaucoma",
        "Cataracts",
        "Diabetic_Retinopathy",
        "Normal"
    ])

    num_classes: int = 4

    image_size: int = 300                     # EfficientNet-B3 native res
    val_split: float = 0.15
    num_workers: int = 4
    batch_size: int = 16

    # ------------------------------------------------------------------ #
    # Preprocessing (Section 2)
    # ------------------------------------------------------------------ #
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: tuple = (8, 8)

    # ------------------------------------------------------------------ #
    # Model (Section 3)
    # ------------------------------------------------------------------ #
    cnn_backbone_name: str = "efficientnet_b3"
    swin_backbone_name: str = "swin_t"         # swin_t (fast) or swin_b (larger)
    fused_dim: int = 512
    cross_attn_heads: int = 8
    classifier_hidden_dim: int = 128
    dropout_1: float = 0.4
    dropout_2: float = 0.2

    # ------------------------------------------------------------------ #
    # Loss (Section 4)
    # ------------------------------------------------------------------ #
    focal_gamma: float = 2.0
    label_smoothing: float = 0.1

    # ------------------------------------------------------------------ #
    # Optimizer / Scheduler (Section 4)
    # ------------------------------------------------------------------ #
    lr_head: float = 1e-4
    lr_backbone: float = 1e-5
    weight_decay: float = 1e-4
    scheduler_T0: int = 10
    scheduler_Tmult: int = 2

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    num_epochs: int = 100
    seed: int = 42
    device: str = "cuda"                       # falls back to cpu automatically

    # ------------------------------------------------------------------ #
    # Checkpointing (Section 5)
    # ------------------------------------------------------------------ #
    checkpoint_dir: str = "./checkpoints"
    latest_ckpt_name: str = "latest_checkpoint.pth"
    best_ckpt_name: str = "best_model.pth"

    # ------------------------------------------------------------------ #
    # Outputs (metrics, XAI, paper text)
    # ------------------------------------------------------------------ #
    output_dir: str = "./outputs"

    def __post_init__(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)


CFG = Config()