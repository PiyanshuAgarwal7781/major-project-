"""
checkpoint.py
=============
Section 5: Critical Checkpointing & Resume System.

Built for shared lab hardware where a job can be pre-empted, the machine
can reboot, or someone else's job can bump yours off the GPU. The
CheckpointManager guarantees that re-running the exact same training
script picks up from where it left off with zero manual intervention.
"""

import os
from typing import Dict, List, Optional

import torch


class CheckpointManager:
    """
    Handles:
      * writing `latest_checkpoint.pth` at the end of every epoch,
      * writing `best_model.pth` whenever validation accuracy improves,
      * auto-resuming (model / optimizer / scheduler / history) at
        script start if a checkpoint already exists on disk.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        latest_name: str = "latest_checkpoint.pth",
        best_name: str = "best_model.pth",
    ):
        self.checkpoint_dir = checkpoint_dir
        self.latest_path = os.path.join(checkpoint_dir, latest_name)
        self.best_path = os.path.join(checkpoint_dir, best_name)
        os.makedirs(checkpoint_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    def save(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        best_val_acc: float,
        val_acc: float,
        train_history: Dict[str, List[float]],
    ) -> None:
        """Save the latest state every epoch; save best_model.pth if improved."""
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "best_val_acc": best_val_acc,
            "train_history": train_history,
        }
        torch.save(state, self.latest_path)

        if val_acc > best_val_acc:
            torch.save(state, self.best_path)

    # ------------------------------------------------------------------ #
    def load_latest(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler=None,
        map_location: str = "cpu",
    ) -> Dict:
        """
        Loads `latest_checkpoint.pth` into model/optimizer/scheduler in
        place, and returns a dict with `start_epoch`, `best_val_acc`, and
        `train_history` so the training loop can resume seamlessly.

        If no checkpoint exists, returns defaults for a fresh run and logs
        that a new session is starting.
        """
        if not os.path.exists(self.latest_path):
            print("[CheckpointManager] No checkpoint found. Initializing new training session from Epoch 1.")
            return {
                "start_epoch": 1,
                "best_val_acc": 0.0,
                "train_history": {
                    "train_loss": [], "val_loss": [],
                    "train_acc": [], "val_acc": [],
                },
            }

        ckpt = torch.load(self.latest_path, map_location=map_location)
        model.load_state_dict(ckpt["model_state_dict"])

        if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt["best_val_acc"]

        print(
            f"[CheckpointManager] Resuming training seamlessly from Epoch "
            f"{start_epoch} with previous Best Val Acc {best_val_acc * 100:.2f}%"
        )

        return {
            "start_epoch": start_epoch,
            "best_val_acc": best_val_acc,
            "train_history": ckpt.get("train_history", {
                "train_loss": [], "val_loss": [],
                "train_acc": [], "val_acc": [],
            }),
        }

    # ------------------------------------------------------------------ #
    def load_best(self, model: torch.nn.Module, map_location: str = "cpu") -> None:
        """Load best_model.pth weights into `model` (e.g. for final test-set eval)."""
        if not os.path.exists(self.best_path):
            raise FileNotFoundError(f"No best checkpoint found at {self.best_path}")
        ckpt = torch.load(self.best_path, map_location=map_location)
        model.load_state_dict(ckpt["model_state_dict"])
        print(
            f"[CheckpointManager] Loaded best model from epoch {ckpt['epoch']} "
            f"(val acc {ckpt['best_val_acc'] * 100:.2f}%)."
        )
