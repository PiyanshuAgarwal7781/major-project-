"""
train.py
========
Main entry point: wires together preprocessing, dataset, model, loss,
optimizer/scheduler, checkpointing, and post-training evaluation/XAI.

Run `python split_dataset.py` ONCE first to create data/splits/{train,val,test}.csv.

Usage:
    python train.py --epochs 30

Re-running the exact same command after an interruption will auto-resume
from `checkpoints/latest_checkpoint.pth` (Section 5).

Data protocol:
    train -> weight updates
    val   -> per-epoch monitoring + best-checkpoint selection
    test  -> touched ONLY once, after training, for the final report
"""

import argparse
import copy
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import CFG
from checkpoint import CheckpointManager
from dataset import make_subsets_from_manifest, compute_class_weights
from losses import build_criterion
from metrics import full_evaluation_report
from model import build_model
from preprocessing import build_transforms
from xai import generate_attention_heatmap
from paper_generator import generate_ieee_sections


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Retinal-SwinNet")
    parser.add_argument("--train_root", type=str, default=CFG.train_root)
    parser.add_argument("--splits_dir", type=str, default=CFG.splits_dir)
    parser.add_argument("--epochs", type=int, default=CFG.num_epochs)
    parser.add_argument("--batch_size", type=int, default=CFG.batch_size)
    parser.add_argument("--swin_variant", type=str, default=CFG.swin_backbone_name,
                         choices=["swin_t", "swin_b"])
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()

    total_loss, total_correct, total_samples = 0.0, 0, 0
    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for images, labels, _ in loader:
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            if train:
                loss.backward()
                optimizer.step()

            preds = logits.argmax(dim=1)
            total_loss += loss.item() * images.size(0)
            total_correct += (preds == labels).sum().item()
            total_samples += images.size(0)

    return total_loss / total_samples, total_correct / total_samples


def main():
    args = parse_args()
    set_seed(CFG.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Setup] Using device: {device}")

    # ------------------------------------------------------------------ #
    # Data  (train / val / test from data/splits/*.csv)
    # ------------------------------------------------------------------ #
    train_tf, eval_tf = build_transforms(CFG)
    train_set, val_set, test_set = make_subsets_from_manifest(
        train_root=args.train_root,
        splits_dir=args.splits_dir,
        class_names=CFG.class_names,
        train_transform=train_tf,
        eval_transform=eval_tf,
    )
    print(f"[Data] train={len(train_set)}  val={len(val_set)}  test={len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                               num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=CFG.num_workers, pin_memory=True)
    # Test loader is only used after training finishes.
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                              num_workers=CFG.num_workers, pin_memory=True)

    # Class weights computed on the *training split* only (avoids leakage).
    class_weights = compute_class_weights(train_set, CFG.num_classes)
    print(f"[Data] class weights (inverse frequency): {class_weights.tolist()}")

    # ------------------------------------------------------------------ #
    # Model / loss / optimizer / scheduler
    # ------------------------------------------------------------------ #
    CFG.swin_backbone_name = args.swin_variant
    model = build_model(CFG).to(device)

    criterion = build_criterion(CFG, class_weights).to(device)

    param_groups = model.get_param_groups(
        lr_head=CFG.lr_head, lr_backbone=CFG.lr_backbone, weight_decay=CFG.weight_decay
    )
    optimizer = torch.optim.AdamW(param_groups)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=CFG.scheduler_T0, T_mult=CFG.scheduler_Tmult
    )

    # ------------------------------------------------------------------ #
    # Checkpointing / auto-resume (Section 5)
    # ------------------------------------------------------------------ #
    ckpt_manager = CheckpointManager(
        checkpoint_dir=CFG.checkpoint_dir,
        latest_name=CFG.latest_ckpt_name,
        best_name=CFG.best_ckpt_name,
    )
    resume_state = ckpt_manager.load_latest(model, optimizer, scheduler, map_location=device)
    start_epoch = resume_state["start_epoch"]
    best_val_acc = resume_state["best_val_acc"]
    history = resume_state["train_history"]

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        print(
            f"[Epoch {epoch:03d}/{args.epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc*100:.2f}% | "
            f"val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}% | "
            f"{elapsed:.1f}s"
        )

        ckpt_manager.save(
            epoch=epoch, model=model, optimizer=optimizer, scheduler=scheduler,
            best_val_acc=best_val_acc, val_acc=val_acc, train_history=history,
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"  -> New best val acc: {best_val_acc*100:.2f}% (best_model.pth updated)")

    # ------------------------------------------------------------------ #
    # Final evaluation on held-out test split, using the best checkpoint
    # ------------------------------------------------------------------ #
    print("\n[Eval] Loading best checkpoint for final test-set evaluation...")
    ckpt_manager.load_best(model, map_location=device)

    report = full_evaluation_report(
        model=model,
        test_loader=test_loader,
        class_names=CFG.class_names,
        device=device,
        output_dir=CFG.output_dir,
    )

    # ------------------------------------------------------------------ #
    # XAI example on one held-out image (Section 6)
    # ------------------------------------------------------------------ #
    if len(test_set) > 0:
        sample_path, _ = test_set.samples[0]
        xai_result = generate_attention_heatmap(
            model=model,
            image_path=sample_path,
            transform=eval_tf,
            class_names=CFG.class_names,
            device=device,
            save_path=f"{CFG.output_dir}/sample_attention_overlay.png",
        )
        print(f"\n[XAI] Sample prediction: {xai_result['predicted_class']} "
              f"(confidence {xai_result['confidence']*100:.1f}%) "
              f"-> saved overlay to {CFG.output_dir}/sample_attention_overlay.png")

    # ------------------------------------------------------------------ #
    # IEEE paper section draft (Section 7)
    # ------------------------------------------------------------------ #
    paper_text = generate_ieee_sections(cfg=CFG, report=report)
    paper_path = f"{CFG.output_dir}/ieee_paper_draft.md"
    with open(paper_path, "w") as f:
        f.write(paper_text)
    print(f"[Paper] Draft IEEE sections written to {paper_path}")


if __name__ == "__main__":
    main()