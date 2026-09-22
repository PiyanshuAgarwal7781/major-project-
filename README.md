# Retinal-SwinNet

A Hybrid CNN-Swin Transformer Architecture with Adaptive CLAHE Preprocessing
for Multi-Class Ocular Disease Diagnostics (Glaucoma / Cataracts / Diabetic
Retinopathy / AMD / Normal), built for an ECE major project + IEEE paper
submission.

## Status

All modules have been individually smoke-tested (forward pass, loss
backward pass, checkpoint save/resume cycle, preprocessing on a synthetic
image, metrics/plotting on synthetic predictions) and compile cleanly.
**No public fundus dataset was trained on** — you need to point `--data_root`
at your own class-per-folder dataset before running.

## File Map

| File | Section | Purpose |
|---|---|---|
| `config.py` | — | Single source of truth for every hyperparameter/path |
| `preprocessing.py` | 2 | ROI crop, green-channel CLAHE, augmentation, `RetinalPreprocessTransform` |
| `dataset.py` | — | `RetinalFundusDataset`, stratified split, class-weight computation |
| `model.py` | 3 | `RetinalSwinNet`: EfficientNet-B3 + Swin-T/B + cross-attention fusion + head |
| `losses.py` | 4 | `FocalLossWithLabelSmoothing` (class-weighted) |
| `checkpoint.py` | 5 | `CheckpointManager`: per-epoch save, best-model isolation, auto-resume |
| `xai.py` | 6 | Grad-CAM (CNN branch) + Swin saliency fused into `generate_attention_heatmap` |
| `metrics.py` | 6 | Accuracy, per-class precision/recall/specificity/F1, confusion matrix PNG, ROC-AUC curves |
| `paper_generator.py` | 7 | Draft IEEE Abstract / Methodology (with LaTeX equations) / Results table generator |
| `train.py` | — | Orchestrates everything end-to-end |

## Expected Data Layout

Dataset A and Dataset B must be combined inside `data/train/`. Dataset C
must be kept separately inside `data/test/` and used only for final testing.

```
data/
  train/                  # Dataset A + Dataset B
    Glaucoma/
    Cataracts/
    Diabetic_Retinopathy/
    AMD/
    Normal/
  test/                   # Independent Dataset C
    Glaucoma/
    Cataracts/
    Diabetic_Retinopathy/
    AMD/
    Normal/
```

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python train.py --train_root ./data/train --test_root ./data/test --epochs 100 --swin_variant swin_t
```

Re-running the same command after any interruption auto-resumes from
`checkpoints/latest_checkpoint.pth`. Outputs land in `outputs/`:
`confusion_matrix.png`, `roc_curves.png`, `sample_attention_overlay.png`,
`ieee_paper_draft.md`.

## Notes for the Paper / Viva

- `EfficientNet-B3` and `Swin-T`/`Swin-B` weights load via
  `torchvision.models` `*_Weights.DEFAULT` (ImageNet-pretrained) — first
  run needs internet access to download them once; they're cached after.
- `swin_t` is the default (faster to train / lower VRAM); switch to
  `swin_b` for a stronger baseline once `swin_t` results are sane, and
  report both as an ablation.
- The 94.5% accuracy target in the spec is a *design goal*, not a
  guarantee — actual results depend entirely on your dataset size/quality.
  Report whatever you actually measure; `paper_generator.py` pulls real
  numbers from `metrics.full_evaluation_report()` automatically, so the
  abstract won't silently print a fabricated number once you run training.
- Before IEEE submission: add a related-work section, an ablation table
  (CNN-only vs Swin-only vs fused), and an explicit statement that this is
  a research prototype, not a regulatory-cleared diagnostic device.
