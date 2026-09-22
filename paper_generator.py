"""
paper_generator.py
===================
Section 7: IEEE Paper Section Generator.

Produces draft Markdown text -- Abstract, Methodology (with LaTeX-ready
equations), and an Experimental Setup / Results table skeleton -- meant
to be copy-edited and pasted into an IEEE conference/journal LaTeX
template (e.g. IEEEtran.cls).

This is a first-draft generator, not a substitute for actually writing
and validating the paper. Fill in the bracketed placeholders with the
real numbers once training completes.
"""

from typing import Dict


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}\\%" if x == x else "N/A"


def generate_abstract(cfg, report: Dict) -> str:
    acc = report.get("overall_test_accuracy", None)
    macro_f1 = report.get("macro_f1", None)

    acc_str = (
        _fmt_pct(acc)
        if acc is not None
        else "[TEST\\_ACCURACY]"
    )

    f1_str = (
        f"{macro_f1:.3f}"
        if macro_f1 is not None
        else "[MACRO\\_F1]"
    )

    return f"""## Abstract

**Objectives:** Automated multi-class ocular disease screening from color
fundus photography remains constrained by the trade-off between diagnostic
accuracy and clinical interpretability. This work investigates a
four-class retinal disease classification task involving Glaucoma,
Cataracts, Diabetic Retinopathy, and Normal retinal images, with emphasis
on cross-dataset generalization and interpretable predictions.

**Proposed Method:** We introduce Retinal-SwinNet, a hybrid architecture
that couples a pre-trained EfficientNet-B3 branch for local lesion-level
feature extraction with a pre-trained Swin-Transformer branch for global
structural context through shifted-window self-attention. A bidirectional
cross-attention fusion layer allows convolutional and transformer
representations to mutually query one another before classification.
Inputs are enhanced using a CLAHE-based preprocessing pipeline applied
to the extracted green channel, combined with circular fundus ROI
processing, prior to model ingestion. Training uses a class-weighted
focal loss with label smoothing to address class imbalance and reduce
overconfident predictions on ambiguous cases.

**Results:** On the independent held-out test dataset, Retinal-SwinNet
achieved an overall Top-1 accuracy of {acc_str} with a macro-averaged
F1-score of {f1_str}. Grad-CAM overlays from the CNN branch combined
with token-saliency maps from the Swin branch provide per-case visual
explanations highlighting diagnostically relevant retinal structures.

**Clinical Significance:** By combining hybrid visual feature extraction
with visual explainability, Retinal-SwinNet is designed as a
research-oriented screening-support framework that enables case-level
inspection of model predictions rather than relying solely on
black-box classification.
"""


def generate_methodology(cfg) -> str:
    return r"""## Proposed Methodology

### A. Adaptive CLAHE Preprocessing

Given an RGB fundus image $I \in \mathbb{R}^{H \times W \times 3}$, we
extract the green channel $I_g = I[:,:,1]$, which provides strong vascular
and retinal structural contrast in fundus photography. Contrast Limited
Adaptive Histogram Equalization is applied locally over $8 \times 8$
tiles with a clip limit $\tau = 2.0$:

$$
I_g^{clahe}(x,y) =
\text{CLAHE}_{\tau=2.0,\,T=8\times8}\big(I_g(x,y)\big)
$$

The enhanced channel is broadcast across all three channels:

$$
I' =
[I_g^{clahe}, I_g^{clahe}, I_g^{clahe}]
$$

This preserves compatibility with ImageNet-pretrained backbones while
emphasizing retinal structural information.

### B. Cross-Attention Feature Fusion

Let $F_{CNN} \in \mathbb{R}^{H_c \times W_c \times C_c}$ be the spatial
feature map from the EfficientNet-B3 branch and
$F_{Swin} \in \mathbb{R}^{H_s \times W_s \times C_s}$ the token grid from
the Swin-Transformer branch.

Both representations are projected to a shared embedding dimension
$d$ using $1{\times}1$ convolutions and flattened into token sequences:

$$
T_{CNN} \in \mathbb{R}^{N_c \times d},
\qquad
T_{Swin} \in \mathbb{R}^{N_s \times d}
$$

Bidirectional multi-head cross-attention is then applied:

$$
\hat{T}_{CNN} =
\text{LN}\Big(
T_{CNN} +
\text{MHA}(Q=T_{CNN},K=T_{Swin},V=T_{Swin})
\Big)
$$

$$
\hat{T}_{Swin} =
\text{LN}\Big(
T_{Swin} +
\text{MHA}(Q=T_{Swin},K=T_{CNN},V=T_{CNN})
\Big)
$$

The attended sequences are globally average-pooled and concatenated:

$$
F_{Fused} =
\left[
\text{GAP}(\hat{T}_{CNN})
\Vert
\text{GAP}(\hat{T}_{Swin})
\right]
\in \mathbb{R}^{512}
$$

The fused representation is passed through the classifier:

Dropout(0.4)
$\rightarrow$ Linear(512, 128)
$\rightarrow$ ReLU
$\rightarrow$ Dropout(0.2)
$\rightarrow$ Linear(128, 4).

The four output classes are Glaucoma, Cataracts,
Diabetic Retinopathy, and Normal.

### C. Class-Weighted Focal Loss with Label Smoothing

To address class imbalance and reduce overconfident predictions on
ambiguous retinal images, class-weighted focal loss is combined with
label smoothing.

For four diagnostic classes:

$$
q'(k) =
(1-\epsilon)q(k) + \frac{\epsilon}{K},
\qquad K=4
$$

where $\epsilon=0.1$ is the label-smoothing factor.

The focal loss is defined as:

$$
\mathcal{L}_{focal}
=
-\alpha_y(1-p_y)^\gamma
\sum_{k=1}^{K}q'(k)\log p(k)
$$

where $p_y$ is the predicted probability of the ground-truth class,
$\alpha_y$ is the inverse-frequency class weight, and
$\gamma=2.0$ is the focal-loss focusing parameter.

### D. Dataset Strategy

Datasets A and B are combined to form the training pool. This combined
dataset is divided into training and validation subsets using a
stratified split.

Dataset C is maintained as a completely independent test dataset.
Images from Dataset C are not used during training, validation,
hyperparameter tuning, or model selection.

This design evaluates the ability of Retinal-SwinNet to generalize
across datasets rather than only measuring performance on images drawn
from the same source as the training data.
"""


def generate_experimental_setup(cfg, report: Dict) -> str:
    per_class = report.get("per_class_metrics", {})
    auc = report.get("per_class_roc_auc", {})

    rows = []

    for name in cfg.class_names:
        m = per_class.get(name, {})
        a = auc.get(name, float("nan"))

        rows.append(
            f"| {name.replace('_', ' ')} "
            f"| {m.get('precision', float('nan')):.3f} "
            f"| {m.get('recall', float('nan')):.3f} "
            f"| {m.get('specificity', float('nan')):.3f} "
            f"| {m.get('f1_score', float('nan')):.3f} "
            f"| {a:.3f} |"
        )

    table_rows = (
        "\n".join(rows)
        if rows
        else "| [fill in after training] | | | | | |"
    )

    overall_acc = report.get("overall_test_accuracy", None)

    overall_acc_str = (
        _fmt_pct(overall_acc)
        if overall_acc is not None
        else "[TEST_ACC]"
    )

    return f"""## Experimental Setup

**Dataset split:** Datasets A and B are combined to form the training
dataset. A {cfg.val_split*100:.0f}% stratified validation split is created
from this combined training pool. Dataset C is reserved exclusively for
final independent testing.

**Classes:** Glaucoma, Cataracts, Diabetic Retinopathy, and Normal.

**Backbones:** EfficientNet-B3 (CNN branch), {cfg.swin_backbone_name}
(Transformer branch), both ImageNet-pretrained.

**Optimizer:** AdamW, differential learning rates
(head: {cfg.lr_head}, backbone: {cfg.lr_backbone}), weight decay
{cfg.weight_decay}. Scheduler: CosineAnnealingWarmRestarts
($T_0={cfg.scheduler_T0}$, $T_{{mult}}={cfg.scheduler_Tmult}$).

**Loss:** Class-weighted focal loss ($\\gamma={cfg.focal_gamma}$) with
label smoothing ($\\epsilon={cfg.label_smoothing}$).

### Comparative Results Table

Overall Top-1 Independent Test Accuracy: **{overall_acc_str}**

| Class | Precision | Recall (Sens.) | Specificity | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
{table_rows}

*Table: Per-class diagnostic performance of Retinal-SwinNet on the
completely independent test dataset (Dataset C). Comparative baseline
results should be added after the corresponding baseline experiments
have been conducted.*
"""


def generate_ieee_sections(cfg, report: Dict = None) -> str:
    """
    Assemble the full IEEE paper draft.

    `report` should be the dictionary returned by
    `metrics.full_evaluation_report`.

    If no report is supplied, placeholders are used so the function
    can run before training completes.
    """

    report = report or {}

    sections = [
        "# Retinal-SwinNet: A Hybrid CNN-Swin Transformer Architecture with "
        "Adaptive CLAHE Preprocessing for Multi-Class Ocular Disease "
        "Diagnostics\n",

        generate_abstract(
            cfg,
            report,
        ),

        generate_methodology(
            cfg,
        ),

        generate_experimental_setup(
            cfg,
            report,
        ),

        "\n---\n"
        "*Draft generated automatically by `paper_generator.py`. "
        "Verify every numeric claim against `outputs/` before submission; "
        "add related work, limitations, and ethical/regulatory statements "
        "before IEEE submission. This system is a research prototype and "
        "not a certified diagnostic device.*",
    ]

    return "\n".join(sections)