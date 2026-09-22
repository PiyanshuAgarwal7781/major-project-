"""
paper_generator.py
===================
Section 7: IEEE Paper Section Generator.

Produces draft Markdown text -- Abstract, Methodology (with LaTeX-ready
equations), and an Experimental Setup / Results table skeleton -- meant
to be copy-edited and pasted into an IEEE conference/journal LaTeX
template (e.g. IEEEtran.cls). This is a *first draft generator*, not a
substitute for actually writing and validating the paper: fill in the
bracketed placeholders with your real numbers once training completes.
"""

from typing import Dict


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}\\%" if x == x else "N/A"  # x == x is a NaN guard


def generate_abstract(cfg, report: Dict) -> str:
    acc = report.get("overall_test_accuracy", None)
    macro_f1 = report.get("macro_f1", None)
    acc_str = _fmt_pct(acc) if acc is not None else "[TEST\\_ACCURACY]"
    f1_str = f"{macro_f1:.3f}" if macro_f1 is not None else "[MACRO\\_F1]"

    return f"""## Abstract

**Objectives:** Automated multi-class ocular disease screening from color
fundus photography remains constrained by the trade-off between diagnostic
accuracy and clinical interpretability. This work targets simultaneous
improvement of both, elevating standard CNN/ViT baseline accuracy
(~88-90%) beyond 94.5% on a five-class diagnostic task spanning Glaucoma,
Cataracts, Diabetic Retinopathy, Age-Related Macular Degeneration, and
Normal controls.

**Proposed Method:** We introduce Retinal-SwinNet, a hybrid architecture
that couples a pre-trained EfficientNet-B3 branch (local lesion-level
feature extraction) with a pre-trained Swin-Transformer branch (global
structural context via shifted-window self-attention). A bidirectional
cross-attention fusion layer allows convolutional and transformer tokens
to mutually query one another before classification. Inputs are enhanced
with an adaptive CLAHE pipeline applied to the extracted green channel,
combined with circular fundus ROI auto-cropping, prior to model ingestion.
Training uses a class-weighted focal loss with label smoothing to address
class imbalance and reduce overconfidence on ambiguous cases.

**Results:** On the held-out test split, Retinal-SwinNet achieved an
overall Top-1 accuracy of {acc_str} with a macro-averaged F1-score of
{f1_str}. Grad-CAM overlays from the CNN branch combined with
token-saliency maps from the Swin branch provide per-case visual
explanations aligned with clinically relevant structures (optic disc
margins, vascular exudates).

**Clinical Significance:** By pairing high diagnostic accuracy with
built-in visual explainability, Retinal-SwinNet is positioned as a
screening-support tool that ophthalmologists can audit case-by-case,
rather than a black-box classifier -- a prerequisite for realistic
clinical adoption.
"""


def generate_methodology(cfg) -> str:
    return r"""## Proposed Methodology

### A. Adaptive CLAHE Preprocessing

Given an RGB fundus image $I \in \mathbb{R}^{H \times W \times 3}$, we
extract the green channel $I_g = I[:,:,1]$, which exhibits the highest
vascular and lesion contrast in fundus photography. Contrast Limited
Adaptive Histogram Equalization is applied locally over $8 \times 8$ tiles
with a clip limit $\tau = 2.0$:

$$
I_g^{clahe}(x,y) = \text{CLAHE}_{\tau=2.0,\, T=8\times8}\big(I_g(x,y)\big)
$$

The enhanced channel is broadcast back across all three channels,
$I' = [I_g^{clahe}, I_g^{clahe}, I_g^{clahe}]$, preserving compatibility
with ImageNet-pretrained backbones while foregrounding the highest-signal
structural information.

### B. Cross-Attention Feature Fusion

Let $F_{CNN} \in \mathbb{R}^{H_c \times W_c \times C_c}$ be the spatial
feature map from the EfficientNet-B3 branch and
$F_{Swin} \in \mathbb{R}^{H_s \times W_s \times C_s}$ the token grid from
the Swin-Transformer branch. Both are projected to a shared embedding
dimension $d$ via $1{\times}1$ convolutions, then flattened into token
sequences $T_{CNN} \in \mathbb{R}^{N_c \times d}$,
$T_{Swin} \in \mathbb{R}^{N_s \times d}$. Bidirectional multi-head
cross-attention is applied:

$$
\hat{T}_{CNN} = \text{LN}\Big(T_{CNN} + \text{MHA}\big(Q{=}T_{CNN},\, K{=}T_{Swin},\, V{=}T_{Swin}\big)\Big)
$$

$$
\hat{T}_{Swin} = \text{LN}\Big(T_{Swin} + \text{MHA}\big(Q{=}T_{Swin},\, K{=}T_{CNN},\, V{=}T_{CNN}\big)\Big)
$$

The attended sequences are globally average-pooled and concatenated into
the fused representation:

$$
F_{Fused} = \big[\, \text{GAP}(\hat{T}_{CNN})\, \Vert\, \text{GAP}(\hat{T}_{Swin}) \,\big] \in \mathbb{R}^{512}
$$

which is passed through the classifier head:
Dropout(0.4) $\rightarrow$ Linear(512, 128) $\rightarrow$ ReLU
$\rightarrow$ Dropout(0.2) $\rightarrow$ Linear(128, 5).

### C. Class-Weighted Focal Loss with Label Smoothing

To address class imbalance (e.g. fewer AMD samples than Normal) and
reduce overconfident predictions on ambiguous fundus edges, we combine
per-class inverse-frequency weighting $\alpha_y$, focal modulation
($\gamma = 2.0$), and label smoothing ($\epsilon = 0.1$):

$$
q'(k) = (1-\epsilon)\, q(k) + \frac{\epsilon}{K}, \qquad K = 5
$$

$$
\mathcal{L}_{focal} = -\alpha_y\, (1 - p_y)^{\gamma} \sum_{k=1}^{K} q'(k) \log p(k)
$$

where $p_y$ is the predicted probability of the ground-truth class $y$.
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
    table_rows = "\n".join(rows) if rows else "| [fill in after training] | | | | | |"

    overall_acc = report.get("overall_test_accuracy", None)
    overall_acc_str = _fmt_pct(overall_acc) if overall_acc is not None else "[TEST_ACC]"

    return f"""## Experimental Setup

**Dataset split:** Training and validation are created from the combined
training datasets using a {cfg.val_split*100:.0f}% validation split. A third
independent dataset is reserved exclusively for final testing.

**Backbones:** EfficientNet-B3 (CNN branch), {cfg.swin_backbone_name}
(Transformer branch), both ImageNet-pretrained.

**Optimizer:** AdamW, differential learning rates
(head: {cfg.lr_head}, backbone: {cfg.lr_backbone}), weight decay
{cfg.weight_decay}. Scheduler: CosineAnnealingWarmRestarts
($T_0={cfg.scheduler_T0}$, $T_{{mult}}={cfg.scheduler_Tmult}$).

**Loss:** Class-weighted focal loss ($\\gamma={cfg.focal_gamma}$) with
label smoothing ($\\epsilon={cfg.label_smoothing}$).

### Comparative Results Table

Overall Top-1 Test Accuracy: **{overall_acc_str}**

| Class | Precision | Recall (Sens.) | Specificity | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
{table_rows}

*Table: Per-class diagnostic performance of Retinal-SwinNet on the held-out
test set. [Insert a second table here comparing against baseline
EfficientNet-B3-only, Swin-only, and prior published benchmarks (~88-90%
accuracy) once those ablations are run.]*
"""


def generate_ieee_sections(cfg, report: Dict = None) -> str:
    """
    Assembles the full draft: Abstract, Methodology, Experimental Setup.
    `report` should be the dict returned by `metrics.full_evaluation_report`;
    if omitted, placeholders are used so the function can still run before
    training completes.
    """
    report = report or {}
    sections = [
        "# Retinal-SwinNet: A Hybrid CNN-Swin Transformer Architecture with "
        "Adaptive CLAHE Preprocessing for Multi-Class Ocular Disease Diagnostics\n",
        generate_abstract(cfg, report),
        generate_methodology(cfg),
        generate_experimental_setup(cfg, report),
        "\n---\n*Draft generated automatically by `paper_generator.py`. "
        "Verify every numeric claim against `outputs/` before submission; "
        "add related-work, limitations, and ethical/regulatory statements "
        "(this system is a research prototype, not a certified diagnostic "
        "device) before IEEE submission.*",
    ]
    return "\n".join(sections)
