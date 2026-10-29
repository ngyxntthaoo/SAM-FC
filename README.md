# SAM-FC: Slot-Attributed Multi-Task Learning for Fact-Checking

Official source code and reproduction package for the paper:  
**"Slot-Attributed Multi-Task Learning for Fact-Checking"** (*ACIIDS 2027*).

---

## 📌 Overview

Automated fact verification typically classifies a claim--evidence pair as **Supported**, **Refuted**, or **Not Enough Information (NEI)** as a whole, without revealing which attribute in a refuted claim conflicts with the evidence. 

**SAM-FC** (Slot-Attributed Multi-Task Learning for Fact-Checking) introduces a pair-level verifier that:
1. **Marks Claim-Evidence Divergences**: Highlights claim words absent from the evidence using `<diff>` tokens at the syllable level.
2. **Joint Pair Pooling**: Pools claim and evidence token embeddings into a unified relation representation $[u; v; |u-v|; u \odot v]$.
3. **Auxiliary Conflict-Slot Supervision**: Jointly predicts fine-grained conflict types (**Numeric**, **Negation**, **Entity**, **Temporal**) on Refuted claims via rule-derived weak supervision.
4. **Conflict-Gated Modulation**: Dynamically modulates the verdict representation using a learned conflict gate $g = \sigma(W_g h_{\text{slot}} + b_g)$.
5. **Multi-Task Objective**: Trains with Refuted class weighting ($w=1.20$), label smoothing ($\epsilon=0.10$), and a logic-consistency loss penalizing contradictory predictions.

Evaluated on multi-domain news (**ViFactCheck**, $n=1{,}447$) and encyclopedic text (**ViWikiFC**, $n=2{,}091$), SAM-FC achieves state-of-the-art Macro-F1 scores (**89.50%** and **87.50%**), outperforming existing Vietnamese fact verifiers including NLI-adapted ViCLSR.

---

## 📂 Repository Structure

```text
SAM-FC/
├── README.md                      # Comprehensive guide and reproduction instructions
├── requirements.txt               # Python package dependencies
├── data/
│   └── dataset_raw/               # Raw train/dev/test CSV datasets
│       ├── train-vifactcheck.csv
│       ├── dev-vifactcheck.csv
│       ├── test-vifactcheck.csv
│       ├── train-factcheckwiki.csv
│       ├── dev-factcheckwiki.csv
│       └── test-factcheckwiki.csv
├── src/
│   ├── __init__.py
│   ├── diff_marker.py             # <diff> marker insertion & pair tokenization
│   ├── slot_detector.py           # Rule-based weak supervision for conflict slot labels
│   ├── models.py                  # SAM-FC PyTorch architecture with gated modulation
│   ├── losses.py                  # Multi-task loss (Verdict + Slot + Logic consistency)
│   ├── dataset.py                 # MultiTaskSample loading & PyTorch DataLoader
│   ├── trainer.py                 # Training loop with warmup, differential lr, & evaluation
│   └── metrics.py                 # Macro-F1, per-class F1, Slot accuracy, & Inconsistency Rate
├── scripts/
│   ├── train_main.py              # Train SAM-FC model CLI
│   ├── run_ablation.py            # Run ablation study configurations
│   ├── reproduce_table2.py        # Reproduce Table 2: Main verification results
│   ├── reproduce_table3.py        # Reproduce Table 3: Ablation study results
│   ├── reproduce_table4.py        # Reproduce Table 4: Claim length stratification
│   ├── reproduce_table5.py        # Reproduce Table 5: Cross-domain conflict attribution
│   ├── plot_confusion_matrix.py   # Generate Figure 4: Cross-domain confusion matrices
│   └── plot_hyperparameter_tuning.py # Generate Figure 3: Loss weight sensitivity heatmaps
├── notebooks/
│   ├── 01_sam_fc_main_colab.ipynb       # Colab notebook for Main Results training & ensemble
│   ├── 02_sam_fc_ablation_colab.ipynb   # Colab notebook for Ablation Study
│   ├── 03_hyperparameter_tuning_lambdas_colab.ipynb # Colab notebook for (lambda_1, lambda_2) tuning
│   ├── 04_verdict_loss_weight_tuning_colab.ipynb     # Colab notebook for (w_refuted, epsilon) tuning
│   └── 05_error_analysis.ipynb          # Interactive Error Analysis notebook
└── outputs/
    ├── predictions/               # Saved model predictions and test metrics
    │   ├── vifactcheck_test_predictions_ensemble.json
    │   ├── viwikifc_test_predictions_ensemble.json
    │   ├── vifactcheck_test_metrics_ensemble.json
    │   └── viwikifc_test_metrics_ensemble.json
    ├── ablation/                  # Ablation evaluation JSON results
    │   ├── vifactcheck_ablation_result.json
    │   └── viwikifc_ablation_results.json
    └── figures/                   # Rendered publication figures (PNG and PDF)
        ├── cross_domain_confusion_matrix.png
        ├── cross_domain_confusion_matrix.pdf
        ├── hyperparameter_tuning_lambdas.png
        └── hyperparameter_tuning_lambdas.pdf
```

---

## 🚀 Quick Start & Installation

### 1. Environment Setup

Python 3.9+ is recommended. Clone the repository and install the dependencies:

```bash
git clone https://github.com/ngyxntthaoo/SAM-FC.git
cd SAM-FC

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📊 One-Click Paper Reproduction

All evaluation scripts read from the exact test predictions and outputs generated by the models in the paper.

### Reproduce Table 2: Main Results
Compares SAM-FC against reported and reproduced baselines on ViFactCheck and ViWikiFC:
```bash
python scripts/reproduce_table2.py
```

### Reproduce Table 3: Ablation Study
Evaluates the contribution of NLI intermediate adaptation, logic-consistency loss, `<diff>` markers, and the auxiliary conflict-slot branch:
```bash
python scripts/reproduce_table3.py
```

### Reproduce Table 4: Claim Length Stratification
Evaluates test accuracy and Macro-F1 across claim length bins (Short $\le 20$, Medium $21-35$, Long $>35$ words):
```bash
python scripts/reproduce_table4.py
```

### Reproduce Table 5: Cross-Domain Conflict-Slot Attribution
Breaks down Refuted instances by conflict type (Numeric, Negation, Entity, Temporal) on both benchmarks ($n=468$ and $n=706$):
```bash
python scripts/reproduce_table5.py
```

### Reproduce Figure 3: Hyperparameter Sensitivity
Generates the $(\lambda_1, \lambda_2)$ sensitivity heatmap and contradiction trade-off curve (saved to `outputs/figures/`):
```bash
python scripts/plot_hyperparameter_tuning.py
```

### Reproduce Figure 4: Cross-Domain Confusion Matrices
Generates row-normalized recall confusion matrices for both benchmarks (saved to `outputs/figures/`):
```bash
python scripts/plot_confusion_matrix.py
```

---

## 🛠️ Model Training & Custom Runs

### Training SAM-FC

To train SAM-FC from scratch on ViFactCheck:
```bash
python scripts/train_main.py \
    --dataset vifactcheck \
    --backbone xlm-roberta-large \
    --epochs 8 \
    --batch_size 4 \
    --grad_accum 4 \
    --lr_backbone 5e-6 \
    --lr_heads 1e-4 \
    --lambda_conflict 0.35 \
    --lambda_consistency 0.20 \
    --refuted_weight 1.20 \
    --label_smoothing 0.10 \
    --seed 42
```

To train on ViWikiFC:
```bash
python scripts/train_main.py --dataset viwikifc --epochs 8 --batch_size 4
```

### Running Ablation Configurations

To evaluate an individual ablation setting:
```bash
# w/o logic consistency loss
python scripts/run_ablation.py --dataset vifactcheck --config wo_consistency

# w/o diff markers
python scripts/run_ablation.py --dataset vifactcheck --config wo_diff_markers

# w/o auxiliary conflict branch (single-task verdict classification)
python scripts/run_ablation.py --dataset vifactcheck --config wo_auxiliary_conflict_branch
```

---

## ☁️ Google Colab Notebooks

Self-contained Colab notebooks are provided in `notebooks/`:
* `01_sam_fc_main_colab.ipynb`: End-to-end training, checkpoint saving, and 5-seed ensembling.
* `02_sam_fc_ablation_colab.ipynb`: Standalone ablation runner across all Table 3 configurations.
* `03_hyperparameter_tuning_lambdas_colab.ipynb`: Grid search for $\lambda_1 \in [0.20, 0.50]$ and $\lambda_2 \in [0.00, 0.30]$.
* `04_verdict_loss_weight_tuning_colab.ipynb`: Sensitivity search for Refuted class weight $w \in [1.0, 1.5]$ and label smoothing $\epsilon \in [0.0, 0.15]$.
* `05_error_analysis.ipynb`: Interactive Jupyter analysis reproducing Tables 4 & 5 and Figure 4.

---

## 📖 Citation

If you find SAM-FC helpful in your research, please cite our paper:

```bibtex
@inproceedings{samfc2027aciids,
  title     = {Slot-Attributed Multi-Task Learning for Fact-Checking},
  author    = {Anonymous Authors},
  booktitle = {Proceedings of the 19th Asian Conference on Intelligent Information and Database Systems (ACIIDS 2027)},
  year      = {2027},
  publisher = {Springer}
}
```
