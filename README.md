# SAM-FC: Slot-Attributed Multi-Task Learning for Fact-Checking

Official PyTorch implementation for **"Slot-Attributed Multi-Task Learning for Fact-Checking"** (*ACIIDS 2027*).

SAM-FC is an explainable pair-level verifier for Vietnamese fact verification. It incorporates syllable-level `<diff>` markers, joint pair pooling, and a conflict-gated modulation branch to jointly predict verdicts and identify fine-grained conflict types (**Numeric**, **Negation**, **Entity**, **Temporal**) on Refuted claims.

---

## ⚡ Quick Start

```bash
git clone https://github.com/ngyxntthaoo/SAM-FC.git
cd SAM-FC
pip install -r requirements.txt
```

---

## 📁 Datasets Setup

Due to dataset licensing and copyright policies of the respective original authors, raw dataset CSV files are not distributed directly in this repository. Please obtain the datasets from the official authors/releases as cited in the paper, and place the CSV files in `data/dataset_raw/`:

### 1. ViFactCheck (Multi-domain News)
* **Citation:** *ViFactCheck: A New Benchmark Dataset and Methods for Multi-Domain News Fact-Checking in Vietnamese* (AAAI 2025).  
  *Authors: T. T. Hoa, T. Q. Duy, K. Q. Tran, K. Van Nguyen.*
* **Files:** `train-vifactcheck.csv`, `dev-vifactcheck.csv`, `test-vifactcheck.csv`
* **Columns:** `Statement`, `Evidence`, `labels` (0: Supported, 1: Refuted, 2: NEI)

### 2. ViWikiFC (Vietnamese Wikipedia)
* **Citation:** *ViWikiFC: Fact-Checking for Vietnamese Wikipedia-Based Textual Knowledge Source* (LREC-COLING 2024 / arXiv).  
  *Authors: H. T. Le, L. T. To, M. T. Nguyen, K. Van Nguyen.*
* **Files:** `train-factcheckwiki.csv`, `dev-factcheckwiki.csv`, `test-factcheckwiki.csv`
* **Columns:** `claim`, `evidence`, `gold_label` (`Supports`, `Refutes`, `Not_Enough_Information`)

### Expected Directory Layout
```text
data/
└── dataset_raw/
    ├── train-vifactcheck.csv
    ├── dev-vifactcheck.csv
    ├── test-vifactcheck.csv
    ├── train-factcheckwiki.csv
    ├── dev-factcheckwiki.csv
    └── test-factcheckwiki.csv
```

---

## 📊 One-Click Paper Reproduction

All evaluation scripts directly reproduce the exact tables and figures in the paper from precomputed test predictions:

| Target | Command | Output Description |
| :--- | :--- | :--- |
| **Table 2** | `python scripts/reproduce_table2.py` | Main results vs. baselines (XLM-R, Vi2FC, LOREN, ViCLSR) |
| **Table 3** | `python scripts/reproduce_table3.py` | Ablation study (5 configurations on both benchmarks) |
| **Table 4** | `python scripts/reproduce_table4.py` | Performance stratified by claim length (Short, Medium, Long) |
| **Table 5** | `python scripts/reproduce_table5.py` | Conflict-slot attribution breakdown on Refuted claims |
| **Figure 3** | `python scripts/plot_hyperparameter_tuning.py` | Grid sensitivity of loss weights $(\lambda_1, \lambda_2)$ |
| **Figure 4** | `python scripts/plot_confusion_matrix.py` | Cross-domain confusion matrices (`outputs/figures/`) |

---

## 🛠️ Training & Custom Runs

Train SAM-FC from scratch:
```bash
# ViFactCheck
python scripts/train_main.py --dataset vifactcheck --epochs 8 --batch_size 4

# ViWikiFC
python scripts/train_main.py --dataset viwikifc --epochs 8 --batch_size 4
```

Run ablation experiments:
```bash
python scripts/run_ablation.py --dataset vifactcheck --config wo_consistency
python scripts/run_ablation.py --dataset vifactcheck --config wo_diff_markers
python scripts/run_ablation.py --dataset vifactcheck --config wo_auxiliary_conflict_branch
```

---

## ☁️ Google Colab Notebooks

Self-contained GPU notebooks are available in `notebooks/`:
* `01_sam_fc_main_colab.ipynb`: Training, checkpoints, and 5-seed ensembling.
* `02_sam_fc_ablation_colab.ipynb`: Ablation study across all Table 3 configurations.
* `03_hyperparameter_tuning_lambdas_colab.ipynb`: Grid search for $(\lambda_1, \lambda_2)$.
* `04_verdict_loss_weight_tuning_colab.ipynb`: Sensitivity search for $w_{\text{refuted}}$ and label smoothing.
* `05_error_analysis.ipynb`: Interactive error analysis.

---

## 📂 Repository Structure

```text
SAM-FC/
├── data/           # Dataset directory layout and raw CSV storage
├── src/            # Core models, loss functions, diff markers, and trainer
├── scripts/        # Reproduction scripts for Tables 2-5, Figures 3-4, and training
├── notebooks/      # Self-contained Google Colab notebooks
└── outputs/        # Precomputed test predictions, metrics, and high-res figures
```
