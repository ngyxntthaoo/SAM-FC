# Datasets for SAM-FC

Due to dataset licensing and copyright policies of the respective original authors, raw dataset CSV files are not distributed directly in this repository. Please obtain the datasets from the official authors/releases as cited in the paper, and place the CSV files in `data/dataset_raw/`.

---

## 1. ViFactCheck (Multi-domain News)

* **Paper:** *ViFactCheck: A New Benchmark Dataset and Methods for Multi-Domain News Fact-Checking in Vietnamese* (AAAI 2025).  
  *Authors: T. T. Hoa, T. Q. Duy, K. Q. Tran, K. Van Nguyen.*
* **Files needed in `data/dataset_raw/`:**
  * `train-vifactcheck.csv` (Train set, ~5,078 samples)
  * `dev-vifactcheck.csv` (Dev set, ~723 samples)
  * `test-vifactcheck.csv` (Test set, ~1,447 samples)
* **Expected columns:** `Statement`, `Evidence`, `labels` (0: Supported, 1: Refuted, 2: NEI).

---

## 2. ViWikiFC (Vietnamese Wikipedia Fact-Checking)

* **Paper:** *ViWikiFC: Fact-Checking for Vietnamese Wikipedia-Based Textual Knowledge Source* (LREC-COLING 2024 / arXiv).  
  *Authors: H. T. Le, L. T. To, M. T. Nguyen, K. Van Nguyen.*
* **Files needed in `data/dataset_raw/`:**
  * `train-factcheckwiki.csv` (Train set, ~16,738 samples)
  * `dev-factcheckwiki.csv` (Dev set, ~2,090 samples)
  * `test-factcheckwiki.csv` (Test set, ~2,091 samples)
* **Expected columns:** `claim`, `evidence`, `gold_label` (`Supports`, `Refutes`, `Not_Enough_Information`).

---

## 3. Directory Layout

Once downloaded, ensure your local directory structure matches:

```text
data/
├── README.md
└── dataset_raw/
    ├── train-vifactcheck.csv
    ├── dev-vifactcheck.csv
    ├── test-vifactcheck.csv
    ├── train-factcheckwiki.csv
    ├── dev-factcheckwiki.csv
    └── test-factcheckwiki.csv
```
