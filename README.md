# PSCSE: Prompt-based Contrastive Learning with Sample Filtering


## 📌 Overview

This repository contains the official implementation of **PSCSE**,
a prompt-based contrastive learning framework with sample filtering for **unsupervised sentence embedding**.

PSCSE improves sentence representation quality by:

* leveraging prompt-based encoding,
* applying contrastive learning objectives, and
* filtering low-quality or noisy samples during training.

The implementation is designed for **reproducibility** and **standard evaluation**, and follows common practices in sentence embedding research.

---

## 📂 Repository Structure

```text
PSCSE/
├── train.py              # Baseline / early training script
├── train_v2.py           # Intermediate version
├── train_v3.py           # Final training script (main results)
├── pscse_model/
│   ├── __init__.py
│   └── models.py         # Final PSCSE model implementation
├── SentEval/             # SentEval evaluation toolkit
├── requirements.txt
├── README.md
└── .gitignore
```

### Notes on the Codebase

* **`train_v3.py`** is the **final training script** used to produce all main experimental results reported in the paper.
* Other training scripts (`train.py`, `train_v2.py`) are kept for reference and reproducibility.
* Some experimental variants explored during development (e.g., SNCSE comparison and soft filtering variants) have been **removed**, as they are **not included in the final version of the paper**.

---

## 🧠 Model Implementation

All model definitions are located in:

```text
pscse_model/models.py
```

This file corresponds to the **final PSCSE model** used in all experiments.

> **Important clarification**
> Although PSCSE is inspired by general prompt-based sentence encoding ideas,
> this implementation **is NOT the original PromptBERT model**.
> The directory name `pscse_model` is intentionally used to avoid confusion with prior work.

---

## ⚙️ Environment Setup

### Requirements

The code is implemented in **Python** and depends on common deep learning and NLP libraries.

Install dependencies with:

```bash
pip install -r requirements.txt
```

> ⚠️ We recommend using a virtual environment (e.g., `conda` or `venv`) for reproducibility.

---

## 🚀 Training

To train the PSCSE model using the final configuration:

```bash
python train_v3.py
```

Key training hyperparameters (e.g., batch size, temperature, pooling strategy) can be configured via command-line arguments or directly in the script.

---

## 📊 Evaluation

We use **SentEval** for standard sentence embedding evaluation, including tasks such as:

* Semantic Textual Similarity (STS)
* Text classification benchmarks (e.g., SST, TREC)

The SentEval toolkit is included in this repository for convenience and reproducibility.

---

## 🔁 Reproducibility

* All reported results are obtained using `train_v3.py` and the model defined in `pscse_model/models.py`.
* Random seeds and evaluation protocols follow common practices in sentence embedding research.
* Training logs and intermediate checkpoints are intentionally **not included** in this repository.

---

---

## 📜 Acknowledgements

* This project uses the **SentEval** evaluation framework.
* We thank the authors of prior work on prompt-based sentence embedding for inspiring this research direction.

---

## 📬 Contact

If you have questions or encounter issues, feel free to open an issue or contact the authors.



