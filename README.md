# BigVul — Semi-Supervised Software Vulnerability Detection

Benchmarking Random Forest, SVM, and CNN for detecting vulnerabilities in C/C++ source code under a realistic low-label constraint, using the BigVul dataset.

---

## 📋 Overview

Automated vulnerability detection is critical in safety-sensitive software systems, yet practical deployments are often bottlenecked by the scarcity of labelled training data. This project simulates that constraint directly: models are trained on only 10% of available vulnerable samples, then evaluated against the remaining 90% — reflecting the real-world scenario where an organisation can only provide a small annotated codebase.

The BigVul dataset contains 188,636 C/C++ functions sourced from real CVE records, each labelled as vulnerable (`vul=1`) or non-vulnerable (`vul=0`). The class distribution is heavily skewed — vulnerable functions represent under 5% of the data — making overall accuracy a misleading metric and recall on the minority class the true measure of model utility.

Three classifiers are developed and compared: a Random Forest, a Support Vector Machine, and a 1D Convolutional Neural Network. All three represent source code as token sequences, using a C-aware lexer to extract syntactically meaningful tokens before training.

---

## 📁 Project Structure

```
bigvul/
├── main.py            # Pipeline entry point — runs all stages in order
├── 1_preprocess.py    # Data loading, splitting, tokenisation, vocab, padding
├── 2_model_rf.py      # Random Forest with randomised hyperparameter search
├── 3_model_svm.py     # SVM with randomised hyperparameter search
├── 4_model_cnn.py     # CNN with trainable embeddings and early stopping
├── utils.py           # Shared evaluation, plotting, and timing utilities
├── data/              # Preprocessed arrays saved by 1_preprocess.py (auto-created)
└── outputs/           # Model weights, confusion matrices, training curves (auto-created)
```

---

## 🔄 Pipeline

**Preprocessing** — Raw C/C++ functions are cleaned (comments stripped, numeric literals normalised) and tokenised using Pygments' `CLexer`, which understands C syntax and produces semantically meaningful tokens. A vocabulary is built by keeping only tokens at or above the 95th frequency percentile, reducing noise from rare identifiers. Sequences are then padded or truncated to the 98th percentile length. All outputs are saved to `data/` for reuse across model runs.

**Code Representation** — Random Forest and SVM operate on Bag-of-Words vectors built from the filtered vocabulary. The CNN uses a trainable embedding layer (dim=100) that learns a dense representation of each token during training, allowing it to capture semantic relationships that BoW misses.

**Class Imbalance** — With a roughly 16:1 non-vulnerable to vulnerable ratio in the training split, uncorrected training produces models that simply predict the majority class. All three models apply balanced class weighting to penalise misclassification of the minority class proportionally.

**Models**

- *Random Forest* — BoW features with `RandomizedSearchCV` over tree depth, estimator count, minimum split size, and class weighting strategy (100 iterations, 3-fold CV).
- *SVM* — BoW features with `RandomizedSearchCV` over kernel type, regularisation strength `C`, and gamma (10 iterations, 3-fold CV).
- *CNN* — Embedding → Conv1D (128 filters, kernel=5) → GlobalMaxPooling → Dense (64) → Dropout (0.5) → sigmoid. Trained with Adam (`lr=0.001`), binary cross-entropy loss, and early stopping on validation loss.

---

## 📊 Results

| Model | Train Acc | Test Acc | Vuln. Precision | Vuln. Recall | Vuln. F1 | Train Time |
|-------|-----------|----------|-----------------|--------------|----------|------------|
| Random Forest | 0.96 | 0.94 | 0.68 | 0.09 | 0.16 | ~37 min |
| SVM | 0.95 | 0.94 | 0.65 | 0.05 | 0.09 | ~10 hrs |
| **CNN** | **0.98** | **0.98** | **0.78** | **0.91** | **0.84** | ~135 min |

Overall accuracy is a poor discriminator here — RF and SVM both reach 94% by overwhelmingly predicting non-vulnerable code. The critical metric is recall on the vulnerable class: RF misses 91% of real vulnerabilities; SVM misses 95%. The CNN, with learned embeddings and class-aware training, achieves 0.91 recall and an F1 of 0.84 on the minority class — the only model suitable for real-world vulnerability screening. The tradeoff is inference speed, though well within acceptable bounds given the accuracy gains.

---

## 🖥️ Usage

```bash
# Run the full pipeline
python main.py

# Skip preprocessing if data/ already exists
python main.py --skip-preprocess

# Run a single model
python main.py --model cnn   # or: rf | svm
```

---

## ⚙️ Requirements

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn pygments tqdm
```

> The BigVul dataset (`df_preprocessed.csv`) is not included due to size. The original dataset is available from the [MSR repository](https://github.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset), with a preprocessed split-function CSV available via [Google Drive](https://drive.google.com/file/d/1-0VhnHBp9IGh90s2wCNjeCMuy70HPl8X/view?usp=sharing) (1.54 GB).
