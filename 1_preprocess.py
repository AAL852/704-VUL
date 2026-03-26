"""
1_preprocess.py
---------------
Loads the BigVul dataset, splits into train/val/test sets, tokenises C/C++
function bodies using a C-aware lexer, builds a frequency-filtered vocabulary,
and pads sequences to a fixed length.

Outputs (saved to disk for reuse by model scripts):
    - data/X_train_padded.npy
    - data/X_val_padded.npy
    - data/X_test_padded.npy
    - data/y_train.npy
    - data/y_val.npy
    - data/y_test.npy
    - data/vocab.pkl
    - data/split_metadata.pkl  (vocab_size, max_length, class counts)
"""

import os
import re
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
from pygments.lexers import CLexer
from pygments import lex
from keras.preprocessing.sequence import pad_sequences

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH       = "data/df_preprocessed.csv"
OUTPUT_DIR      = "data"
RANDOM_SEED     = 42
TRAIN_RATIO     = 0.60
VAL_RATIO       = 0.20
VOCAB_PERCENTILE = 95   # keep tokens above this frequency percentile
PAD_PERCENTILE   = 98   # truncate/pad sequences to this length percentile

os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("Loading dataset...")
df = pd.read_csv(DATA_PATH, na_filter=False).reset_index(drop=True)
print(f"  {df.shape[0]:,} records, {df.shape[1]} columns")

X_raw = df["func_before"].tolist()
Y = np.array(df["vul"])
Y[Y > 0] = 1  # binarise: 0 = non-vulnerable, 1 = vulnerable


# ---------------------------------------------------------------------------
# 2. Train / val / test split (60 / 20 / 20)
# ---------------------------------------------------------------------------
n = len(X_raw)
idx = np.random.choice(n, n, replace=False)

n_train = int(round(n * TRAIN_RATIO))
n_val   = int(round(n * VAL_RATIO))

X_train = [X_raw[i] for i in idx[:n_train]]
y_train = Y[idx[:n_train]]

X_val   = [X_raw[i] for i in idx[n_train:n_train + n_val]]
y_val   = Y[idx[n_train:n_train + n_val]]

X_test  = [X_raw[i] for i in idx[n_train + n_val:]]
y_test  = Y[idx[n_train + n_val:]]

# Log class distribution
for name, y in [("train", y_train), ("val", y_val), ("test", y_test)]:
    unique, counts = np.unique(y, return_counts=True)
    print(f"  {name}: {dict(zip(unique, counts))}")

# Plot split distribution
labels = ["Training", "Validation", "Test"]
sizes  = [len(y_train), len(y_val), len(y_test)]
bars   = plt.bar(labels, sizes, color=["steelblue", "seagreen", "tomato"])
for bar in bars:
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
             str(bar.get_height()), ha="center", va="bottom")
plt.title("Dataset Split Distribution")
plt.ylabel("Number of Samples")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/split_distribution.png", dpi=150)
plt.close()


# ---------------------------------------------------------------------------
# 3. Tokenisation
# ---------------------------------------------------------------------------
def preprocess_code(code: str) -> str:
    """Strip comments, string literals, and normalise numeric literals."""
    code = re.sub(r"//.*",          "",  code)
    code = re.sub(r"/\*.*?\*/",     "",  code, flags=re.DOTALL)
    code = re.sub(r'".*?"',         "",  code)
    code = re.sub(r"\b[-+]?\d*\.?\d+\b", "NUMBER", code)
    return code.strip()


def clexer_tokenize(code: str) -> list:
    """Tokenise a C/C++ function body using Pygments CLexer."""
    lexer = CLexer()
    return [token[1] for token in lex(preprocess_code(code), lexer)]


print("\nTokenising training data...")
X_train_tok = [clexer_tokenize(c) for c in tqdm(X_train, desc="  train")]

token_freq = Counter(tok for toks in X_train_tok for tok in toks)
print(f"  Total unique tokens: {len(token_freq):,}")


# ---------------------------------------------------------------------------
# 4. Vocabulary — keep tokens at or above the 95th frequency percentile
# ---------------------------------------------------------------------------
frequencies       = list(token_freq.values())
freq_threshold    = np.percentile(frequencies, VOCAB_PERCENTILE)
frequent_tokens   = {t for t, f in token_freq.items() if f >= freq_threshold}
vocab             = {t: idx + 1 for idx, t in enumerate(frequent_tokens)}  # 0 reserved for padding
vocab_size        = len(vocab)
print(f"  Vocabulary size (≥{VOCAB_PERCENTILE}th percentile): {vocab_size:,}")


def encode(code: str) -> list:
    """Tokenise and convert to integer indices, filtering out-of-vocab tokens."""
    return [vocab[t] for t in clexer_tokenize(code) if t in vocab]


# ---------------------------------------------------------------------------
# 5. Encode all splits
# ---------------------------------------------------------------------------
print("\nEncoding all splits...")
X_train_seq = [encode(c) for c in tqdm(X_train, desc="  train")]
X_val_seq   = [encode(c) for c in tqdm(X_val,   desc="  val  ")]
X_test_seq  = [encode(c) for c in tqdm(X_test,  desc="  test ")]


# ---------------------------------------------------------------------------
# 6. Sequence length analysis and padding
# ---------------------------------------------------------------------------
all_lengths = [len(s) for s in X_train_seq + X_val_seq + X_test_seq]
max_length  = int(np.percentile(all_lengths, PAD_PERCENTILE))
print(f"\n  {PAD_PERCENTILE}th percentile sequence length: {max_length}")

X_train_padded = pad_sequences(X_train_seq, maxlen=max_length, padding="post", truncating="post")
X_val_padded   = pad_sequences(X_val_seq,   maxlen=max_length, padding="post", truncating="post")
X_test_padded  = pad_sequences(X_test_seq,  maxlen=max_length, padding="post", truncating="post")


# ---------------------------------------------------------------------------
# 7. Save to disk
# ---------------------------------------------------------------------------
np.save(f"{OUTPUT_DIR}/X_train_padded.npy", X_train_padded)
np.save(f"{OUTPUT_DIR}/X_val_padded.npy",   X_val_padded)
np.save(f"{OUTPUT_DIR}/X_test_padded.npy",  X_test_padded)
np.save(f"{OUTPUT_DIR}/y_train.npy",        y_train)
np.save(f"{OUTPUT_DIR}/y_val.npy",          y_val)
np.save(f"{OUTPUT_DIR}/y_test.npy",         y_test)

with open(f"{OUTPUT_DIR}/vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)

metadata = {"vocab_size": vocab_size, "max_length": max_length}
with open(f"{OUTPUT_DIR}/split_metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

print("\nPreprocessing complete. Outputs saved to data/")
