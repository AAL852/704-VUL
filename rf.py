"""
2_model_rf.py
-------------
Random Forest classifier for C/C++ vulnerability detection.

Loads preprocessed data from 1_preprocess.py, vectorises using Bag-of-Words,
tunes hyperparameters via RandomizedSearchCV, and evaluates on the test set.
Results and plots are saved to outputs/.
"""

import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import RandomizedSearchCV

from utils import elapsed, load_splits, evaluate_ml_model
import time

os.makedirs("outputs", exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load preprocessed splits
# ---------------------------------------------------------------------------
print("Loading preprocessed data...")
splits = load_splits("data")

# RF and SVM operate on raw text — reload original string splits
# (padded arrays are used by CNN; RF/SVM use BoW over raw tokens)
import pickle
X_train = pickle.load(open("data/X_train_raw.pkl", "rb"))
X_val   = pickle.load(open("data/X_val_raw.pkl",   "rb"))
X_test  = pickle.load(open("data/X_test_raw.pkl",  "rb"))
y_train = splits["y_train"]
y_val   = splits["y_val"]
y_test  = splits["y_test"]

vocab_size = splits["vocab_size"]


# ---------------------------------------------------------------------------
# 2. Bag-of-Words vectorisation
# ---------------------------------------------------------------------------
print("Vectorising with Bag-of-Words...")
vectorizer        = CountVectorizer(max_features=vocab_size)
X_train_vec       = vectorizer.fit_transform(X_train)
X_val_vec         = vectorizer.transform(X_val)
X_test_vec        = vectorizer.transform(X_test)


# ---------------------------------------------------------------------------
# 3. Hyperparameter search
# ---------------------------------------------------------------------------
param_distributions = {
    "n_estimators":      [1, 5, 10, 15, 20, 25, 50, 100],
    "max_depth":         [1, 2, 4, 5, 6, 7, 8, 16, 32, 64, 128, 256],
    "min_samples_split": [2, 3, 4, 5, 6, 7, 8, 9, 10],
    "class_weight":      [None, "balanced", "balanced_subsample"],
}

rf = RandomForestClassifier(random_state=42)

search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_distributions,
    n_iter=100,
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1,
)

print("\nRunning RandomizedSearchCV (100 iterations, 3-fold CV)...")
train_start = time.time()
search.fit(X_train_vec, y_train)
train_time  = elapsed(train_start)

best_rf = search.best_estimator_
print(f"\nBest parameters: {search.best_params_}")
print(f"Training time:   {train_time:.2f}s")


# ---------------------------------------------------------------------------
# 4. Evaluation
# ---------------------------------------------------------------------------
print("\nEvaluating...")
evaluate_ml_model(
    model=best_rf,
    X_train=X_train_vec,
    y_train=y_train,
    X_test=X_test_vec,
    y_test=y_test,
    train_time=train_time,
    model_name="random_forest",
)

