"""
3_model_svm.py
--------------
Support Vector Machine classifier for C/C++ vulnerability detection.

Loads preprocessed data from 1_preprocess.py, vectorises using Bag-of-Words,
tunes hyperparameters via RandomizedSearchCV, and evaluates on the test set.
Results and plots are saved to outputs/.
"""

import os
import pickle
import time
import numpy as np
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import RandomizedSearchCV

from utils import elapsed, load_splits, evaluate_ml_model

os.makedirs("outputs", exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load preprocessed splits
# ---------------------------------------------------------------------------
print("Loading preprocessed data...")
splits = load_splits("data")

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
vectorizer  = CountVectorizer(max_features=vocab_size)
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec   = vectorizer.transform(X_val)
X_test_vec  = vectorizer.transform(X_test)


# ---------------------------------------------------------------------------
# 3. Hyperparameter search
# ---------------------------------------------------------------------------
param_distributions = {
    "C":      [0.1, 1, 10],
    "gamma":  ["scale"],
    "kernel": ["linear", "rbf"],
}

svm = SVC(random_state=42)

search = RandomizedSearchCV(
    estimator=svm,
    param_distributions=param_distributions,
    n_iter=10,
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1,
)

print("\nRunning RandomizedSearchCV (10 iterations, 3-fold CV)...")
train_start = time.time()
search.fit(X_train_vec, y_train)
train_time  = elapsed(train_start)

best_svm = search.best_estimator_
print(f"\nBest parameters: {search.best_params_}")
print(f"Training time:   {train_time:.2f}s")


# ---------------------------------------------------------------------------
# 4. Evaluation
# ---------------------------------------------------------------------------
print("\nEvaluating...")
evaluate_ml_model(
    model=best_svm,
    X_train=X_train_vec,
    y_train=y_train,
    X_test=X_test_vec,
    y_test=y_test,
    train_time=train_time,
    model_name="svm",
)

# train_accuracy = accuracy_score(y_train, y_train_pred)
# test_accuracy = accuracy_score(y_test, y_test_pred)
# print(f"Train Accuracy: {train_accuracy:.2f}")
# print(f"Test Accuracy: {test_accuracy:.2f}")
# print(classification_report(y_test, y_test_pred))
