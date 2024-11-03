import numpy as np
import pandas as pd
import time
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from pygments.lexers import CLexer
from pygments import lex
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings

# Define the file path and read the CSV file
file_path = "/home/lyz/vuln/df_preprocessed.csv"
try:
    df1 = pd.read_csv(file_path, na_filter=False)
except FileNotFoundError:
    print("The CSV file does not exist.")

# Reset index and extract features
df = df1.reset_index(drop=True)
Y = np.array(df["vul"])  # Define the target/response feature
X = df["func_before"].tolist()  # Feature: function code before fixing

# Set binary classification: 0 = non-vulnerable, 1 = vulnerable
Y[Y > 0] = 1

print(f"Total samples: {len(Y)}")
print(f"Vulnerable samples: {np.sum(Y==1)}")
print(f"Non-vulnerable samples: {np.sum(Y==0)}")

# ------------------------------
# Step 2: Ensure 10% Vulnerable Code in Training Set
# ------------------------------

# Identify indices for vulnerable and non-vulnerable samples
vulnerable_indices = np.where(Y == 1)[0]
non_vulnerable_indices = np.where(Y == 0)[0]

# Calculate training sizes
vuln_train_size = max(int(0.1 * len(vulnerable_indices)), 1)  # At least 10% vulnerable
non_vuln_train_size = int((vuln_train_size / 0.1) * 0.9)  # Ensuring overall training proportion

# Split vulnerable data
vuln_train_indices, vuln_test_indices = train_test_split(
    vulnerable_indices, test_size=0.8, random_state=42, stratify=Y[vulnerable_indices]
)

# Split non-vulnerable data
non_vuln_train_indices, non_vuln_test_indices = train_test_split(
    non_vulnerable_indices, test_size=0.8, random_state=42, stratify=Y[non_vulnerable_indices]
)

# Combine training and testing indices
train_indices = np.concatenate([vuln_train_indices[:vuln_train_size], non_vuln_train_indices[:non_vuln_train_size]])
test_indices = np.concatenate([vuln_test_indices, non_vuln_test_indices])

# Extract training and testing data
X_train = [X[i] for i in train_indices]
y_train = Y[train_indices]
X_test = [X[i] for i in test_indices]
y_test = Y[test_indices]

print(f"Labeled training samples: {len(y_train)}")
print(f"Test samples: {len(y_test)}")

# ------------------------------
# Step 3: Split Validation Set from Training Data
# ------------------------------

# Further split the training data to create a validation set
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
)

print(f"After split - Labeled Training samples: {len(y_train)}")
print(f"Validation samples: {len(y_val)}")

# ------------------------------
# Step 4: Pseudo-Labeling (Optional)
# ------------------------------

# **Note:** Since we are using `SelfTrainingClassifier`, pseudo-labeling with LogisticRegression is not required.
# The `SelfTrainingClassifier` handles pseudo-labeling internally.
# However, if you wish to perform an initial pseudo-labeling step, you can uncomment the following code.

"""
# Train a simple Logistic Regression model to generate pseudo-labels
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train, y_train)
pseudo_labels = clf.predict(X_test)

# Add pseudo-labeled test data to the training set
X_train += X_test
y_train = np.concatenate([y_train, pseudo_labels])

# Confirm the distribution
unique, counts = np.unique(y_train, return_counts=True)
class_distribution = dict(zip(unique, counts))
vuln_percentage = (class_distribution.get(1, 0) / len(y_train)) * 100
print(f"Vulnerable code percentage in training set after pseudo-labeling: {vuln_percentage:.2f}%")
"""

# ------------------------------
# Step 5: Preprocessing and Tokenization
# ------------------------------

from pygments.lexers import CLexer
from pygments import lex

# Define preprocessing function
def preprocess_code(code):
    code = re.sub(r'//.*', '', code)  # remove single-line comments
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)  # remove multi-line comments
    code = re.sub(r'".*?"', '', code)  # remove string literals
    code = re.sub(r'\b[-+]?\d*\.?\d+\b', 'NUMBER', code)  # normalize numbers
    code = re.sub(r'\s+', ' ', code)  # replace multiple whitespace with single space
    return code.strip()

# Define tokenizer using Pygments CLexer
def clexer_tokenize(code):
    lexer = CLexer()
    tokens = [token[1] for token in lex(preprocess_code(code), lexer)]
    # Remove any empty tokens or tokens that are just whitespace
    tokens = [token for token in tokens if token.strip()]
    return tokens

# Tokenize training data
print("Tokenizing training data...")
X_train_tokenized = [clexer_tokenize(code) for code in tqdm(X_train, desc="Tokenizing Train Data")]

# Tokenize validation data
print("Tokenizing validation data...")
X_val_tokenized = [clexer_tokenize(code) for code in tqdm(X_val, desc="Tokenizing Validation Data")]

# Tokenize test data
print("Tokenizing test data...")
X_test_tokenized = [clexer_tokenize(code) for code in tqdm(X_test, desc="Tokenizing Test Data")]

# ------------------------------
# Step 6: Token Frequency Analysis
# ------------------------------

# Calculate token frequencies from training data
token_freq = Counter(token for tokens in X_train_tokenized for token in tokens)

# Print the total number of unique tokens
print(f"Total unique tokens in training data: {len(token_freq)}")

# Plot Token Frequency Distribution
tokens, frequencies = zip(*sorted(token_freq.items(), key=lambda x: x[1], reverse=True))
percentiles = [75, 90, 95, 97, 99]
percentile_values = np.percentile(frequencies, percentiles)

plt.figure(figsize=(10,6))
plt.hist(frequencies, bins=100, edgecolor='black', alpha=0.7, log=True)
plt.title('Token Frequency Distribution')
plt.xlabel('Frequency')
plt.ylabel('Number of Tokens (log scale)')

# Plot percentile lines
for perc, value in zip(percentiles, percentile_values):
    plt.axvline(value, color='red', linestyle='dashed', linewidth=1, label=f'{perc}th Percentile')

plt.legend()
plt.tight_layout()
plt.show()

# Print percentile thresholds
for perc, value in zip(percentiles, percentile_values):
    num_tokens_above = sum(f >= value for f in frequencies)
    num_tokens_below = sum(f < value for f in frequencies)
    print(f"{perc}th percentile frequency: {value} (Above: {num_tokens_above}, Below: {num_tokens_below})")

# Compute tokens above threshold (e.g., 95th percentile)
threshold_frequency = percentile_values[2]  # 95th percentile
frequent_tokens = {token for token, freq in token_freq.items() if freq >= threshold_frequency}

print(f"Number of frequent tokens (>= {threshold_frequency} occurrences): {len(frequent_tokens)}")

# ------------------------------
# Step 7: Filter Tokens Based on Frequency
# ------------------------------

# Function to filter tokens
def filter_tokens(token_list, frequent_tokens):
    return [token for token in token_list if token in frequent_tokens]

# Filter training, validation, and test token lists
print("Filtering tokens in training data...")
X_train_filtered = [filter_tokens(tokens, frequent_tokens) for tokens in tqdm(X_train_tokenized, desc="Filtering Train Tokens")]

print("Filtering tokens in validation data...")
X_val_filtered = [filter_tokens(tokens, frequent_tokens) for tokens in tqdm(X_val_tokenized, desc="Filtering Validation Tokens")]

print("Filtering tokens in test data...")
X_test_filtered = [filter_tokens(tokens, frequent_tokens) for tokens in tqdm(X_test_tokenized, desc="Filtering Test Tokens")]

# ------------------------------
# Step 8: Create Vocabulary
# ------------------------------

# Create a vocabulary dictionary
vocab = {token: idx+1 for idx, token in enumerate(frequent_tokens)}  # Start indexing at 1
vocab_size = len(vocab)
print(f"Vocabulary size: {vocab_size}")

# ------------------------------
# Step 9: Vectorization
# ------------------------------

# Initialize CountVectorizer with the created vocabulary
vectorizer = CountVectorizer(vocabulary=vocab, tokenizer=lambda x: x, lowercase=False)

# Fit and transform the training data
print("Vectorizing training data...")
X_train_vectorized = vectorizer.fit_transform(X_train_filtered)

# Transform validation and test data
print("Vectorizing validation data...")
X_val_vectorized = vectorizer.transform(X_val_filtered)

print("Vectorizing test data...")
X_test_vectorized = vectorizer.transform(X_test_filtered)

print(f"Vectorized training data shape: {X_train_vectorized.shape}")
print(f"Vectorized validation data shape: {X_val_vectorized.shape}")
print(f"Vectorized test data shape: {X_test_vectorized.shape}")

# ------------------------------
# Step 10: Prepare Semi-Supervised Labels
# ------------------------------

# Combine training and validation data for semi-supervised learning
X_semi = np.concatenate((X_train_vectorized.toarray(), X_val_vectorized.toarray()))
y_semi = np.concatenate((y_train, y_val))

# In `SelfTrainingClassifier`, unlabeled data should have labels set to -1
# Here, we'll assume that only the original training data is labeled, and the validation data is unlabeled
# Adjust as needed based on your specific requirements

# For demonstration, let's treat the validation data as unlabeled
y_semi[-len(y_val):] = -1

print(f"Total training samples (labeled + unlabeled): {X_semi.shape[0]}")

# ------------------------------
# Step 11: Set Up the Semi-Supervised Model
# ------------------------------

# Initialize the base Random Forest classifier
base_rf = RandomForestClassifier(random_state=42)

# Initialize the Self-Training classifier
self_training_clf = SelfTrainingClassifier(base_estimator=base_rf, threshold=0.8, verbose=1)

# ------------------------------
# Step 12: Hyperparameter Tuning with RandomizedSearchCV
# ------------------------------

# Define parameter distributions for RandomizedSearchCV
param_distributions = {
    'base_estimator__n_estimators': [100, 200, 300],
    'base_estimator__max_depth': [None, 10, 20, 30],
    'base_estimator__min_samples_split': [2, 5, 10],
    'base_estimator__class_weight': [None, 'balanced'],
    'threshold': [0.6, 0.7, 0.8, 0.9],  # SelfTrainingClassifier parameter
}

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=self_training_clf,
    param_distributions=param_distributions,
    n_iter=20,  # Number of parameter settings sampled
    cv=3,  # 3-fold cross-validation
    verbose=2,
    random_state=42,
    n_jobs=-1
)

# ------------------------------
# Step 13: Define Helper Functions
# ------------------------------

def track_time(start_time):
    return time.time() - start_time

def plot_confusion_matrix_custom(y_true, y_pred, title='Confusion Matrix'):
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.title(title)
    plt.show()

# ------------------------------
# Step 14: Train the Model
# ------------------------------

print("Starting training...")
train_start = time.time()
random_search.fit(X_semi, y_semi)
train_time = track_time(train_start)
print(f"Training completed in {train_time:.2f} seconds")

# ------------------------------
# Step 15: Retrieve the Best Estimator and Parameters
# ------------------------------

best_estimator = random_search.best_estimator_
best_params = random_search.best_params_

print("\nBest Parameters:")
for param, value in best_params.items():
    print(f"{param}: {value}")

# ------------------------------
# Step 16: Evaluate the Model
# ------------------------------

# Function to evaluate the model
def evaluate_model(model, X, y, dataset_name='Dataset'):
    pred_start = time.time()
    y_pred = model.predict(X)
    pred_time = track_time(pred_start)
    accuracy = accuracy_score(y, y_pred)
    print(f"{dataset_name} Accuracy: {accuracy:.4f}")
    print(f"Inference Time ({dataset_name}): {pred_time:.2f} seconds")
    print(f"Classification Report ({dataset_name}):")
    print(classification_report(y, y_pred))
    plot_confusion_matrix_custom(y, y_pred, title=f'Confusion Matrix - {dataset_name}')

# Evaluate on the original labeled training set
print("\nEvaluating on the labeled training set...")
evaluate_model(best_estimator, X_train_vectorized, y_train, dataset_name='Training Set')

# Evaluate on the test set
print("\nEvaluating on the test set...")
evaluate_model(best_estimator, X_test_vectorized, y_test, dataset_name='Test Set')

