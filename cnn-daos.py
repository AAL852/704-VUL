import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# Define the file path and read the CSV file
file_path = "//Users/united-al/Desktop/QUT/Year 2 Sem 2/IFN 703 & 704/704 vul assignment/Data/MSR_data_cleaned.csv"
try:
    df1 = pd.read_csv(file_path, na_filter=False)
except FileNotFoundError:
    print("The CSV file does not exist.")

# Reset index and extract features
df = df1.reset_index(drop=True)
Y = np.array(df["vul"])  # Define the target/response feature
df = df["func_before"].tolist()  # Feature: function code before fixing

# Set binary classification: 0 = non-vulnerable, 1 = vulnerable
Y[Y > 0] = 1

# Ensure 10% of vulnerable code in the training set
vulnerable_indices = np.where(Y == 1)[0]
non_vulnerable_indices = np.where(Y == 0)[0]
vuln_train_size = max(int(0.1 * len(vulnerable_indices)), 1)  # Ensure at least 10% vulnerable code
non_vuln_train_size = len(vulnerable_indices) - vuln_train_size  # Adjust remaining split

# Split vulnerable and non-vulnerable data separately
vuln_train_indices, vuln_test_indices = train_test_split(vulnerable_indices, test_size=0.2, random_state=42)
non_vuln_train_indices, non_vuln_test_indices = train_test_split(non_vulnerable_indices, test_size=0.2, random_state=42)

# Combine training and testing sets with 10% vulnerable code in the training set
train_indices = np.concatenate([vuln_train_indices[:vuln_train_size], non_vuln_train_indices[:non_vuln_train_size]])
test_indices = np.concatenate([vuln_test_indices, non_vuln_test_indices])

X_train = [df[i] for i in train_indices]
y_train = Y[train_indices]
X_test = [df[i] for i in test_indices]
y_test = Y[test_indices]

# Split a validation set from the test data
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Pseudo-labeling: Predict labels for test and validation sets
from sklearn.linear_model import LogisticRegression

# Train a simple model to generate pseudo-labels
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
pseudo_labels = clf.predict(X_test)  # Pseudo-label the test set

# Add pseudo-labeled test data to the training set
X_train += X_test
y_train = np.concatenate([y_train, pseudo_labels])

# Confirm that the training set has at least 10% vulnerable code
unique, counts = np.unique(y_train, return_counts=True)
class_distribution = dict(zip(unique, counts))
vuln_percentage = (class_distribution[1] / len(y_train)) * 100
print(f"Vulnerable code percentage in training set: {vuln_percentage:.2f}%")

# Calculate class distribution per each dataset
unique, counts = np.unique(y_train, return_counts=True)
class_counts_y_train = dict(zip(unique, counts))

unique, counts = np.unique(y_val, return_counts=True)
class_counts_y_val = dict(zip(unique, counts))

unique, counts = np.unique(y_test, return_counts=True)
class_counts_y_test = dict(zip(unique, counts))

print("class_counts_y_train:", class_counts_y_train)
print("class_counts_y_val:", class_counts_y_val)
print("class_counts_y_test:", class_counts_y_test)

from collections import defaultdict, Counter
from pygments.lexers import CLexer
from pygments import lex
from tqdm import tqdm
import re
import random
import string

# Data Augmentation Functions
def augment_code_variable_renaming(code, num_augments=1):
    augmented_codes = []
    for _ in range(num_augments):
        lexer = CLexer()
        tokens = list(lex(code, lexer))
        new_code_tokens = []
        var_mapping = {}
        for token_type, token_value in tokens:
            if token_type in Token.Name and token_value not in var_mapping:
                # Generate a new variable name
                new_var_name = ''.join(random.choices(string.ascii_letters, k=5))
                var_mapping[token_value] = new_var_name
                new_code_tokens.append((token_type, new_var_name))
            elif token_value in var_mapping:
                new_code_tokens.append((token_type, var_mapping[token_value]))
            else:
                new_code_tokens.append((token_type, token_value))
        # Reconstruct code from tokens
        augmented_code = ''.join(token_value for token_type, token_value in new_code_tokens)
        augmented_codes.append(augmented_code)
    return augmented_codes

def augment_code_insert_noop(code, num_augments=1):
    augmented_codes = []
    noop_statements = ["; ", "/* noop */ ", "int temp_var = 0; "]
    for _ in range(num_augments):
        insert_point = random.randint(0, len(code))
        noop_statement = random.choice(noop_statements)
        augmented_code = code[:insert_point] + noop_statement + code[insert_point:]
        augmented_codes.append(augmented_code)
    return augmented_codes

# Apply Data Augmentation to Training Data
augmented_X_train = []
augmented_y_train = []

for code, label in tqdm(zip(X_train, y_train), desc="Augmenting Training Data", total=len(X_train)):
    # Variable Renaming Augmentation
    augmented_codes_var = augment_code_variable_renaming(code, num_augments=2)
    # Insert No-op Statements Augmentation
    augmented_codes_noop = augment_code_insert_noop(code, num_augments=2)
    # Combine original and augmented codes
    augmented_X_train.extend([code] + augmented_codes_var + augmented_codes_noop)
    augmented_y_train.extend([label] * (1 + len(augmented_codes_var) + len(augmented_codes_noop)))

# Update training data with augmented data
X_train = augmented_X_train
y_train = np.array(augmented_y_train)

# Confirm the new size of the training set
print(f"New training set size: {len(X_train)}")

# Proceed with Tokenization and Preprocessing
# Define preprocessing and tokenization
def preprocess_code(code):
    code = re.sub(r'//.*', '', code)  # remove single-line comments
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)  # remove multi-line comments
    code = re.sub(r'".*?"', '', code)  # remove string literals
    code = re.sub(r'\b[-+]?\d*\.?\d+\b', 'NUMBER', code)  # normalize numbers
    return code.strip()

# Lexex tokenizer for C programming
def clexer_tokenize(code):
    lexer = CLexer()
    return [token[1] for token in lex(preprocess_code(code), lexer)]

# Tokenize training data
X_train_tokenized = [clexer_tokenize(code) for code in tqdm(X_train, desc="Tokenizing Train Data")]

# Calculate Token Frequencies
token_freq = Counter(token for tokens in X_train_tokenized for token in tokens)

# Print the total number of tokens
print("Total number of tokens:", len(token_freq))

import matplotlib.pyplot as plt

# Determine a frequency threshold from percentiles
tokens, frequencies = zip(*sorted(token_freq.items(), key=lambda x: x[1], reverse=True))
percentiles = [75, 90, 95, 97, 99]
percentile_values = np.percentile(frequencies, percentiles)

# Plot the token frequencies
plt.figure(figsize=(10,6))
plt.hist(frequencies, bins=100, edgecolor='black', alpha=0.7, log=True)  # log scale for better visualization
plt.title('Token Frequency Distribution')
plt.xlabel('Frequency')
plt.ylabel('Number of Tokens (log scale)')

# Plot the percentiles on the histogram
for perc, value in zip(percentiles, percentile_values):
    plt.axvline(value, color='red', linestyle='dashed', linewidth=1, label=f'{perc}th Percentile')

# Print the determined threshold frequencies and count tokens above and below each threshold
for perc, value in zip(percentiles, percentile_values):
    # Get the number of tokens above the current percentile value
    num_tokens_above_value = sum(i >= value for i in frequencies)
    # Get the number of tokens below the current percentile value
    num_tokens_below_value = sum(i < value for i in frequencies)
    print(f"{perc}th percentile frequency: {value} (Number of tokens above this value: {num_tokens_above_value}, Number of tokens below or equal to this value: {num_tokens_below_value})")

plt.legend()
plt.tight_layout()
plt.show()
import numpy as np
import matplotlib.pyplot as plt

# Compute the number of tokens above and below each percentile threshold
tokens_above = [sum(i > value for i in token_freq.values()) for value in percentile_values]
tokens_below = [sum(i <= value for i in token_freq.values()) for value in percentile_values]

# Settings for the bar plot
barWidth = 0.25
r1 = np.arange(len(tokens_above))
r2 = [x + barWidth for x in r1]

# Create the grouped bar plot
plt.bar(r1, tokens_above, color='blue', width=barWidth, edgecolor='grey', label='Tokens Above')
plt.bar(r2, tokens_below, color='red', width=barWidth, edgecolor='grey', label='Tokens Below or Equal')

# Label and style the chart
plt.xlabel('Percentiles', fontweight='bold')
plt.ylabel('Number of Tokens', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(tokens_above))], [f"{perc}%" for perc in percentiles])
plt.legend()
plt.title("Number of Tokens Above and Below Each Percentile")

# Display the plot
plt.show()
# Choose a threshold value (95th percentile)
threshold_frequency = percentile_values[2]

# Filter out tokens that are below the threshold
frequent_tokens = {token for token, freq in token_freq.items() if freq >= threshold_frequency}
print("Number of frequent tokens:", len(frequent_tokens))

# Filter the tokenized data based on the threshold
X_train_filtered = [[token for token in tokens if token in frequent_tokens] for tokens in X_train_tokenized]
print("Example of filtered data:", X_train_filtered[0])
# Use frequent tokens for filtering
frequent_tokens = set(token for token, freq in token_freq.items() if freq >= threshold_frequency)

# Create a vocabulary with the frequent tokens
vocab = {token: idx+1 for idx, token in enumerate(frequent_tokens)}  # Adding 1 in the index because 0 is reserved for padding
vocab_size = len(vocab)

print("vocab_size:", vocab_size)
# Function to tokenize dataset with frequent tokens/vocabulary
def frequent_tokens_filter(code):
    return [token for token in clexer_tokenize(code) if token in frequent_tokens]

# Tokenize using the frequent tokens
X_train_tokenized = [frequent_tokens_filter(code) for code in tqdm(X_train, desc="Tokenizing Train Data with Frequent Tokens")]
X_val_tokenized = [frequent_tokens_filter(code) for code in tqdm(X_val, desc="Tokenizing Validation Data with Frequent Tokens")]
X_test_tokenized = [frequent_tokens_filter(code) for code in tqdm(X_test, desc="Tokenizing Test Data with Frequent Tokens")]

# Function to convert tokens to integers
def tokens_to_integers(tokens):
    return [vocab[token] for token in tokens if token in vocab]

# Convert tokens to integers
X_train_seq = [tokens_to_integers(tokens) for tokens in X_train_tokenized]
X_val_seq = [tokens_to_integers(tokens) for tokens in X_val_tokenized]
X_test_seq = [tokens_to_integers(tokens) for tokens in X_test_tokenized]
import matplotlib.pyplot as plt

# Calculate sequence lengths
sequence_lengths = [len(seq) for seq in X_train_seq + X_val_seq + X_test_seq]

# Calculate percentiles
percentiles = [np.percentile(sequence_lengths, p) for p in [75, 95, 98, 99]]

# Plotting the distribution
plt.figure(figsize=(10,6))
plt.hist(sequence_lengths, bins=50, edgecolor='black', alpha=0.7)
plt.title('Sequence Length Distribution')
plt.xlabel('Length of Sequences')
plt.ylabel('Number of Sequences')

# Plotting the percentiles
for p in percentiles:
    plt.axvline(p, color='red', linestyle='dashed', linewidth=1)

# Display the plot with legend
labels = ['75th', '95th', '98th', '95th']
plt.legend(['Sequence Lengths'] + labels)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Calculate percentiles values
percentile_values = [75, 95, 98, 99, 100]
percentiles = [np.percentile(sequence_lengths, p) for p in percentile_values]

for p_val, p_result in zip(percentile_values, percentiles):
    print(f"The {p_val}th percentile is: {p_result}")
    
# Padding and truncating sequences with 98th percentile length of sequences
from keras.preprocessing.sequence import pad_sequences

max_length = int(np.percentile([len(seq) for seq in X_train_seq + X_val_seq + X_test_seq], 98))
X_train_padded = pad_sequences(X_train_seq, maxlen=max_length, padding='post', truncating='post' )
X_val_padded = pad_sequences(X_val_seq, maxlen=max_length, padding='post', truncating='post')
X_test_padded = pad_sequences(X_test_seq, maxlen=max_length, padding='post', truncating='post')

# Vector representation: set dimension embedding
embedding_dim = 100
from time import process_time
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Evaluation of Model

def plot_training(history):
    fig = plt.figure(figsize=[20, 5])
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(history.history['loss'], label="Training Loss")
    ax.plot(history.history['val_loss'], label="Validation Loss")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss vs Validation Loss')
    ax.legend()

    ax = fig.add_subplot(1, 2, 2)
    ax.plot(history.history['accuracy'], label="Train Accuracy")
    ax.plot(history.history['val_accuracy'], label="Val Accuracy")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training Accuracy vs Validation Accuracy')
    ax.legend()

    plt.show()


def eval_model(model, X_train, y_train, X_test, y_test, train_time=None):
    """
    Evaluates the model and prints/visualizes various performance metrics.

    Arguments:
    - model: Keras model that needs to be evaluated.
    - X_train, y_train: Training data and labels.
    - X_test, y_test: Testing data and labels.
    - train_time: Optional, time taken to train the model.

    Returns:
    None
    """

    # Evaluate on the test set
    test_scores = model.evaluate(X_test, y_test, verbose=2)
    print('Test loss:', test_scores[0])
    print('Test accuracy:', test_scores[1])
    print("")

    fig, axes = plt.subplots(1, 2, figsize=[20, 8])

    # Predict on the training set and calculate the inference time
    inference_start = process_time()
    train_pred = (model.predict(X_train) > 0.5).astype("int32")
    inference_end = process_time()

    # Compute and visualize confusion matrix for training data
    confusion_mtx_train = confusion_matrix(y_train, train_pred)
    sns.heatmap(confusion_mtx_train, annot=True, fmt='g', ax=axes[0])
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')
    axes[0].set_title(f'Training, F1 Score: {f1_score(y_train, train_pred):.4f}')
    print(classification_report(y_train, train_pred))

    # Predict on the test set and calculate the inference time
    pred_start = process_time()
    test_pred = (model.predict(X_test) > 0.5).astype("int32")
    pred_end = process_time()

    # Compute and visualize confusion matrix for test data
    confusion_mtx_test = confusion_matrix(y_test, test_pred)
    sns.heatmap(confusion_mtx_test, annot=True, fmt='g', ax=axes[1])
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_ylabel('True Label')
    axes[1].set_title(f'Testing, F1 Score: {f1_score(y_test, test_pred):.4f}')

    if train_time:
        print(f'Training Time: {train_time:.4f} seconds')

    print(f'Inference Time (training set): {inference_end - inference_start:.4f} seconds')
    print(f'Inference Time (testing set): {pred_end - pred_start:.4f} seconds\n')
    print(classification_report(y_test, test_pred))

    plt.tight_layout()
    plt.show()


# Allow you to stop model training after a set period of time
class TrainForTime(keras.callbacks.Callback):
    def __init__(self, train_time_mins=5,):
        super().__init__()

        self.train_time_mins = train_time_mins
        self.epochs = 0
        self.train_time = 0
        self.end_early = False

    def on_train_begin(self, logs=None):
        # Save the start time
        self.start_time = tf.timestamp()

    def on_epoch_end(self, epoch, logs=None):
        self.epochs += 1
        current_time = tf.timestamp()
        training_time = (current_time - self.start_time)
        if (training_time / 60) > self.train_time_mins:
            self.train_time = current_time - self.start_time
            self.model.stop_training = True
            self.end_early = True

    def on_train_end(self, logs=None):
        if self.end_early:
            print('Training time exceeded and ending early')
            print(f'Training ended on epoch {self.epochs}')
            print(f'Training time = {self.train_time / 60} mins')

import time
def track_time(start_time):
    return time.time() - start_time

import tensorflow as tf
from tensorflow.keras import backend as K

class Attention(tf.keras.layers.Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = tf.keras.initializers.get('glorot_uniform')

        self.W_regularizer = W_regularizer
        self.b_regularizer = b_regularizer
        self.W_constraint = W_constraint
        self.b_constraint = b_constraint

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight(shape=(input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))
        if self.bias:
            eij += self.b
        eij = K.tanh(eij)
        a = K.exp(eij)
        if mask is not None:
            a *= K.cast(mask, K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim

from keras.models import Sequential
from sklearn.metrics import accuracy_score
from keras.layers import Embedding, Conv1D, Dense, Dropout, GlobalMaxPooling1D
from keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
from sklearn.utils.class_weight import compute_class_weight
import time

# Since we've augmented the data, the class distribution might have changed
# Let's compute class weights again if needed
classes = np.unique(y_train)
weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, weights))

# Optimizer with learning rate
optimizer = Adam(learning_rate=0.001)

# Model definition
model_CNN = Sequential()
# Add an embedding layer
model_CNN.add(Embedding(input_dim=vocab_size + 1, output_dim=embedding_dim, input_length=max_length, trainable=True))
# Add a Conv1D layer
model_CNN.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
# Add a Global MaxPooling layer
model_CNN.add(GlobalMaxPooling1D())
# Add an additional Dense layer with ReLU activation
model_CNN.add(Dense(64, activation='relu'))
# Add a Dropout layer
model_CNN.add(Dropout(0.5))
# Add the output Dense layer with sigmoid activation for binary classification
model_CNN.add(Dense(1, activation='sigmoid'))
# Compile the model with optimizer, loss, and metrics
model_CNN.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

print(model_CNN.summary())
model_CNN.build(input_shape=(None, max_length))
# plot_model(model_CNN, show_shapes=True)

# Training phase
def track_time(start_time):
    """Calculate the elapsed time in seconds given a starting time"""
    return time.time() - start_time

def calculate_accuracy(y_true, y_pred_probs):
    """Calculate accuracy given true labels and predicted probabilities"""
    y_pred = (y_pred_probs > 0.5).astype("int32")
    return accuracy_score(y_true, y_pred)

# 1. Training the Model
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
train_time_mins = 90  # Early time
train_time_callback = TrainForTime(train_time_mins=train_time_mins)  # Early time
class_weights = class_weight_dict

train_start = time.time()

history_CNN = model_CNN.fit(
    X_train_padded, y_train,
    validation_data=(X_val_padded, y_val),
    epochs=10,
    batch_size=128,
    class_weight=class_weights,
    callbacks=[train_time_callback]
)

train_time = track_time(train_start)

# 2. Predictions
# Predictions on train set
train_pred_start = time.time()
train_predictions = model_CNN.predict(X_train_padded)
train_pred_time = track_time(train_pred_start)

# Predictions on test set
test_pred_start = time.time()
test_predictions = model_CNN.predict(X_test_padded)
test_pred_time = track_time(test_pred_start)

# Calculating accuracy scores
train_accuracy = calculate_accuracy(y_train, train_predictions)
test_accuracy = calculate_accuracy(y_test, test_predictions)

print(f"Train Accuracy: {train_accuracy:.2f}")
print(f"Test Accuracy: {test_accuracy:.2f}")
print(f"Training Time: {train_time:.2f} seconds")
print(f"Inference Time (training set): {train_pred_time:.2f} seconds")
print(f"Inference Time (testing set): {test_pred_time:.2f} seconds")

# Evaluating the Model
plot_training(history_CNN)
eval_model(model_CNN, X_train_padded, y_train, X_test_padded, y_test)