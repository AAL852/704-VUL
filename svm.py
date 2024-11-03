import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# Define the file path and read the CSV file
file_path = "/home/lyz/vuln/df_preprocessed.csv"
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

# Define preprocessing and tokenization
def preprocess_code(code):
    code = re.sub(r'//.*', '', code)  # remove single-line comments
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)  # remove multi-line comments
    code = re.sub(r'".*?"', '', code)  # remove string literals
    code = re.sub(r'\b[-+]?\d*\.?\d+\b', 'NUMBER', code)  # normalize numbers
    return code.strip()

# Lexex tokeniser for C programing
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

#Create a vocabulary with the frequent tokens
vocab = {token: idx+1 for idx, token in enumerate(frequent_tokens)}  # Adding 1 in the index because 0 is reserved for padding
vocab_size = len(vocab)

print("vocab_size:", vocab_size)
#from collections import defaultdict
#from operator import itemgetter
from keras.preprocessing.sequence import pad_sequences
from pygments.lexers import CLexer
from pygments import lex
from tqdm import tqdm
import re


# Function to tokenise dataset with frequent tokens/vocabulary
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


# allow you to stop model training after a set period of time
class TrainForTime(keras.callbacks.Callback):
    def __init__(self, train_time_mins=5,):
        super().__init__()

        self.train_time_mins = train_time_mins
        self.epochs = 0
        self.train_time = 0
        self.end_early = False

    def on_train_begin(self, logs=None):
        # save the start time
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
            print('training time exceeded and ending early')
            print(f'training ended on epoch {self.epochs}')
            print(f'training time = {self.train_time / 60} mins')

# # Save the best model
# from keras.callbacks import ModelCheckpoint
# checkpoint_filepath = "/content/drive/MyDrive/Colab Notebooks/best_model_{epoch:02d}-{val_loss:.2f}.h5"
# model_checkpoint = ModelCheckpoint(filepath=checkpoint_filepath,
#                                    save_best_only=True,  # Only save a model if 'val_loss' has improved.
#                                    monitor='val_loss',
#                                    mode='min',  # 'min' mode means the callback saves when 'val_loss' is minimized.
#                                    verbose=1)

import time
from sklearn.metrics import ConfusionMatrixDisplay

def track_time(start_time):
    return time.time() - start_time

# Function to evaluate the traditional ML models
def eval_model_tml(model, X_train, Y_train, X_test, Y_test):
    fig = plt.figure(figsize=[15, 6])

    ax = fig.add_subplot(1, 2, 1)
    conf = ConfusionMatrixDisplay.from_estimator(model, X_train, Y_train, normalize='true', ax=ax)
    pred = model.predict(X_train)
    conf.ax_.set_title('Training Set Performance: ' + str(sum(pred == Y_train)/len(Y_train)))

    ax = fig.add_subplot(1, 2, 2)
    conf = ConfusionMatrixDisplay.from_estimator(model, X_test, Y_test, normalize='true', ax=ax)
    pred = model.predict(X_test)
    conf.ax_.set_title('Test Set Performance: ' + str(sum(pred == Y_test)/len(Y_test)))

    plt.show()


import matplotlib.pyplot as plt

def plot_training_histories(histories):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))

    for history in histories:
        ax1.plot(history.history['loss'], label="Training Loss")
        ax1.plot(history.history['val_loss'], label="Validation Loss")
        ax2.plot(history.history['accuracy'], label="Train Accuracy")
        ax2.plot(history.history['val_accuracy'], label="Val Accuracy")

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss vs Validation Loss')
    ax1.legend()

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training Accuracy vs Validation Accuracy')
    ax2.legend()

    plt.show()


import pickle
import matplotlib.pyplot as plt

def save_history_to_file(history, filename):
    with open(filename, 'wb') as file:
        pickle.dump(history.history, file)

def load_history_from_file(filename):
    with open(filename, 'rb') as file:
        history = pickle.load(file)
    return history


def save_metrics_to_file(metrics, filename):
    with open(filename, 'wb') as file:
        pickle.dump(metrics, file)

def load_metrics_from_file(filename):
    with open(filename, 'rb') as file:
        metrics = pickle.load(file)
    return metrics
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

#SVM
import numpy as np
import time
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import RandomizedSearchCV

# Mount Google Drive to save logs
# from google.colab import drive
# drive.mount('/content/drive')

# Bag of Words:
vectorizer = CountVectorizer(max_features=vocab_size)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_val_vectorized = vectorizer.transform(X_val)
X_test_vectorized = vectorizer.transform(X_test)

# 7. Train the model using RandomizedSearchCV
param_distributions = {
    'C': [0.1, 1, 10],
    'gamma': ['scale'],
    'kernel': ['linear', 'rbf']
}


svm = SVC(random_state=42)

# The number of iterations (n_iter) is set to 100 as an example.
# You can adjust this value based on your preferences and available computational resources.
random_search = RandomizedSearchCV(estimator=svm, param_distributions=param_distributions,
                                   n_iter=10, cv=3, verbose=2, random_state=42, n_jobs=-1)

train_start = time.time()
random_search.fit(X_train_vectorized, y_train)
train_time = time.time() - train_start

best_svm = random_search.best_estimator_
best_params = best_svm.get_params()
print("Best Parameters:")
for param, value in best_params.items():
    print(f"{param}: {value}")

# 8. Evaluate the model
train_pred_start = time.time()
y_train_pred = best_svm.predict(X_train_vectorized)
train_accuracy = accuracy_score(y_train, y_train_pred)
train_pred_time = time.time() - train_pred_start

test_pred_start = time.time()
y_test_pred = best_svm.predict(X_test_vectorized)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_pred_time = time.time() - test_pred_start

print(f"Train Accuracy: {train_accuracy:.2f}")
print(f"Test Accuracy: {test_accuracy:.2f}")
print(f"Training Time: {train_time:.2f} seconds")
print(f"Inference Time (training set): {train_pred_time:.2f} seconds")
print(f"Inference Time (testing set): {test_pred_time:.2f} seconds")
print(classification_report(y_test, y_test_pred))

# Save the results to Google Drive
log_data = {
    'Best Parameters': [best_params],
    'Train Accuracy': [train_accuracy],
    'Test Accuracy': [test_accuracy],
    'Training Time (s)': [train_time],
    'Inference Time (Training Set) (s)': [train_pred_time],
    'Inference Time (Testing Set) (s)': [test_pred_time],
    'Classification Report': [classification_report(y_test, y_test_pred, output_dict=True)]
}

log_df = pd.DataFrame(log_data)
log_df.to_csv('./a_log_svm.csv', index=False)

eval_model_tml(best_svm, X_train_vectorized, y_train, X_test_vectorized, y_test)


#CNN

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.metrics import accuracy_score, classification_report

# # Define the CNN model
# class TextCNN(nn.Module):
#     def __init__(self, vocab_size, embed_dim, num_classes):
#         super(TextCNN, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, embed_dim)
#         self.conv1 = nn.Conv2d(1, 100, kernel_size=(3, embed_dim))
#         self.conv2 = nn.Conv2d(1, 100, kernel_size=(4, embed_dim))
#         self.conv3 = nn.Conv2d(1, 100, kernel_size=(5, embed_dim))
#         self.dropout = nn.Dropout(0.5)
#         self.fc = nn.Linear(300, num_classes)

#     def forward(self, x):
#         x = self.embedding(x)
#         x = x.unsqueeze(1)  # Add a channel dimension (batch_size, 1, seq_len, embed_dim)
#         x1 = torch.relu(self.conv1(x)).squeeze(3)  # (batch_size, num_filters, seq_len - filter_size + 1)
#         x1 = torch.max_pool1d(x1, x1.size(2)).squeeze(2)
#         x2 = torch.relu(self.conv2(x)).squeeze(3)
#         x2 = torch.max_pool1d(x2, x2.size(2)).squeeze(2)
#         x3 = torch.relu(self.conv3(x)).squeeze(3)
#         x3 = torch.max_pool1d(x3, x3.size(2)).squeeze(2)
#         x = torch.cat((x1, x2, x3), 1)
#         x = self.dropout(x)
#         logits = self.fc(x)
#         return logits

# # Convert text data to tensors
# vectorizer = CountVectorizer(max_features=vocab_size)
# X_train_vectorized = vectorizer.fit_transform(X_train).toarray()
# X_test_vectorized = vectorizer.transform(X_test).toarray()

# X_train_tensor = torch.tensor(X_train_vectorized, dtype=torch.long)
# y_train_tensor = torch.tensor(y_train, dtype=torch.long)
# X_test_tensor = torch.tensor(X_test_vectorized, dtype=torch.long)
# y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# # Create DataLoader
# train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
# test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# # Set up the CNN model, loss function, and optimizer
# embed_dim = 128  # Embedding dimension
# num_classes = len(set(y_train))  # Number of output classes
# model = TextCNN(vocab_size=vocab_size, embed_dim=embed_dim, num_classes=num_classes)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Training loop
# num_epochs = 10
# for epoch in range(num_epochs):
#     model.train()
#     epoch_loss = 0
#     for X_batch, y_batch in train_loader:
#         optimizer.zero_grad()
#         outputs = model(X_batch)
#         loss = criterion(outputs, y_batch)
#         loss.backward()
#         optimizer.step()
#         epoch_loss += loss.item()
#     print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_loader):.4f}')

# # Evaluate the model
# model.eval()
# with torch.no_grad():
#     y_train_pred = []
#     y_test_pred = []

#     for X_batch, _ in train_loader:
#         outputs = model(X_batch)
#         _, predicted = torch.max(outputs, 1)
#         y_train_pred.extend(predicted.tolist())

#     for X_batch, _ in test_loader:
#         outputs = model(X_batch)
#         _, predicted = torch.max(outputs, 1)
#         y_test_pred.extend(predicted.tolist())

# train_accuracy = accuracy_score(y_train, y_train_pred)
# test_accuracy = accuracy_score(y_test, y_test_pred)
# print(f"Train Accuracy: {train_accuracy:.2f}")
# print(f"Test Accuracy: {test_accuracy:.2f}")
# print(classification_report(y_test, y_test_pred))
