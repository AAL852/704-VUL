"""
4_model_cnn.py
--------------
1D Convolutional Neural Network for C/C++ vulnerability detection.

Loads preprocessed padded sequences from 1_preprocess.py, builds and trains
a CNN with an embedding layer, and evaluates on the test set.
Best model weights are saved to outputs/best_cnn_model.keras.
"""

import os
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight

from utils import elapsed, load_splits, evaluate_dl_model, plot_training_curves

os.makedirs("outputs", exist_ok=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EMBEDDING_DIM  = 100
FILTERS        = 128
KERNEL_SIZE    = 5
DENSE_UNITS    = 64
DROPOUT_RATE   = 0.5
LEARNING_RATE  = 0.001
BATCH_SIZE     = 128
MAX_EPOCHS     = 50
PATIENCE       = 10
TRAIN_TIME_CAP = 90   # minutes — training stops early if this is exceeded


# ---------------------------------------------------------------------------
# 1. Load preprocessed splits
# ---------------------------------------------------------------------------
print("Loading preprocessed data...")
splits = load_splits("data")

X_train = splits["X_train_padded"]
X_val   = splits["X_val_padded"]
X_test  = splits["X_test_padded"]
y_train = splits["y_train"]
y_val   = splits["y_val"]
y_test  = splits["y_test"]

vocab_size = splits["vocab_size"]
max_length = splits["max_length"]


# ---------------------------------------------------------------------------
# 2. Class weights (address minority class imbalance)
# ---------------------------------------------------------------------------
classes      = np.unique(y_train)
weights      = compute_class_weight("balanced", classes=classes, y=y_train)
class_weight = dict(zip(classes, weights))
print(f"Class weights: {class_weight}")


# ---------------------------------------------------------------------------
# 3. Model definition
# ---------------------------------------------------------------------------
model = Sequential(name="cnn_vulnerability_detector")
model.add(Embedding(input_dim=vocab_size + 1,
                    output_dim=EMBEDDING_DIM,
                    input_length=max_length,
                    trainable=True))
model.add(Conv1D(filters=FILTERS, kernel_size=KERNEL_SIZE, activation="relu"))
model.add(GlobalMaxPooling1D())
model.add(Dense(DENSE_UNITS, activation="relu"))
model.add(Dropout(DROPOUT_RATE))
model.add(Dense(1, activation="sigmoid"))

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.summary()


# ---------------------------------------------------------------------------
# 4. Time-cap callback — stops training if wall time exceeds cap
# ---------------------------------------------------------------------------
class TimeCap(keras.callbacks.Callback):
    """Stop training after a fixed number of minutes."""
    def __init__(self, max_minutes: int):
        super().__init__()
        self.max_seconds = max_minutes * 60

    def on_train_begin(self, logs=None):
        self._start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        if time.time() - self._start >= self.max_seconds:
            print(f"\nTime cap reached ({self.max_seconds / 60:.0f} min). Stopping.")
            self.model.stop_training = True


# ---------------------------------------------------------------------------
# 5. Training
# ---------------------------------------------------------------------------
callbacks = [
    EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True),
    TimeCap(max_minutes=TRAIN_TIME_CAP),
]

print("\nTraining CNN...")
train_start = time.time()
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=1,
)
train_time = elapsed(train_start)
print(f"Training time: {train_time:.2f}s")


# ---------------------------------------------------------------------------
# 6. Save model and training curves
# ---------------------------------------------------------------------------
model.save("outputs/best_cnn_model.keras")
plot_training_curves(history, model_name="cnn")


# ---------------------------------------------------------------------------
# 7. Evaluation
# ---------------------------------------------------------------------------
print("\nEvaluating...")
evaluate_dl_model(
    model=model,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    train_time=train_time,
)


