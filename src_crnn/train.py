%%writefile src/train.py

import os
import numpy as np
import pandas as pd

from scipy.io import loadmat

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.callbacks import EarlyStopping

import config
from utils import normalize_signals, pad_signal, plot_history
from model import build_model


# ==========================================
# Load Labels
# ==========================================

labels_df = pd.read_csv(
    config.LABEL_PATH,
    header=None,
    names=["record", "label"]
)

# ==========================================
# Load ECG Signals
# ==========================================

X = []
Y = []

for _, row in labels_df.iterrows():

    record = row["record"]
    label = row["label"]

    mat_path = os.path.join(
        config.DATASET_PATH,
        record + ".mat"
    )

    signal = loadmat(mat_path)["val"][0]

    signal = pad_signal(
        signal,
        config.INPUT_LENGTH
    )

    X.append(signal)
    Y.append(label)

# Convert to numpy arrays
X = np.array(X)
Y = np.array(Y)

# ==========================================
# Normalize Signals
# ==========================================

X = normalize_signals(X)

# ==========================================
# Encode Labels
# ==========================================

encoder = LabelEncoder()

Y = encoder.fit_transform(Y)

# ==========================================
# Train/Test Split
# ==========================================

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# Add Channel Dimension
# ==========================================

X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# ==========================================
# Build Model
# ==========================================

model = build_model()

# ==========================================
# Early Stopping
# ==========================================

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True
)

# ==========================================
# Train Model
# ==========================================

history = model.fit(
    X_train,
    Y_train,

    epochs=config.EPOCHS,

    batch_size=config.BATCH_SIZE,

    validation_data=(X_test, Y_test),

    callbacks=[early_stop]
)

# ==========================================
# Save Model
# ==========================================

os.makedirs("models", exist_ok=True)

model.save(config.MODEL_SAVE_PATH)

print(f"\nModel saved to: {config.MODEL_SAVE_PATH}")

# ==========================================
# Plot Results
# ==========================================

plot_history(history)
