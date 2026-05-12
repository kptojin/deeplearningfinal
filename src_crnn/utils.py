%%writefile src/utils.py

import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# Normalize ECG Signals
# ==========================================

def normalize_signals(X):
    return X / np.max(np.abs(X), axis=1, keepdims=True)

# ==========================================
# Pad or Truncate ECG Signal
# ==========================================

def pad_signal(signal, max_length=9000):

    if len(signal) > max_length:
        return signal[:max_length]

    padded = np.zeros(max_length)
    padded[:len(signal)] = signal

    return padded

# ==========================================
# Plot Accuracy and Loss
# ==========================================

def plot_history(history):

    # Accuracy
    plt.figure()

    plt.plot(history.history['accuracy'],
             label='Train Accuracy')

    plt.plot(history.history['val_accuracy'],
             label='Validation Accuracy')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Epoch')
    plt.legend()
    plt.show()

    # Loss
    plt.figure()

    plt.plot(history.history['loss'],
             label='Train Loss')

    plt.plot(history.history['val_loss'],
             label='Validation Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs Epoch')
    plt.legend()
    plt.show()
