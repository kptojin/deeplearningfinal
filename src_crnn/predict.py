%%writefile src/predict.py

import numpy as np
from scipy.io import loadmat

from tensorflow.keras.models import load_model

import config
from utils import pad_signal, normalize_signals


# ==========================================
# Label Mapping
# ==========================================

LABELS = {
    0: "Atrial Fibrillation (A)",
    1: "Normal Rhythm (N)",
    2: "Other Rhythm (O)",
    3: "Noise (~)"
}


# ==========================================
# Load Trained Model
# ==========================================

model = load_model(config.MODEL_SAVE_PATH)

print("Model loaded successfully.")


# ==========================================
# Predict ECG Function
# ==========================================

def predict_ecg(mat_file_path):

    # Load ECG signal
    signal = loadmat(mat_file_path)["val"][0]

    # Pad/truncate
    signal = pad_signal(
        signal,
        config.INPUT_LENGTH
    )

    # Convert to array
    signal = np.array([signal])

    # Normalize
    signal = normalize_signals(signal)

    # Add channel dimension
    signal = signal[..., np.newaxis]

    # Predict
    prediction = model.predict(signal)

    predicted_class = np.argmax(prediction)

    confidence = np.max(prediction)

    return LABELS[predicted_class], confidence


# ==========================================
# Example Prediction
# ==========================================

if __name__ == "__main__":

    sample_file = "/content/training2017/A00001.mat"

    label, confidence = predict_ecg(sample_file)

    print("\nPrediction:")
    print("Class:", label)
    print("Confidence:", confidence)

!python src/predict.py
2026-05-12 13:15:25.170765: E external/local_xla/xla/stream_executor/cuda/cuda_platform.cc:51] failed call to cuInit: INTERNAL: CUDA error: Failed call to cuInit: UNKNOWN ERROR (303)
WARNING:absl:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
Model loaded successfully.
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 195ms/step

Prediction:
Class: Normal Rhythm (N)
Confidence: 0.6626125
