%%writefile src/config.py

# =========================
# Training Configuration
# =========================

BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 0.0005

# =========================
# Data Configuration
# =========================

INPUT_LENGTH = 9000
INPUT_SHAPE = (INPUT_LENGTH, 1)

NUM_CLASSES = 4

DATASET_PATH = "/content/training2017"
LABEL_PATH = "/content/training2017/REFERENCE.csv"

# =========================
# Model Save Path
# =========================

MODEL_SAVE_PATH = "models/ecg_cnn_model.h5"
