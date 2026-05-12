%%writefile src/model.py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D,
    MaxPooling1D,
    Dense,
    Dropout,
    BatchNormalization,
    GlobalAveragePooling1D
)

from tensorflow.keras.optimizers import Adam

import config


# ==========================================
# Build CNN Model
# ==========================================

def build_model():

    model = Sequential([

        Conv1D(
            32,
            7,
            activation='relu',
            input_shape=config.INPUT_SHAPE
        ),

        BatchNormalization(),

        MaxPooling1D(2),

        Conv1D(
            64,
            5,
            activation='relu'
        ),

        BatchNormalization(),

        MaxPooling1D(2),

        Conv1D(
            128,
            3,
            activation='relu'
        ),

        BatchNormalization(),

        MaxPooling1D(2),

        GlobalAveragePooling1D(),

        Dense(
            64,
            activation='relu'
        ),

        Dropout(0.5),

        Dense(
            config.NUM_CLASSES,
            activation='softmax'
        )
    ])

    model.compile(
        optimizer=Adam(
            learning_rate=config.LEARNING_RATE
        ),

        loss='sparse_categorical_crossentropy',

        metrics=['accuracy']
    )

    return model
