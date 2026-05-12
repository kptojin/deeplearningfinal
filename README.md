
## ECG Arrhythmia Classification Using the PhysioNet CHallenge 2017 Dataset
This repository contains a deep learning implentation for automated atrial fibrillation (AF) and arrhythmia classification using the PhysioNet Computing in Cardiology Challenge 2017 ECG dataset. The project was developed and executed in Google Colab using Python and TensorFlow/Keras. 

The dataset consits of short single-lead ECG recordings collected from real patients and labeled into four rhythm categories: 
  1. N - Normal sinus rhythm
  2. A - Atrial fibrillation (AF)
  3. O - Other abnormal rhythms
  4. ~ - Noisy recordings
Each ECG recording is stored as a .mat signal file with associated reference labels. The recordings vary in length and require preprocessing techniques such as signal padding, normalization, and label encoding prior to model training.

The goal is to utilize this dataset to investigate how deep learning models can automatically classify heart rhythm abnormalities from short ECG recordings while analyzing the effects of training parameters such as batch size and epoch count on the model's performance.

How to run in Google Colab

!pip install wfdb scipy pandas numpy

wget -O training2017.zip https://physionet.org/files/challenge-2017/1.0.0/training2017.zip

!unzip training2017.zip

import os
print(len(os.listdir("training2017")))

!mkdir -p models
!mkdir -p images
!mkdir -p src

!python src/train.py

from tensorflow.keras.models import load_model
model = load_model("models/ecg_cnn_model.h5")

!python src/predict.py





The open source by authors
https://physionet.org/content/challenge-2017/1.0.0/#files-panel
