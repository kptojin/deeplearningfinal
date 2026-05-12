## ECG Arrhythmia Classification Using the PhysioNet CHallenge 2017 Dataset
This repository contains a deep learning implentation for automated atrial fibrillation (AF) and arrhythmia classification using the PhysioNet Computing in Cardiology Challenge 2017 ECG dataset. The project was developed and executed in Google Colab using Python and TensorFlow/Keras. 

The dataset consits of short single-lead ECG recordings collected from real patients and labeled into four rhythm categories: 
  1. N - Normal sinus rhythm
  2. A - Atrial fibrillation (AF)
  3. O - Other abnormal rhythms
  4. ~ - Noisy recordings
Each ECG recording is stored as a .mat signal file with associated reference labels. The recordings vary in length and require preprocessing techniques such as signal padding, normalization, and label encoding prior to model training.
This repository provides the following:
* ECG dataset preprocessing and loading pipeline
* Signal padding and normalization methods
* Label extraction and encoding
* 1D Convolutional Neural Network (CNN) implementation
* Training and validation workflows in Google Colab
* Accuracy vs Epoch and Loss vs Epoch visualizations

The goal is to utilize this dataset to investigate how deep learning models can automatically classify heart rhythm abnormalities from short ECG recordings while analyzing the effects of training parameters such as batch size and epoch count on the model's performance.
