## ECG classification using MIT-BIH dataset
This repo is an implementation of https://www.nature.com/articles/s41591-018-0268-3 and https://arxiv.org/abs/1707.01836 and focus on training using a MIT-BIH dataset. If you want to train using CINC or open irhythm data, see the open source which the authors of the original research paper have coded at https://github.com/awni/ecg

Introduction to MIT-BIH dataset at physionet : https://physionet.org/physiobank/database/mitdb/

How to run on Google Colab:

!nvidia-smi !python --version

from google.colab import drive drive.mount('/content/drive')

%cd /content/drive/MyDrive

!pwd

!ls

!git clone https://github.com/physhik/ecg-mit-bih.git

%cd /content/drive/MyDrive/ecg-mit-bih

!pwd

!ls

!ls src

!pip install --upgrade pip

!pip install -r requirements.txt

%cd /content/drive/MyDrive/ecg-mit-bih

!pip install --upgrade wfdb==4.3.1 pandas==3.0.2

!python -u src/data.py --downloading True

!find dataset -maxdepth 2 -type f

!ls -lh dataset

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

%cd /content/drive/MyDrive/ecg-mit-bih

!pwd

!ls

!python -u src/train.py

import matplotlib.pyplot as plt

import numpy as np

The original research papers

https://www.nature.com/articles/s41591-018-0268-3

https://arxiv.org/abs/1707.01836

The open source by authors

https://github.com/awni/ecg

also noticable https://github.com/fernandoandreotti/cinc-challenge2017/tree/master/deeplearn-approach




## ECG Arrhythmia Classification Using the PhysioNet CHallenge 2017 Dataset
This repository contains a deep learning implentation for automated atrial fibrillation (AF) and arrhythmia classification using the PhysioNet Computing in Cardiology Challenge 2017 ECG dataset. The project was developed and executed in Google Colab using Python and TensorFlow/Keras. 

The dataset consits of short single-lead ECG recordings collected from real patients and labeled into four rhythm categories: 
  1. N - Normal sinus rhythm
  2. A - Atrial fibrillation (AF)
  3. O - Other abnormal rhythms
  4. ~ - Noisy recordings
Each ECG recording is stored as a .mat signal file with associated reference labels. The recordings vary in length and require preprocessing techniques such as signal padding, normalization, and label encoding prior to model training.

The goal is to utilize this dataset to investigate how deep learning models can automatically classify heart rhythm abnormalities from short ECG recordings while analyzing the effects of training parameters such as batch size and epoch count on the model's performance.

The open source by authors
https://physionet.org/content/challenge-2017/1.0.0/#files-panel
