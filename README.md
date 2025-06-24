# Seismic-Signal-Quality-Assessment-

This repository contains a **Convolutional Neural Network (CNN)** framework for fully automatic classification of seismic signal quality. We use data from the **Engineering Strong-Motion (ESM) Database** to train and evaluate custom deep learning models on spectrogram representations of seismic waveforms.



## Abstract

The rapid expansion of seismic data from global networks demands reliable, scalable, and automated methods for evaluating signal quality—an essential step in ensuring accurate seismic analyses. In this work, we present a **CNN framework** designed for automatic seismic signal quality assessment, utilizing data from the **Engineering Strong-Motion (ESM) Database**.  

We explore three custom architectures:  
- `VGG-Variant`  
- `ResNet-Variant`  
- `InceptionNet-Variant`  

Data augmentation is critical. We identify **time stretching**, **time shifting**, and **horizontal flipping** as effective, while techniques like rotation, Gaussian noise, and frequency masking degrade performance.  

Our best-performing model, the `ResNet-Variant`, achieves:  
-  **Accuracy**: 99.61%  
-  **AUC**: 0.9999  

These results show that **domain-aware augmentations** can significantly improve CNN-based seismic classification, automating high-fidelity signal assessment and reducing reliance on manual inspection in seismology.

---

## Dataset

We use processed strong-motion records from the **ESM Database**, which compiles over **80,000 three-component waveforms** from earthquakes in the Euro-Mediterranean region. Major sources include:

- ITACA (Italy)  
- TR-NSMN (Turkey)  
- HEAD (Greece)

Each data sample consists of 3 accelerograms (N-S, E-W, vertical), processed and labeled as:
- `GoodQuality`: meets strict signal-to-noise and metadata criteria.  
- `BadQuality`: contains artifacts or fails quality control.


## Project Structure

Seismic-Signal-Quality-Assessment/
├── src/
│   ├── main.py             # Runs 10-fold cross-validation on all models
│   ├── dataset.py          # Loads spectrogram triplets, applies augmentations
│   ├── models.py           # VGG, ResNet, and Inception custom architectures
│   ├── train_eval.py       # Cross-validation logic, training, evaluation
│   ├── metrics.py          # Evaluation: accuracy, AUC, confusion matrix, ROC
│   ├── plotting.py         # Accuracy/loss visualization for each fold
├── data/
│   └── Pipeline/
│       ├── GoodQuality/
│       └── BadQuality/
├── saved_models/           # Trained model weights per fold
├── results/                # Saved plots (confusion, ROC, accuracy)
└── README.md
