# Seismic-Signal-Quality-Assessment

This repository contains a **Convolutional Neural Network (CNN)** framework for fully automatic classification of seismic signal quality. It is based on our study using the **Engineering Strong-Motion (ESM) Database** and evaluates multiple custom CNN architectures on spectrograms of seismic waveforms.

---

## Abstract

The rapid expansion of seismic data from global networks demands reliable, scalable, and automated methods for evaluating signal quality—an essential step in ensuring accurate seismic analyses.

This project presents a CNN-based framework for automatic seismic signal quality assessment. We explore three custom architectures:
- `VGG-Variant`  
- `ResNet-Variant`  
- `InceptionNet-Variant`  

Through a systematic study of data augmentation techniques, we find that:
- **Time stretching**, **time shifting**, and **horizontal flipping** improve model generalization.
- Techniques like **rotation**, **Gaussian noise**, and **frequency masking** degrade performance.

Our best model, the `ResNet-Variant`, achieves:
- **Accuracy**: 99.61%  
- **AUC**: 0.9999  

These findings highlight the critical role of **domain-aware data augmentation** in CNN-based seismic classification.

---

## Dataset

We use processed strong-motion data from the **Engineering Strong-Motion (ESM) Database**, which contains over **80,000 three-component waveforms** from:

* **ITACA (Italy)**
* **TR-NSMN (Turkey)**
* **HEAD (Greece)**

Each data sample includes three orthogonal accelerograms (north-south, east-west, vertical) and is labeled as:

* **`GoodQuality`**:
  Signals that meet strict quality-control criteria, including:

  * Complete metadata
  * All three channels present and non-empty
  * Sufficient pre-event and coda durations
  * High signal-to-noise ratio
  * Reasonable frequency passband
  * Only one seismic event per record
  * Consistent peak values across components
  * Free of spikes or discontinuities
  * Physically plausible acceleration and displacement spectra

* **`BadQuality`**:
  Signals that fail to meet one or more of the above criteria due to noise contamination, missing metadata, spurious spikes, clipping, or recording artifacts.

To access the dataset:

* Visit the [ESM Database](https://esm.mi.ingv.it/) 
* Organize your data into:

  ```plaintext
  ./data/Pipeline/
  ├── GoodQuality/
  └── BadQuality/
  ```

---

## Project Structure

```plaintext
Seismic-Signal-Quality-Assessment/
├── src/
│   ├── main.py             # Runs 10-fold cross-validation on all models
│   ├── dataset.py          # Loads spectrogram triplets, applies augmentations
│   ├── models.py           # VGG, ResNet, and Inception custom architectures
│   ├── train_eval.py       # Cross-validation logic, training, evaluation
│   ├── metrics.py          # Test metrics: accuracy, AUC, confusion, ROC
│   ├── plotting.py         # Accuracy/loss visualization for each fold
├── data/
│   └── Pipeline/
│       ├── GoodQuality/
│       └── BadQuality/
├── saved_models/           # Trained model weights (auto-saved)
├── results/                # Fold-wise plots (confusion, ROC, accuracy)
└── README.md
