# Seismic-Signal-Quality-Assessment

_Deep learning pipeline for automatic seismic waveform quality classification using spectrograms._

---

## Abstract

The rapid expansion of seismic data from global networks demands scalable, automated methods for evaluating waveform quality—an essential step in ensuring accurate seismic analyses.

We introduce a CNN-based framework for automatic seismic signal quality assessment. The system operates on spectrograms of three-component strong-motion records and is trained using data from the Engineering Strong-Motion (ESM) Database. Three custom architectures are implemented and compared:

- `VGG-Variant`
- `ResNet-Variant`
- `InceptionNet-Variant`

A systematic analysis of data augmentation techniques reveals that:

- Domain-aware augmentations such as **time stretching**, **time shifting**, and **horizontal flipping** improve model generalization.
- Non-physical augmentations like **rotation**, **Gaussian noise**, and **frequency masking** degrade performance.

Our best-performing model, the `ResNet-Variant`, achieves:

- **Accuracy**: 99.61%  
- **AUC**: 0.9999

These results emphasize the importance of domain-specific augmentation in improving CNN performance for seismic tasks.

---

## Dataset

The dataset consists of processed strong-motion recordings from the **Engineering Strong-Motion (ESM) Database**, comprising over **80,000** three-component accelerograms from:

- **ITACA (Italy)**
- **TR-NSMN (Turkey)**
- **HEAD (Greece)**

Each sample contains three orthogonal accelerograms (HNE, HNN, HNZ) and is labeled as follows:

**Labels:**

- `GoodQuality`: Signals meeting strict quality-control criteria, including:
  - Complete metadata
  - Three non-empty components
  - Sufficient pre-event and coda duration
  - High signal-to-noise ratio
  - Reasonable frequency passband
  - No overlapping events
  - Physically consistent component amplitudes
  - No spikes, clipping, or discontinuities
  - Valid displacement/acceleration spectra

- `BadQuality`: Signals that fail one or more of the above criteria due to noise, spikes, clipping, missing metadata, or artifacts.


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
├── data/
│   └── Pipeline/
│       ├── GoodQuality/
│       └── BadQuality/
├── saved_models/           # Trained model weights 
├── results/                # Fold-wise plots (confusion, ROC, accuracy)
└── README.md
