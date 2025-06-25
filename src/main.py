#!/usr/bin/env python

import os
import torch
from models import VGG_like_model, ResNet_like_model, InceptionNet_like_model, ResidualBlock
from train_eval import run_cross_validation, generate_10_fold_cv_indices
from dataset import prepare_dataset, SeismicDataset

# ---------------- CONFIGURATION ---------------- #
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log_dir = './saved_models'
os.makedirs(log_dir, exist_ok=True)

# ---------------- LOAD & PREPARE DATA ---------------- #
data_dir = './data/Pipeline'
samples = prepare_dataset(data_dir)
labels = [sample['label'] for sample in samples]
full_dataset = SeismicDataset(samples)

# ---------------- GENERATE 10-FOLD SPLITS ---------------- #
cv_splits = generate_10_fold_cv_indices(labels)

# ---------------- RUN CROSS-VALIDATION FOR MODELS ---------------- #
# VGG-like
# VGG-like
vgg_results = run_cross_validation(
    dataset=full_dataset,
    model_class=VGG_like_model,
    model_args=(),
    model_name="VGG_like_model",
    device=device,
    cv_splits=cv_splits,
    batch_size=128,
    num_epochs=50,
    patience=10
)

# ResNet-like
resnet_results = run_cross_validation(
    dataset=full_dataset,
    model_class=ResNet_like_model,
    model_args=(ResidualBlock, [1, 1, 1, 1]),
    model_name="ResNet_like_model",
    device=device,
    cv_splits=cv_splits,
    batch_size=128,
    num_epochs=50,
    patience=10
)

# Inception-like
inception_results = run_cross_validation(
    dataset=full_dataset,
    model_class=InceptionNet_like_model,
    model_args=(),
    model_name="InceptionNet_like_model",
    device=device,
    cv_splits=cv_splits,
    batch_size=128,
    num_epochs=50,
    patience=10
)


print("\n✅ Cross-validation completed for all models.")
