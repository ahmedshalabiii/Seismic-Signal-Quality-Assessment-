#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# train/__init__.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Subset, DataLoader

import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from utils.plotting import plot_loss_curves, plot_accuracy_curves
from utils.metrics import evaluate_model


def train_and_validate(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, model_name):
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies, val_roc_aucs = [], [], []

    best_val_loss = float('inf')
    best_model_path = f"./saved_models/best_{model_name}.pth"

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.cuda(), targets.float().cuda()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            predicted = (outputs > 0.5).float()
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Validation
        model.eval()
        running_loss, correct, total = 0.0, 0, 0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.cuda(), targets.float().cuda()
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, targets)

                running_loss += loss.item() * inputs.size(0)
                predicted = (outputs > 0.5).float()
                correct += (predicted == targets).sum().item()
                total += targets.size(0)

                all_preds.append(outputs.cpu())
                all_targets.append(targets.cpu())

        val_loss = running_loss / total
        val_acc = correct / total
        val_auc = evaluate_model(torch.cat(all_preds), torch.cat(all_targets))

        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        val_roc_aucs.append(val_auc)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

        print(f"Epoch {epoch+1}/{num_epochs} "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}")

    return train_losses, val_losses, train_accuracies, val_accuracies, val_roc_aucs


def test_model(model, test_loader, criterion, model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.cuda(), targets.float().cuda()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            predicted = (outputs > 0.5).float()
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())

    test_loss = running_loss / total
    test_acc = correct / total
    test_auc = evaluate_model(torch.cat(all_preds), torch.cat(all_targets))

    print(f"\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}, Test AUC: {test_auc:.4f}")


def generate_10_fold_cv_indices(labels):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    folds = [test_idx for _, test_idx in skf.split(np.zeros(len(labels)), labels)]

    cv_splits = []
    for i in range(10):
        test_idx = folds[i]
        val_idx = folds[(i + 1) % 10]
        train_idx = np.hstack([folds[j] for j in range(10) if j not in (i, (i + 1) % 10)])

        cv_splits.append({
            'train_idx': train_idx,
            'val_idx': val_idx,
            'test_idx': test_idx
        })

    return cv_splits


def run_cross_validation(dataset, model_class, model_args, model_name, device,
                         cv_splits, batch_size=128, num_epochs=50):

    all_fold_metrics = []

    for fold_idx, split in enumerate(cv_splits):
        print(f"\n=== Fold {fold_idx + 1}/10 ===")

        train_loader = DataLoader(Subset(dataset, split['train_idx']), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(Subset(dataset, split['val_idx']), batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(Subset(dataset, split['test_idx']), batch_size=batch_size, shuffle=False)

        model = model_class(*model_args).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

        train_losses, val_losses, train_acc, val_acc, val_auc = train_and_validate(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            num_epochs=num_epochs, model_name=f"{model_name}_fold{fold_idx+1}"
        )

        best_model_path = f"./saved_models/best_{model_name}_fold{fold_idx+1}.pth"
        test_model(model, test_loader, criterion, best_model_path)

        fold_metrics = {
            'train_accuracy': train_acc[-1],
            'val_accuracy': val_acc[-1],
            'val_auc': val_auc[-1]
        }
        all_fold_metrics.append(fold_metrics)

        plot_loss_curves(train_losses, val_losses)
        plot_accuracy_curves(train_acc, val_acc)

    return all_fold_metrics

