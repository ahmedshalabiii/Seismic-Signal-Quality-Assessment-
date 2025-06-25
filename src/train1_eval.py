#!/usr/bin/env python

import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from torch.utils.data import Subset, DataLoader

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from utils.metrics import test_model  



# -------------------------- Plotting -------------------------- #
def plot_loss_curves(train_losses, val_losses):
    plt.figure()
    plt.plot(train_losses, label="Train Loss", color="blue")
    plt.plot(val_losses, label="Validation Loss", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_accuracy_curves(train_acc, val_acc):
    plt.figure()
    plt.plot(train_acc, label="Train Accuracy", color="green")
    plt.plot(val_acc, label="Validation Accuracy", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training & Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.close()


# -------------------------- Training & Validation -------------------------- #

def train_and_validate(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, model_name, device, patience=10):
    start_time = time.time()
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies, val_roc_aucs = [], [], []

    best_val_loss = float('inf')
    best_model_state = None
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.float().to(device)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, targets)
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

        # ---- Validation ---- #
        model.eval()
        running_loss, correct, total = 0.0, 0, 0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.float().to(device)
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

        print(f"Epoch {epoch+1}/{num_epochs} "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}")

        # ---- Early Stopping Check ---- #
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"⏹️ Early stopping triggered after {patience} epochs without improvement.")
                break

    # ---- Save and Load Best Model ---- #
    if best_model_state is not None:
        model_path = f"./saved_models/best_{model_name}.pth"
        torch.save(best_model_state, model_path)
        model.load_state_dict(best_model_state)

    return train_losses, val_losses, train_accuracies, val_accuracies, val_roc_aucs

# -------------------------- Cross-Validation -------------------------- #
def run_cross_validation(dataset, model_class, model_args, model_name, device,
                         cv_splits, batch_size=128, num_epochs=50):

    all_fold_metrics = []

    os.makedirs("results", exist_ok=True)

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
        test_metrics = test_model(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            model_path=best_model_path,
            device=device,
            save_path=f"./results/{model_name}_fold{fold_idx+1}"
        )

        fold_metrics = {
            'train_accuracy': train_acc[-1],
            'val_accuracy': val_acc[-1],
            'val_auc': val_auc[-1],
            'test_loss': test_metrics[0],
            'test_accuracy': test_metrics[1],
            'precision': test_metrics[2],
            'recall': test_metrics[3],
            'f1_score': test_metrics[4],
            'test_auc': test_metrics[5]
        }

        all_fold_metrics.append(fold_metrics)
        df = pd.DataFrame(all_fold_metrics)
        df.to_csv(f"./results/{model_name}_cv_results.csv", index=False)

        plot_loss_curves(train_losses, val_losses)
        plot_accuracy_curves(train_acc, val_acc)

    return all_fold_metrics


# -------------------------- K-Fold Split Generator -------------------------- #
def generate_10_fold_cv_indices(labels):
    from sklearn.model_selection import StratifiedKFold
    import numpy as np

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    folds = [test_idx for _, test_idx in skf.split(np.zeros(len(labels)), labels)]

    cv_splits = []
    for i in range(10):
        test_idx_1 = folds[i]
        test_idx_2 = folds[(i + 1) % 10]
        val_idx_1 = folds[(i + 2) % 10]
        val_idx_2 = folds[(i + 3) % 10]

        test_idx = np.hstack([test_idx_1, test_idx_2])
        val_idx = np.hstack([val_idx_1, val_idx_2])
        train_idx = np.hstack([
            folds[j] for j in range(10) if j not in (i, (i + 1) % 10, (i + 2) % 10, (i + 3) % 10)
        ])

        cv_splits.append({
            'train_idx': train_idx,
            'val_idx': val_idx,
            'test_idx': test_idx
        })

    return cv_splits
