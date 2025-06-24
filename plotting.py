#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# utils/plotting.py

import matplotlib.pyplot as plt
import os

def plot_loss_curves(train_losses, val_losses, save_path=None):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Training Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='o')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(os.path.join(save_path, "loss_curve.png"))
    else:
        plt.show()
    plt.close()


def plot_accuracy_curves(train_acc, val_acc, save_path=None):
    plt.figure(figsize=(8, 5))
    plt.plot(train_acc, label='Training Accuracy', marker='o')
    plt.plot(val_acc, label='Validation Accuracy', marker='o')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(os.path.join(save_path, "accuracy_curve.png"))
    else:
        plt.show()
    plt.close()

