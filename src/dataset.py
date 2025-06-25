#!/usr/bin/env python

import os
import random
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch


# -------------------------- Data Augmentations -------------------------- #
class TimeStretch:
    def __init__(self, min_rate=0.8, max_rate=1.2):
        self.min_rate = min_rate
        self.max_rate = max_rate

    def __call__(self, img):
        img = np.array(img)
        rate = random.uniform(self.min_rate, self.max_rate)
        img_height, img_width = img.shape
        new_width = int(img_width * rate)
        img_resampled = np.array(Image.fromarray(img).resize((new_width, img_height), Image.BILINEAR))
        if new_width > img_width:
            start = (new_width - img_width) // 2
            img_stretched = img_resampled[:, start:start + img_width]
        else:
            padding = (img_width - new_width) // 2
            img_stretched = np.pad(img_resampled, ((0, 0), (padding, img_width - new_width - padding)), mode='constant')
        return Image.fromarray(img_stretched)


# -------------------------- Dataset Class -------------------------- #
class SeismicDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        try:
            hne_img = Image.open(sample['HNE']).convert('L')
            hnn_img = Image.open(sample['HNN']).convert('L')
            hnz_img = Image.open(sample['HNZ']).convert('L')

            if self.transform:
                hne_tensor = self.transform(hne_img)
                hnn_tensor = self.transform(hnn_img)
                hnz_tensor = self.transform(hnz_img)
            else:
                hne_tensor = transforms.ToTensor()(hne_img)
                hnn_tensor = transforms.ToTensor()(hnn_img)
                hnz_tensor = transforms.ToTensor()(hnz_img)

            stacked_tensor = torch.cat([hne_tensor, hnn_tensor, hnz_tensor], dim=0)
            label_tensor = torch.tensor(sample['label'], dtype=torch.float32).unsqueeze(0)
            return stacked_tensor, label_tensor

        except FileNotFoundError:
            return self.__getitem__((idx + 1) % len(self.samples))


# -------------------------- Sample Handling -------------------------- #
def count_and_load_complete_samples(data_path, prefix, label):
    samples, complete = {}, {}
    for file in os.listdir(data_path):
        if file.endswith(".png") and file.startswith(prefix):
            parts = file.split('_')
            if len(parts) < 3:
                continue
            direction = parts[1]
            index = parts[2].split('.')[0]
            if index not in samples:
                samples[index] = {'HNE': None, 'HNN': None, 'HNZ': None, 'label': label}
            path = os.path.join(data_path, file)
            samples[index][direction] = path
            if all(samples[index][d] for d in ['HNE', 'HNN', 'HNZ']):
                complete[index] = samples[index]
    return complete


# -------------------------- Load & Split -------------------------- #
def prepare_dataset(data_dir, max_samples_per_class=22250, balance_classes=True):
    good_dir = os.path.join(data_dir, "GoodQuality")
    bad_dir = os.path.join(data_dir, "BadQuality")

    good_samples = count_and_load_complete_samples(good_dir, "GQ", 0)
    bad_samples = count_and_load_complete_samples(bad_dir, "BQ", 1)

    if balance_classes:
        good_samples = dict(random.sample(list(good_samples.items()), min(len(good_samples), max_samples_per_class)))
        bad_samples = dict(random.sample(list(bad_samples.items()), min(len(bad_samples), max_samples_per_class)))

    all_samples = list(good_samples.values()) + list(bad_samples.values())
    random.shuffle(all_samples)

    return all_samples


# -------------------------- Transforms & Loaders -------------------------- #
def get_transforms():
    train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    TimeStretch(min_rate=0.8, max_rate=1.2),
    transforms.RandomHorizontalFlip(p=0.5),  # HZ flipping
    transforms.ToTensor(),
    ])
    val_test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    return train_transform, val_test_transform


def create_dataloaders(train, val, test, batch_size=128, num_workers=0):
    train_tf, val_test_tf = get_transforms()
    train_ds = SeismicDataset(train, transform=train_tf)
    val_ds = SeismicDataset(val, transform=val_test_tf)
    test_ds = SeismicDataset(test, transform=val_test_tf)

    return {
        'train': DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        'val': DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        'test': DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }
