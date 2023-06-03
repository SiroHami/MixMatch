"""
Data Loader
"""

__all__ = ['get_cifar10_set', 'get_cifar10_loaders']

import os
import math
import numpy as np
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

from utils import normalize, transpose


def label_unlabel_val_split(labels, n_labeled_per_class):
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    val_idxs = []

    for i in range(10):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        train_labeled_idxs.extend(idxs[:n_labeled_per_class])
        train_unlabeled_idxs.extend(idxs[n_labeled_per_class:-500])
        val_idxs.extend(idxs[-500:])
    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)
    np.random.shuffle(val_idxs)

    return train_labeled_idxs, train_unlabeled_idxs, val_idxs


def Kth_transform(transform, K):
    transform = transform * K
    return transform


def get_cifar10_set(root, transform_labeled, transform_unlabeled, transform_val, num_labels, K=2):

    base_dataset = CIFAR10(root, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs, val_idxs = label_unlabel_val_split(base_dataset.targets, int(num_labels/10))

    train_labeled_dataset = CIFAR10_labeled(root, train_labeled_idxs, train=True, transform=transforms.Compose(transform_labeled))

    train_unlabeled_dataset = CIFAR10_unlabeled(root, train_unlabeled_idxs, train=True, transform=transforms.Compose(Kth_transform(transform_unlabeled, K)))

    val_dataset = CIFAR10_labeled(root, val_idxs, train=True, transform=transforms.Compose(transform_val), download=True)

    test_dataset = CIFAR10_labeled(root,  train=False, transform=transforms.Compose(transform_val), download=True)

    print (f"#Labeled: {len(train_labeled_idxs)} #Unlabeled: {len(train_unlabeled_idxs)} #Val: {len(val_idxs)}")

    return train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_dataset


def get_cifar10_loaders(train_labeled_dataset,
                         train_unlabeled,
                         val_dataset,
                         test_dataset,
                         batch_size,
                         num_workers,
                         pin_memory=True,):
    
    labeled_trainloader = data.DataLoader(
        train_labeled_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True)
    
    unlabeled_trainloader = data.DataLoader(
        train_unlabeled,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True)
    
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False)
    
    test_loader = data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False)
    
    return labeled_trainloader, unlabeled_trainloader, val_loader, test_loader

class CIFAR10_labeled(torchvision.datasets.CIFAR10):

    def __init__(self, root, indexs=None, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR10_labeled, self).__init__(root, train=train,
                 transform=transform, target_transform=target_transform,
                 download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
        self.data = transpose(normalize(self.data, (0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    

class CIFAR10_unlabeled(CIFAR10_labeled):

    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR10_unlabeled, self).__init__(root, indexs, train=train,
                 transform=transform, target_transform=target_transform,
                 download=download)
        self.targets = np.array([-1 for i in range(len(self.targets))])