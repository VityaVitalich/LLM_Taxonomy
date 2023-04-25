import pandas as pd
import numpy as np

import os
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

import multiprocessing

from tqdm import tqdm_notebook as tqdm
import h5py
from torch.utils.data import TensorDataset

import gc

from torch.utils.data import Subset, DataLoader
from torchvision.transforms import Compose, Resize, Normalize, ToTensor, RandomCrop
from torchvision.datasets import ImageFolder


def load_dataset(name, path, batch_size=64, test_ratio=0.1, device='cpu'):

    # load the data
    # dataset = ...
    
    idx = list(range(len(dataset)))
        
    test_size = int(len(idx) * test_ratio)
    train_idx, test_idx = idx[:-test_size], idx[-test_size:]
    train_set, test_set = Subset(dataset, train_idx), Subset(dataset, test_idx)

    train_sampler = LoaderSampler(DataLoader(train_set, shuffle=True, num_workers=8, batch_size=batch_size), device)
    test_sampler = LoaderSampler(DataLoader(test_set, shuffle=True, num_workers=8, batch_size=batch_size), device)
    return train_sampler, test_sampler

class Sampler:
    def __init__(
        self, device='cpu',
    ):
        self.device = device
    
    def sample(self, size=5):
        pass
    
class LoaderSampler(Sampler):
    def __init__(self, loader, device='gpu'):
        super(LoaderSampler, self).__init__(device)
        self.loader = loader
        self.it = iter(self.loader)
        
    def sample(self, size=5):
        assert size <= self.loader.batch_size
        try:
            batch, _ = next(self.it)
        except StopIteration:
            self.it = iter(self.loader)
            return self.sample(size)
        if len(batch) < size:
            return self.sample(size)
            
        return batch[:size].to(self.device)
