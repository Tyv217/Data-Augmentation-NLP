import torch
import pytorch_lightning as pl
from abc import abstractmethod

class CustomDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def split_and_tokenize(self, data, format, augment):
        pass

    @property
    @abstractmethod
    def train_dataset(self):
        pass

    @property
    @abstractmethod
    def valid_dataset(self):
        pass

    @property
    @abstractmethod
    def test_dataset(self):
        pass