import torch
import pytorch_lightning as pl
from abc import abstractmethod

class CustomDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def split_and_tokenize(self, data, format, augment):
        pass