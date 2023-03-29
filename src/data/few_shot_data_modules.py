import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import DistilBertTokenizer
from torchtext.datasets import AG_NEWS as agnews
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
import numpy as np
import random


class FewShotTextClassifyModule(pl.LightningDataModule):
    def __init__(self, data_module: pl.LightningDataModule, train_samples_per_class = 1, use_train_valid = False):
        super().__init__()
        self.batch_size = data_module.batch_size
        self.tokenizer = data_module.tokenizer
        self.id2label = data_module.id2label
        self.label2id = data_module.label2id
        self.augmentors = data_module.augmentors
        self.train_dataset = data_module.train_dataset
        self.valid_dataset = data_module.valid_dataset
        self.test_dataset = data_module.test_dataset
        self.format_data = data_module.format_data
        self.split_and_tokenizer = data_module.split_and_tokenize

        self.use_train_valid = use_train_valid
        
        train_dataset, valid_dataset = self.extract_samples_per_class(train_samples_per_class)
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset

    def extract_samples_per_class(self, samples_per_class):
        train_input_lines = []
        train_labels = []
        valid_input_lines = []
        valid_labels = []

        input_lines, labels = self.format_data(self.train_dataset)
        input_lines = np.array(input_lines)
        labels = np.array(labels)

        for label in self.id2label.keys():
            label = np.identity(len(self.id2label))[label]
            matching_samples = input_lines[labels == label]
            if(len(matching_samples) < samples_per_class):
                raise Exception("Insufficient samples per class, expected", samples_per_class, "samples per class but class", str(label), "only has", len(matching_samples), "samples")
            np.random.shuffle(matching_samples)
            matching_samples = matching_samples[:samples_per_class]
            if self.use_train_valid:
                TRAIN_SPLIT = 0.8
                train_samples_per_class = int(samples_per_class * TRAIN_SPLIT)

                train_matching_samples = matching_samples[:train_samples_per_class]
                valid_matching_samples = matching_samples[train_samples_per_class:]

                for sample in train_matching_samples:
                    train_input_lines.append(sample)
                    train_labels.append(label)

                for sample in valid_matching_samples:
                    valid_input_lines.append(sample)
                    valid_labels.append(label)

            else:
                for sample in matching_samples:
                    train_input_lines.append(sample)
                    train_labels.append(label)
                    valid_input_lines.append(sample)
                    valid_labels.append(label)

        train_dataset = (train_input_lines, train_labels)
        valid_dataset = (valid_input_lines, valid_labels)

        return train_dataset, valid_dataset

    def setup(self, stage: str):
        pass

    def train_dataloader(self):
        return DataLoader(self.split_and_tokenize(self.train_dataset, format = False, augment = True), batch_size=self.batch_size, shuffle = True)

    def val_dataloader(self):
        return DataLoader(self.split_and_tokenize(self.valid_dataset, format = False), batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.split_and_tokenize(self.test_dataset), batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.split_and_tokenize(self.test_dataset), batch_size=self.batch_size)

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        pass

    def get_dataset_text(self):
        text, _ = self.format_data(self.train_dataset)
        return text