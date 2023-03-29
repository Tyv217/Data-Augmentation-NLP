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
        self.train_dataset = self.extract_samples_per_class(train_samples_per_class)
        self.use_train_valid = use_train_valid

    def extract_samples_per_class(self, train_samples_per_class):
        num_classes = len(self.id2label)
        valid_samples_per_class = 0
        if self.use_train_valid:
            TRAIN_SPLIT = 0.8
            temp = int(train_samples_per_class * 0.8)
            valid_samples_per_class = train_samples_per_class - temp
            train_samples_per_class = temp
        
        samples = []

        for k in self.id2label.keys():
            pass

    def split_and_tokenize(self, data, augment = False):
        input_lines, labels = self.format_data(data)
        if augment and self.augmentors is not None:
            for augmentor in self.augmentors:
                input_lines, _, labels = augmentor.augment_dataset(input_lines, None, labels)
        input_encoding = self.tokenizer.batch_encode_plus(
            input_lines,
            add_special_tokens = True,
            max_length = 400,
            padding = "max_length",
            truncation = True,
            return_attention_mask = True,
            return_tensors = "pt",
        )
        input_ids, attention_masks = input_encoding.input_ids, input_encoding.attention_mask

        data_seq = []
        for input_id, attention_mask, label in zip(input_ids, attention_masks, labels):
            data_seq.append({"input_id": input_id, "attention_mask": attention_mask, "label": torch.tensor(label, dtype = torch.float)})
        return data_seq

    def setup(self, stage: str):
        pass

    def train_dataloader(self):
        return DataLoader(self.split_and_tokenize(self.train_dataset, augment = True), batch_size=self.batch_size, shuffle = True)

    def val_dataloader(self):
        return DataLoader(self.split_and_tokenize(self.valid_dataset), batch_size=self.batch_size)

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