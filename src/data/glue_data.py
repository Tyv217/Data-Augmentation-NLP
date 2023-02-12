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


class GlueDataModule(pl.LightningDataModule):
    def __init__(self, dataset_percentage, augmentors = [], glue_task = "cola", batch_size: int = 32):
        super().__init__()
        self.batch_size = batch_size
        self.glue_task = glue_task
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)
        self.augmentors = augmentors
        self.dataset_percentage = dataset_percentage
        self.id2label =  {0: "unacceptable", 1: "acceptable"}
        self.label2id = {"unacceptable": 0, "acceptable": 1}


    def format_data(self, data):
        input_lines = []
        labels = []
        for i in data:
            input_lines.append(i['text'])
            labels.append(i['label'])
        return input_lines, labels

    def split_and_pad_data(self, data, augment = False):
        input_lines, labels = self.format_data(data)
        if augment and self.augmentors is not None:
            for augmentor in self.augmentors:
                input_lines = augmentor.augment_dataset(input_lines, self.augmentation_percentage, has_label = False)
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
            data_seq.append({"input_id": input_id, "attention_mask": attention_mask, "label": torch.tensor(label, dtype = torch.long)})
        return data_seq

    def setup(self, stage: str):
        dataset = load_dataset("glue", self.glue_task)
        self.train_dataset = dataset['train']
        self.validation_dataset = dataset['validation']
        self.test_dataset = dataset['test']

    def train_dataloader(self):
        return DataLoader(self.split_and_pad_data(self.train_dataset, augment = True), batch_size=self.batch_size, shuffle = True)

    def val_dataloader(self):
        return DataLoader(self.split_and_pad_data(self.validation_dataset), batch_size=self.batch_size, shuffle = True)

    def test_dataloader(self):
        return DataLoader(self.split_and_pad_data(self.test_dataset), batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.split_and_pad_data(self.test_dataset), batch_size=self.batch_size)

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        pass