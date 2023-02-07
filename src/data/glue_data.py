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
    def __init__(self, augmentation_percentage, dataset_percentage, augmentors, batch_size: int = 32):
        super().__init__()
        self.batch_size = batch_size
        self.glue_task = "cola"
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)
        self.augmentation_percentage = augmentation_percentage
        self.augmentors = augmentors
        self.dataset_percentage = dataset_percentage

    def format_data(self, data):
        input_lines = []
        labels = []
        for i in data:
            input_lines.append(i['sentence'])
            labels.append(i['label'])
        return input_lines, labels

    def split_and_pad_data(self, data):
        input_lines, labels = self.format_data(data)
        if self.augmentors is not None and self.augmentation_percentage is not None:
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
        import pdb
        pdb.set_trace()
        return data_seq

    def shuffle_train_valid_iters(self):
        num_train = int(len(self.train_dataset) * 0.95)
        self.split_train, self.split_valid = random_split(self.train_dataset, [num_train, len(self.train_dataset) - num_train])

    def setup(self, stage: str):
        dataset = load_dataset("glue", self.glue_task)
        self.train_dataset = dataset['train']
        self.validation_dataset = dataset['validation']
        self.test_dataset = dataset['test']

    def train_dataloader(self):
        self.shuffle_train_valid_iters()
        return DataLoader(self.split_and_pad_data(self.train_dataset), batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.split_and_pad_data(self.validation_dataset), batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.split_and_pad_data(self.test_dataset), batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.split_and_pad_data(self.test_dataset), batch_size=self.batch_size)

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        pass