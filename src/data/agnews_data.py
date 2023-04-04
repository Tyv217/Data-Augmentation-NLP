import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer
from torchtext.datasets import AG_NEWS as agnews
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
import numpy as np
import random


class AGNewsDataModule(pl.LightningDataModule):
    def __init__(self, dataset_percentage,  augmentors = [], batch_size: int = 32, tokenize = True):
        super().__init__()
        self.batch_size = batch_size
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)
        self.dataset_percentage = dataset_percentage
        self.id2label = {0: "WORLD", 1: "SPORTS", 2: "BUSINESS", 3: "SCIENCE"}
        self.label2id = {"WORLD": 0, "SPORTS": 1, "BUSINESS": 2, "SCIENCE": 3}
        
        train_iter, test_iter = agnews()
        train_dataset = list(train_iter)
        random.shuffle(train_dataset)
        self.train = to_map_style_dataset(train_dataset[:int(len(train_dataset) * self.dataset_percentage)])
        self.test_dataset = to_map_style_dataset(test_iter)
        self.tokenize = tokenize
        self.augmentors = augmentors

    def format_data(self, data):
        labels, inputs = zip(*data)
        labels = np.array(labels) - 1
        return list(inputs), np.identity(len(self.id2label))[labels]

    def split_and_tokenize(self, data, format = True, augment = False):
        if format:
            input_lines, labels = self.format_data(data)
        else:
            input_lines, labels = data
        
        data_seq = []
        if self.tokenize:
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
            for input_id, attention_mask, label in zip(input_ids, attention_masks, labels):
                data_seq.append({"input_id": input_id, "attention_mask": attention_mask, "label": torch.tensor(label, dtype = torch.float)})
        else:
            for input_line, label in zip(input_lines, labels):
                data_seq.append({"input_lines": input_line, "label": torch.tensor(label, dtype = torch.float)})
        return data_seq

    def shuffle_train_valid_iters(self):
        num_train = int(len(self.train) * 0.95)
        self.train_dataset, self.valid_dataset = random_split(self.train, [num_train, len(self.train) - num_train])

    def setup(self, stage: str):
        pass

    def train_dataloader(self):
        self.shuffle_train_valid_iters()
        return DataLoader(self.split_and_tokenize(self.train_dataset, augment = True), batch_size=self.batch_size, shuffle = True, num_workers = 128)

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