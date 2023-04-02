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


class TrecDataModule(pl.LightningDataModule):
    def __init__(self, dataset_percentage, augmentors = [], batch_size: int = 32, tokenize = True):
        super().__init__()
        self.batch_size = batch_size
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)
        self.dataset_percentage = dataset_percentage
        self.id2label =  {0: "Abbreviation", 1: "Entity", 2: "Description and abstract concept", 3: "Human being", 4: "Location", 5: "Numeric value"}
        self.label2id = {"Abbreviation": 0, "Entity": 1, "Description and abstract concept": 2, "Human being": 3, "Location": 4, "Numeric value": 5}
        dataset = load_dataset("trec")
        train = list(dataset['train'])
        random.shuffle(train)
        self.train = train[:int(len(train) * self.dataset_percentage)]
        self.test_dataset = list(dataset['test'])
        self.augmentors = augmentors
        self.tokenize = tokenize
        self.augmentors = augmentors

    def format_data(self, data):
        input_lines = []
        labels = []
        for i in data:
            input_lines.append(i['text'])
            labels.append(i['coarse_label'])
        return input_lines, np.identity(len(self.id2label))[labels]

    def split_and_tokenize(self, data, format = True):
        if format:
            input_lines, labels = self.format_data(data)
        else:
            input_lines, labels = data
        
        data_seq = []
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
        return DataLoader(self.split_and_tokenize(self.train_dataset), batch_size=self.batch_size, shuffle = True)

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