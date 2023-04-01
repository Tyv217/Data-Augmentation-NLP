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


class TwitterDataModule(pl.LightningDataModule):
    def __init__(self, dataset_percentage, twitter_task = "sentiment", batch_size: int = 32):
        super().__init__()
        self.batch_size = batch_size
        self.twitter_task = twitter_task
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)
        self.dataset_percentage = dataset_percentage
        self.id2label =  {0: "negative", 1: "neutral", 2: "positive"}
        self.label2id = {"negative": 0, "neutral": 1, "positive": 2}
        dataset = load_dataset("tweet_eval", self.twitter_task)
        train = list(dataset['train'])
        random.shuffle(train)
        self.train_dataset = train[:int(len(train) * self.dataset_percentage)]
        self.valid_dataset = dataset['validation']
        self.test_dataset = dataset['test']

    def preprocess(self, text):
        data = []
        for t in text.split(" "):
            if(t.startswith('@') and len(t) > 1):
                data.append("@user")
            elif t.startswith('http'):
                data.append("http")
            else:
                data.append(t)
        return " ".join(data)

    def format_data(self, data):
        input_lines = []
        labels = []
        for i in data:
            input_lines.append(self.preprocess(i['text']))
            labels.append(i['label'])
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

    def setup(self, stage: str):
        pass

    def train_dataloader(self):
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