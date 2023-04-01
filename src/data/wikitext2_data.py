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

class WikiText2DataModule(pl.LightningDataModule):
    def __init__(self, dataset_percentage, augmentors = [], batch_size: int = 32):
        super().__init__()
        self.batch_size = batch_size
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)
        self.augmentors = augmentors
        self.dataset_percentage = dataset_percentage
        label_names = ["Company",
                        "EducationalInstitution",
                        "Artist",
                        "Athlete",
                        "OfficeHolder",
                        "MeanOfTransportation",
                        "Building",
                        "NaturalPlace",
                        "Village",
                        "Animal",
                        "Plant",
                        "Album",
                        "Film",
                        "WrittenWork"]
        self.id2label =  {}
        self.label2id = {}

        for i, name in zip(np.arange(len(label_names)), label_names):
            self.id2label[i] = name
            self.label2id[name] = i

        dataset = load_dataset("dbpedia_14")
        train = list(dataset['train'])
        random.shuffle(train)
        self.train = train[:int(len(train) * self.dataset_percentage)]
        self.test_dataset = list(dataset['test'])

    def format_data(self, data):
        input_lines = []
        labels = []
        for i in data:
            input_lines.append(i['text'])
            labels.append(i['coarse_label'])
        return input_lines, np.identity(len(self.id2label))[labels]

    def split_and_tokenize(self, data, format = True, augment = False):
        if format:
            input_lines, labels = self.format_data(data)
        else:
            input_lines, labels = data
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
            data_seq.append({"input_id": input_id, "attention_mask": attention_mask, "label": torch.tensor(label, dtype = torch.long)})
        return data_seq

    def shuffle_train_valid_iters(self):
        num_train = int(len(self.train) * 0.95)
        self.train_dataset, self.valid_dataset = random_split(self.train, [num_train, len(self.train) - num_train])

    def setup(self, stage: str):
        pass

    def train_dataloader(self):
        self.shuffle_train_valid_iters()
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