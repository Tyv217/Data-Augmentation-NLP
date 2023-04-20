import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import DistilBertTokenizer
from torchtext.datasets import AG_NEWS as agnews
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
import numpy as np, pandas as pd
import random


class BabeDataModule(pl.LightningDataModule):
    def __init__(self, dataset_percentage, augmentors = [], batch_size: int = 32, tokenize = True):
        super().__init__()
        self.batch_size = batch_size
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)
        self.dataset_percentage = dataset_percentage
        self.id2label =  {0: "Non-biased", 1: "Biased"}
        self.label2id = {"Non-biased": 0, "Biased": 1}
        try:
            PATH_sg2 = "src/data/external/bias_detection_data_files/final_labels_SG2.xlsx"
            df_sg2 = pd.read_excel(PATH_sg2)
        except FileNotFoundError:
            try:
                PATH_sg2 = "project/src/data/external/bias_detection_data_files/final_labels_SG2.xlsx"
                df_sg2 = pd.read_excel(PATH_sg2)
            except FileNotFoundError:
                PATH_sg2 = "Data-Augmentation-NLP/src/data/external/bias_detection_data_files/final_labels_SG2.xlsx"
                df_sg2 = pd.read_excel(PATH_sg2)
        df_sg2 = df_sg2[["text", "label_bias"]]

        df = df_sg2
        df = df.sample(frac=1).reset_index() # Shuffles df
        df = df.drop(df[df['label_bias'] == 'No agreement'].index)
        
        df['label_bias'] = df['label_bias'].map(self.label2id)
        
        train_size = 0.9
        valid_size = 0.05

        train_index = int(len(df) * train_size)
        valid_index = int(len(df) * valid_size)
        train = df[0 : train_index]
        self.train_dataset = train[:int(len(train) * self.dataset_percentage)]
        self.valid_dataset = df[train_index : train_index + valid_index]
        self.test_dataset = df[train_index + valid_index : ]
        self.tokenize = tokenize
        self.augmentors = augmentors


    def format_data(self, data):
        return data['text'], np.identity(len(self.id2label))[data['label_bias']]

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

    def setup(self, stage: str):
        pass
        
        # self.train_dataset = dataset['train']
        # self.validation_dataset = dataset['validation']
        # self.test_dataset = dataset['test']

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