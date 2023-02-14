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


class BiasDetectionDataModule(pl.LightningDataModule):
    def __init__(self, dataset_percentage, augmentors = [], batch_size: int = 32):
        super().__init__()
        self.batch_size = batch_size
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)
        self.augmentors = augmentors
        self.dataset_percentage = dataset_percentage
        self.id2label =  {0: "Non-biased", 1: "No agreement", 2: "Biased"}
        self.label2id = {"Non-biased": 0, "No agreement": 1, "Biased": 2}


    def format_data(self, data):
        return data['text'], data['label_bias']

    def split_and_pad_data(self, data, augment = False):
        input_lines, labels = self.format_data(data)
        if augment and self.augmentors is not None:
            for augmentor in self.augmentors:
                input_lines = augmentor.augment_dataset(input_lines, has_label = False)
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
        try:
            PATH_sg1 = "src/data/bias_detection_data_files/final_labels_SG1.xlsx"
            PATH_sg2 = "src/data/bias_detection_data_files/final_labels_SG2.xlsx"
            df_sg1 = pd.read_excel(PATH_sg1)
            df_sg2 = pd.read_excel(PATH_sg2)
        except FileNotFoundError:
            PATH_sg1 = "project/src/data/bias_detection_data_files/final_labels_SG1.xlsx"
            PATH_sg2 = "project/src/data/bias_detection_data_files/final_labels_SG2.xlsx"
            df_sg1 = pd.read_excel(PATH_sg1)
            df_sg2 = pd.read_excel(PATH_sg2)
        df_sg1 = df_sg1[["text", "label_bias"]]
        df_sg2 = df_sg2[["text", "label_bias"]]

        df = pd.concat([df_sg1, df_sg2])
        df = df.sample(frac=1).reset_index() # Shuffles df

        df['label_bias'] = df['label_bias'].map(self.label2id)
        
        train_size = 0.9
        valid_size = 0.05

        train_index = int(len(df) * train_size)
        valid_index = int(len(df) * valid_size)
        train = df[0 : train_index]
        self.train_dataset = train[:int(len(train) * self.dataset_percentage)]
        self.validation_dataset = df[train_index : train_index + valid_index]
        self.test_dataset = df[train_index + valid_index : ]
        
        # self.train_dataset = dataset['train']
        # self.validation_dataset = dataset['validation']
        # self.test_dataset = dataset['test']

    def train_dataloader(self):
        return DataLoader(self.split_and_pad_data(self.train_dataset, augment = True), batch_size=self.batch_size, shuffle = True)

    def val_dataloader(self):
        return DataLoader(self.split_and_pad_data(self.validation_dataset), batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.split_and_pad_data(self.test_dataset), batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.split_and_pad_data(self.test_dataset), batch_size=self.batch_size)

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        pass