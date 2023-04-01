import random
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import T5Tokenizer


class TranslationDataModule(pl.LightningDataModule):
    def __init__(self, model_name = "t5-small", dataset_percentage = 1, batch_size: int = 32, task_prefix = "translate English to German: ", input_language = "en", output_language = "de", model_max_length = 256):
        super().__init__()

        self.dataset = load_dataset("iwslt2017", "iwslt2017-" + input_language + "-" + output_language)
        self.batch_size = batch_size
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, model_max_length = model_max_length)
        self.task_prefix = task_prefix
        self.input_language = input_language
        self.output_language = output_language
        self.dataset_percentage = dataset_percentage
        train = list(self.dataset['train'])
        random.shuffle(train)
        self.train_dataset = train[:int(len(train) * self.dataset_percentage)]
        self.valid_dataset = self.dataset['validation']
        self.test_dataset = self.dataset['test']

    def format_data(self, data):
        input_lines = []
        output_lines = []
        for line in data:
            input_lines.append(line['translation'][self.input_language])
            output_lines.append(line['translation'][self.output_language])
        return input_lines, output_lines

    def split_and_tokenize(self, data, format = True):
        if format:
            input_lines, output_lines = self.format_data(data)
        else:
            input_lines, output_lines = data

        output_encoding = self.tokenizer(
            output_lines,
            padding = "longest",
            truncation = True,
            return_tensors = "pt",
        )

        labels = output_encoding.input_ids

        labels[labels == self.tokenizer.pad_token_id] = -100

        data_seq = []   
        for input_line, label in zip(input_lines, labels):
            data_seq.append({"input_lines": input_line, "label": label})
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
