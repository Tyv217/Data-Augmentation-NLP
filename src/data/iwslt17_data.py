import random
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import T5Tokenizer


class TranslationDataModule(pl.LightningDataModule):
    def __init__(self, model_name = "t5-small", dataset_percentage = 1, augmentors = [], batch_size: int = 32, task_prefix = "translate English to German: ", input_language = "en", output_language = "de", model_max_length = 256):
        super().__init__()

        self.dataset = load_dataset("iwslt2017", "iwslt2017-" + input_language + "-" + output_language)
        self.batch_size = batch_size
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, model_max_length = model_max_length)
        self.task_prefix = task_prefix
        self.input_language = input_language
        self.output_language = output_language
        self.augmentors_on_words = []
        self.augmentors_on_tokens = []
        for augmentor in augmentors:
            if augmentor.operate_on_tokens:
                self.augmentors_on_tokens.append(augmentor)
            else:
                self.augmentors_on_words.append(augmentor)
        self.dataset_percentage = dataset_percentage
        train = list(self.dataset['train'])
        random.shuffle(train)
        self.train_dataset = train[:int(len(train) * self.dataset_percentage)]

    def format_data(self, data):
        input_lines = []
        output_lines = []
        for line in data:
            input_lines.append(line['translation'][self.input_language])
            output_lines.append(line['translation'][self.output_language])
        return input_lines, output_lines

    def split_and_pad_data(self, data, augment = False, augment_target = False):
        input_lines, output_lines = self.format_data(data)

        if augment and self.augmentors_on_words is not None:
            for augmentor in self.augmentors_on_words:
                if augmentor.require_label:
                    zipped_lines = zip(labels, input_lines)
                    input_lines, labels = augmentor.augment_dataset(zipped_lines, has_label = True)
                else:
                    input_lines = augmentor.augment_dataset(input_lines, has_label = False)

        input_encoding = self.tokenizer(
            # [self.task_prefix + sequence for sequence in input_lines],
            input_lines,
            padding = "longest",
            truncation = True,
            return_tensors = "pt",
        )
        input_ids, attention_masks = input_encoding.input_ids, input_encoding.attention_mask

        if augment and self.augmentor_on_tokens is not None:
            for augmentor in self.augmentor_on_tokens:
                if augmentor.require_label:
                    zipped_lines = zip(labels, input_ids)
                    input_ids, labels = augmentor.augment_dataset(zipped_lines, has_label = True)
                else:
                    input_ids = augmentor.augment_dataset(input_ids, has_label = False)

        output_encoding = self.tokenizer(
            output_lines,
            padding = "longest",
            truncation = True,
            return_tensors = "pt",
        )

        labels = output_encoding.input_ids

        labels[labels == self.tokenizer.pad_token_id] = -100

        data_seq = []   
        for input_id, attention_mask, label in zip(input_ids, attention_masks, labels):
            data_seq.append({"input_id": input_id, "attention_mask": attention_mask, "label": label})
        return data_seq

    def setup(self, stage: str):
        self.valid_iterator = self.split_and_pad_data(self.dataset['validation'])
        self.test_iterator = self.split_and_pad_data(self.dataset['test'])

    def train_dataloader(self):
        self.train_iterator = self.split_and_pad_data(self.train_dataset, augment = True)
        return DataLoader(self.train_iterator, batch_size=self.batch_size, shuffle = True)

    def val_dataloader(self):
        return DataLoader(self.valid_iterator, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_iterator, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.test_iterator, batch_size=self.batch_size)

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        pass

    def get_dataset_text(self):
        text, _ = self.format_data(self.train_dataset)
        return text
