import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from ..helpers import CustomTokenizerEncoder


class TranslationDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32, tokenize = True):
        super().__init__()

        self.dataset = load_dataset("iwslt2017", "iwslt2017-en-de")
        self.batch_size = batch_size

    def build_vocab(self):
        en_lines = []
        de_lines = []
        for line in self.dataset['train']:
            en_lines.append(line['translation']['en'])
            de_lines.append(line['translation']['de'])

        self.en_tokenizer = CustomTokenizerEncoder(en_lines, append_eos = True)
        self.de_tokenizer = CustomTokenizerEncoder(de_lines, append_eos = True)
        self.padding_index = self.en_tokenizer.padding_index
        
        self.input_vocab_size = self.en_tokenizer.vocab_size
        self.output_vocab_size = self.de_tokenizer.vocab_size

    def split_and_pad_data(self, data):
        en_index_lines = []
        de_index_lines = []
        en_lengths = []
        de_lengths = []

        for line in data:
            en_line = self.en_tokenizer.encode(line['translation']['en'])
            en_index_lines.append(en_line)
            en_lengths.append(len(en_line))
            de_line = self.de_tokenizer.encode(line['translation']['de'])
            de_index_lines.append(de_line)
            de_lengths.append(len(de_line))
        en_index_lines = pad_sequence(en_index_lines, batch_first = True)
        de_index_lines = pad_sequence(de_index_lines, batch_first = True)

        data_seq = []
        for i in range(len(en_index_lines)):
            if(en_lengths[i] > 0 and de_lengths[i] > 0):
                input = {'src': en_index_lines[i], 'src_len': en_lengths[i]}
                output = {'trg': de_index_lines[i], 'trg_len': de_lengths[i]}
                data_seq.append((input,output))
        return data_seq

    def setup(self, stage: str):
        self.build_vocab() 
        self.train_iterator = self.split_and_pad_data(self.dataset['train'])
        self.valid_iterator = self.split_and_pad_data(self.dataset['validation'])
        self.test_iterator = self.split_and_pad_data(self.dataset['test'])

    def train_dataloader(self):
        return DataLoader(self.train_iterator, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.valid_iterator, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_iterator, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.test_iterator, batch_size=self.batch_size)

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        pass