from abc import ABC, abstractmethod
import torch
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

class PreProcessor(ABC):
    def __init__(self, data_iter):
        self.data_iter = data_iter
        self.tokenizer = self.get_tokenizer()
        self.vocab = build_vocab_from_iterator(self.yield_tokens(), specials = ["<unk>"])
        self.vocab.set_default_index(self.vocab["<unk>"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataloader = DataLoader(data_iter, batch_size = 8, shuffle = False, collate_fn = self.collate_batch)

    @abstractmethod
    def get_tokenizer(self):
        pass
            
    def yield_tokens(self):
        for _, text in self.data_iter:
            yield self.tokenizer(text)

    def get_vocab(self):
        return self.vocab

    def text_pipeline(self, input_string):
        return self.vocab(self.tokenizer(input_string))

    def label_pipeline(self, input_string):
        return int(input_string) - 1
    
    def collate_batch(self, data_batch):
        label_list, text_list, offsets = [], [], [0]
        for (label, text) in data_batch:
            label_list.append(self.label_pipeline(label))
            text_indices = torch.tensor(self.text_pipeline(text), dtype = torch.int64)
            text_list.append(text_indices)
            offsets.append(text_indices.size(dim = 0))
        label_list = torch.tensor(label_list, dtype = torch.int64)
        print(offsets)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim = 0)
        print(offsets)
        text_list = torch.cat(text_list)
        return label_list.to(self.device), text_list.to(self.device), offsets.to(self.device)

    def get_dataloader(self):
        return self.dataloader

class EnglishPreProcessor(PreProcessor):
    def get_tokenizer(self):
        return get_tokenizer('basic_english')