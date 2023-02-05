from abc import ABC, abstractmethod
import torch
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchnlp.encoders.text.static_tokenizer_encoder import StaticTokenizerEncoder
from torchnlp.encoders.text.text_encoder import TextEncoder

class PreProcessor(ABC):
    def __init__(self, data_iters):
        self.data_iters = data_iters
        self.tokenizer = self.get_tokenizer()
        self.vocab = build_vocab_from_iterator(self.yield_tokens(), specials = ["<unk>"])
        self.vocab.set_default_index(self.vocab["<unk>"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def get_tokenizer(self):
        pass
            
    def get_vocab(self):
        return self.vocab
        
    def get_dataloader(self):
        return self.dataloader

    def get_device(self):
        return self.device

    def yield_tokens(self):
        for _, text in self.data_iters:
            yield self.tokenizer(text)

    def get_text_indices(self, input_string):
        return self.vocab(self.tokenizer(input_string))

    def get_label(self, input_string):
        return int(input_string) - 1
    
    def collate_batch(self, data_batch):
        text_list, label_list, offsets = [], [], [0]
        for (label, text) in data_batch:
            text_indices = torch.tensor(self.get_text_indices(text))
            text_list.append(text_indices)
            label_list.append(self.get_label(label))
            offsets.append(text_indices.size(dim = 0))
        label_list = torch.tensor(label_list, dtype = torch.int64)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim = 0)
        text_list = torch.cat(text_list)
        return label_list.to(self.device), text_list.to(self.device), offsets.to(self.device)
    
    def get_dataloader(self, data_iters, batch_size, shuffle = False):
        return DataLoader(data_iters, batch_size = batch_size, shuffle = shuffle, collate_fn = self.collate_batch)

class EnglishPreProcessor(PreProcessor):
    def get_tokenizer(self):
        return get_tokenizer('basic_english')

class CustomTokenizerEncoder(StaticTokenizerEncoder):
    def decode_index_to_token(self, index):
        if(index > self.vocab_size):
            return '<unk>'
        else:
            return self.index_to_token[index]

    def decode(self, encoded):
        """ Decodes a tensor into a sequence.

        Args:
            encoded (torch.Tensor): Encoded sequence.

        Returns:
            str: Sequence decoded from ``encoded``.
        """
        encoded = TextEncoder.decode(self, encoded)
        tokens = [self.decode_index_to_token(index) for index in encoded]
        return self.detokenize(tokens)