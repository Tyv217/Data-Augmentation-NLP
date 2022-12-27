
from datasets import load_dataset
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data import get_tokenizer

from torchnlp.encoders.text.static_tokenizer_encoder import StaticTokenizerEncoder
from torch.nn.utils.rnn import pad_sequence

def main():
    dataset = load_dataset("iwslt2017", "iwslt2017-en-de")
    en_lines = []
    de_lines = []
    for line in dataset['train']:
        en_lines.append(line['translation']['en'])
        de_lines.append(line['translation']['de'])
    
    en_tokenizer = StaticTokenizerEncoder(en_lines, append_eos = True)
    de_tokenizer = StaticTokenizerEncoder(de_lines, append_eos = True)

    en_index_lines = []
    de_index_lines = []
    for line in dataset['train']:
        en_index_lines.append(en_tokenizer.encode(line['translation']['en']))
        de_index_lines.append(de_tokenizer.encode(line['translation']['de']))
    en_index_lines = pad_sequence(en_index_lines, batch_first = True)
    de_index_lines = pad_sequence(de_index_lines, batch_first = True)

if __name__ == "__main__":
    line = [1,2,3,4,5]
    print(line[-1])