import torch, time
from torchtext.datasets import AG_NEWS as agnews
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
from ..helpers import EnglishPreProcessor, Logger

# TODO: Change to better implementation
# Taken from https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html
class TextClassifierEmbeddingModel(torch.nn.Module):

    def __init__(self, vocab_size, embed_dim, num_classes):
        super(TextClassifierEmbeddingModel, self).__init__()
        self.embedding = torch.nn.EmbeddingBag(vocab_size, embed_dim, sparse = True)
        self.fc = torch.nn.Linear(embed_dim, num_classes)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)



    
