from src.models import better_text_classify, text_classify, seq2seq_translate, image_classify
import random, torch, numpy
import pytorch_lightning.utilities.seed as plseed
from googletrans import Translator
from argparse import ArgumentParser

from src.helpers import EnglishPreProcessor, Logger

SEED = 0
random.seed(SEED)
numpy.random.seed(SEED)
g = torch.Generator()
g.manual_seed(SEED)
torch.cuda.manual_seed(SEED)     
torch.cuda.manual_seed_all(SEED)
plseed.seed_everything(SEED)

if __name__ == "__main__":
    image_classify()