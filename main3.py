from src.models import better_text_classify, text_classify, seq2seq_translate
import random, torch, numpy
import pytorch_lightning.utilities.seed as plseed
from googletrans import Translator

from argparse import ArgumentParser

# from src.helpers import EnglishPreProcessor, Logger

SEED = 0
random.seed(SEED)
numpy.random.seed(SEED)
g = torch.Generator()
g.manual_seed(SEED)
torch.cuda.manual_seed(SEED)           
torch.cuda.manual_seed_all(SEED)
plseed.seed_everything(SEED)

if __name__ == "__main__":
    # learning_rate = 1.2
    # accuracy = text_classify(augmentors = [Back_Translator("en", "de")], learning_rate = learning_rate, augmentation_percentage = 0.25, dataset_percentage = 0.5)
    # if(accuracy > max_acc):
    #     max_acc = accuracy
    # print("Max accuracy = ", max_acc)

    # accuracy = text_classify(augmentors = [Synonym_Replacer("english"), Insertor("english")], learning_rate = learning_rate)
    # if(accuracy > max_acc):
    #     max_acc = accuracy
    # print("Max accuracy = ", max_acc)
    better_text_classify()

    # python main.py --devices=-1 --accelerator="auto" --max_epochs=20 --auto_scale_batch_size="binsearch" --auto_select_gpus=True