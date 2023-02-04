from src.models import better_text_classify, text_classify, seq2seq_translate, Synonym_Replacer, Back_Translator, Insertor, Deletor
import random, torch, numpy
import pytorch_lightning.utilities.seed as plseed
from googletrans import Translator

from argparse import ArgumentParser

# from src.helpers import EnglishPreProcessor, Logger

# SEED = 0
# print("Seed 1")
# random.seed(SEED)
# print("Seed 2")
# numpy.random.seed(SEED)
# print("Seed 3")
# g = torch.Generator()
# print("Seed 4")
# g.manual_seed(SEED)
# print("Seed 5")
# torch.cuda.manual_seed(SEED)     
# print("Seed 6")          
# torch.cuda.manual_seed_all(SEED)
# print("Seed 7")
# plseed.seed_everything(SEED)
# print("Seed 8")

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
    seq2seq_translate()

    # python main.py --devices=-1 --accelerator="auto" --max_epochs=20 --auto_scale_batch_size="binsearch" --auto_select_gpus=True