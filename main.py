from src.models import text_classify, seq2seq_translate, Synonym_Replacer, Back_Translator
import random, torch, numpy
import pytorch_lightning.utilities.seed as plseed
from googletrans import Translator

SEED = 0
random.seed(SEED)
numpy.random.seed(SEED)
g = torch.Generator()
g.manual_seed(SEED)
torch.cuda.manual_seed(SEED)               
torch.cuda.manual_seed_all(SEED)
plseed.seed_everything(SEED)

if __name__ == "__main__":
    # AUGMENTATION_PERCENTAGE = 0.01

    # accuracy = text_classify(AUGMENTATION_PERCENTAGE, Back_Translator(Translator(), "en", "de"))

    # with open("accuracies.txt", "a") as f:
    #     f.write("Percentage: " + str(AUGMENTATION_PERCENTAGE * 100) + "%, accuracy: " + "{0:.3g}\n".format(accuracy))

    seq2seq_translate()

    # python main.py --devices=-1 --accelerator="auto" --max_epochs=20 --auto_select_gpus=True
