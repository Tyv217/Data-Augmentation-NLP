from src.models import text_classify, seq2seq_translate, Synonym_Replacer, Back_Translator
import random, torch, numpy
import pytorch_lightning.utilities.seed as plseed

SEED = 42
random.seed(SEED)
numpy.random.seed(SEED)
g = torch.Generator()
g.manual_seed(SEED)
torch.cuda.manual_seed(SEED)               
torch.cuda.manual_seed_all(SEED)
plseed.seed_everything(SEED)

if __name__ == "__main__":

    accuracy = text_classify(Back_Translator("en", "de"))

    # seq2seq_translate()

    # python main.py --devices=-1 --accelerator="auto" --max_epochs=20 --auto_scale_batch_size="binsearch" --auto_select_gpus=True