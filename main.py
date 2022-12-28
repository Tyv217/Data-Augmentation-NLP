from src.models import text_classify, seq2seq_translate
import random, torch, numpy
import pytorch_lightning.utilities.seed as plseed

SEED = 0
random.seed(SEED)
numpy.random.seed(SEED)
g = torch.Generator()
g.manual_seed(SEED)
torch.cuda.manual_seed(SEED)               
torch.cuda.manual_seed_all(SEED)
plseed.seed_everything(SEED)

if __name__ == "__main__":
    # AUGMENTATION_PERCENTAGE = 0.16

    # accuracy = text_classify(AUGMENTATION_PERCENTAGE)

    # with open("accuracies.txt", "a") as f:
    #     f.write("Percentage: " + str(AUGMENTATION_PERCENTAGE * 100) + "%, accuracy: " + "{0:.3g}\n".format(accuracy))

    seq2seq_translate()

    # python main.py --devices=-1 --accelerator="auto" --max_epochs=20 --auto_scale_batch_size="binsearch" --auto_select_gpus=True
