from src.models import text_classify, seq2seq_translate
import random, torch, numpy

SEED = 0
random.seed(SEED)
numpy.random.seed(SEED)
g = torch.Generator()
g.manual_seed(SEED)
torch.cuda.manual_seed(SEED)               
torch.cuda.manual_seed_all(SEED)

if __name__ == "__main__":
    AUGMENTATION_PERCENTAGE = 0.16

    accuracy = text_classify(AUGMENTATION_PERCENTAGE)

    with open("accuracies.txt", "a") as f:
        f.write("Percentage: " + str(AUGMENTATION_PERCENTAGE * 100) + "%, accuracy: " + "{0:.3g}\n".format(accuracy))

    # seq2seq_translate()
