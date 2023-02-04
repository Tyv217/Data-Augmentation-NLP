from src.models import better_text_classify, text_classify, seq2seq_translate, Synonym_Replacer, Back_Translator, Insertor, Deletor
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
    parser = ArgumentParser()
    parser.add_argument("--data_augmentors", type = lambda s: [item for item in s.split(",")])
    parser.add_argument("--output_location", type = str)
    parser.add_argument("--augmentation_percentage", type = int)
    parser.add_argument("--dataset_percentage", type = int)
    args = parser.parse_args()
    augmentor_mapping = {"sr": Synonym_Replacer("english"), "bt": Back_Translator("en", "de"), "in": Insertor("english"), "de": Deletor()}
    augmentors = [augmentor_mapping[i] for i in args.data_augmentors if i != '']
    augmentation_percentage = args.augmentation_percentage / 10
    dataset_percentage = args.dataset_percentage / 100
    max_acc = 0
    max_lr = 0
    for i in [1, 1.25, 1.5, 1.75]:
        learning_rate = i
        accuracy = text_classify(augmentors = augmentors, learning_rate = learning_rate, dataset_percentage = dataset_percentage, augmentation_percentage = augmentation_percentage)
        if(accuracy > max_acc):
            max_acc = accuracy
            max_lr = learning_rate
    print("Max accuracy = ", max_acc)
    print("Max learning rate = ", max_lr)
    if args.output_location is not None:
        augmentor_names = {"sr": "Synonym_Replacer", "bt": "Back_Translator", "in": "Insertor", "de": "Deletor"}
        with open(args.output_location, "a") as f:
            augmentors_used = (", ").join([augmentor_names[i] for i in args.data_augmentors if i != ""])
            f.write("Dataset percentage: " + str(args.dataset_percentage) + "%\nAugmentation percentage: " + str(args.augmentation_percentage * 10) +"%\nAugmentors used: " + augmentors_used + "\n")
            f.write("Max accuracy: " + str(max_acc) + ", corresponding learning rate: " + str(max_lr) + "\n\n")
    # if(accuracy > max_acc):
    #     max_acc = accuracy
    # print("Max accuracy = ", max_acc)

    # accuracy = text_classify(augmentors = [Synonym_Replacer("english"), Insertor("english")], learning_rate = learning_rate)
    # if(accuracy > max_acc):
    #     max_acc = accuracy
    # print("Max accuracy = ", max_acc)
    # seq2seq_translate()

    # python main.py --devices=-1 --accelerator="auto" --max_epochs=20 --auto_scale_batch_size="binsearch" --auto_select_gpus=True