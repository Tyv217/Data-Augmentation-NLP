import torch, time, random
import pytorch_lightning as pl
from argparse import ArgumentParser
from ..helpers import set_seed, parse_augmentors, plot_and_compare_emb, plot_emb
from ..data import IWSLT17DataModule, AGNewsDataModule, ColaDataModule, TwitterDataModule, BabeDataModule
from .data_augmentors import Synonym_Replacer, Back_Translator, Insertor, Deletor
from sentence_transformers import SentenceTransformer

def visualize_back_translation_embedding():
    
    parser = ArgumentParser()

    # add PROGRAM level args
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task", type=str, default="babe")
    parser.add_argument("--deterministic", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--augmentor", type=str, default="bt")
    parser.add_argument("--augmentation_params", type=int, default=0)

    args = parser.parse_args()
    set_seed(args.seed)

    augmentator_mapping = {"sr": Synonym_Replacer("english"), "bt": Back_Translator("en"), "in": Insertor("english"), "de": Deletor()}
    augmentor = augmentator_mapping[args.augmentor]

    data_modules = {"cola": ColaDataModule, "twitter": TwitterDataModule, "babe": BabeDataModule}

    data = data_modules[args.task](
        dataset_percentage = 1,
        augmentors = [],
        batch_size = args.batch_size
    )

    data.prepare_data()
    data.setup("fit")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data1 = list(data.get_dataset_text())
    # random.shuffle(train_data1)
    # train_data1 = train_data1[:1000]
    train_data2 = train_data1.copy()

    augmentor.set_augmentation_percentage(args.augmentation_params) # So guaranteed augmentation
    print("Start augmenting!")
    start_time = time.time()
    train_data2 = augmentor.augment_dataset(train_data2)
    print("Finish augmenting! Time taken: " + str(time.time() - start_time))

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(device)

    embeddings1 = model.encode(train_data1)
    embeddings2 = model.encode(train_data2)

    difference = []
    
    for e1, e2 in zip(embeddings1, embeddings2):
        difference.append(e2 - e1)

    # plot_and_compare_emb(embeddings1, embeddings2, args.task + '.png')

    plot_emb(difference, args.task + '_' + args.augmentor + '.png')

    




