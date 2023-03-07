import torch, time, random
import pytorch_lightning as pl
from argparse import ArgumentParser
from ..helpers import set_seed, parse_augmentors, plot_and_compare_emb, plot_emb
from ..data import TranslationDataModule, AGNewsDataModule, GlueDataModule, TwitterDataModule, BiasDetectionDataModule
from .data_augmentors import Synonym_Replacer, Back_Translator, Insertor, Deletor
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import os
import pathlib

def visualize_back_translation_embedding():
    
    parser = ArgumentParser()

    # add PROGRAM level args
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task", type=str, default="bias_detection")
    parser.add_argument("--deterministic", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--augmentor", type=str, default="bt")
    parser.add_argument("--augmentation_params", type=int, default=0)
    parser.add_argument("--datapoints", type=int, default=250)

    args = parser.parse_args()
    set_seed(args.seed)

    augmentator_mapping = {"sr": Synonym_Replacer("english"), "bt": Back_Translator("en"), "in": Insertor("english"), "de": Deletor()}
    augmentor = augmentator_mapping[args.augmentor]

    data_modules = {"glue": GlueDataModule, "twitter": TwitterDataModule, "bias_detection": BiasDetectionDataModule}

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
    
    AUGMENT_LOOPS = 3

    dir = pathlib.Path(__file__).parent.resolve()
    filename = '../data/augmented_data/' + args.task + '_' + args.augmentor + str(AUGMENT_LOOPS) + '.csv'
    filepath = os.path.join(dir, filename)

    try:
        df = pd.read_csv(filepath)
        train_data2 = list(df['0'])
    except:
        train_data2 = train_data1.copy()
        augmentor.set_augmentation_percentage(args.augmentation_params) # So guaranteed augmentation
        print("Start augmenting!")
        start_time = time.time()
        for i in range(AUGMENT_LOOPS):
            train_data2 = augmentor.augment_dataset(train_data2)
        print("Finish augmenting! Time taken: " + str(time.time() - start_time))
        df = pd.DataFrame(train_data2)
        df.to_csv(filepath, index = False) 

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(device)

    embeddings1 = model.encode(train_data1)
    embeddings2 = model.encode(train_data2)

    difference = []
    
    for e1, e2 in zip(embeddings1, embeddings2):
        difference.append(e2 - e1)

    # cosine_similarities = cosine_similarity(embeddings1, embeddings2)

    # plot_and_compare_emb(embeddings1, embeddings2, args.task + '.png')

    plot_emb(difference, args.task + '_' + args.augmentor + str(AUGMENT_LOOPS) + '.png', args.datapoints)

    




