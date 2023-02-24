import torch, time
import pytorch_lightning as pl
from argparse import ArgumentParser
from ..helpers import set_seed, plot_and_compare_emb
from ..data import TranslationDataModule, AGNewsDataModule, GlueDataModule, TwitterDataModule, BiasDetectionDataModule
from .data_augmentors import Back_Translator
from sentence_transformers import SentenceTransformer

def visualize_back_translation_embedding():
    
    parser = ArgumentParser()

    # add PROGRAM level args
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task", type=str, default="bias_detection")
    parser.add_argument("--deterministic", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    set_seed(args.seed)

    augmentor = Back_Translator("en")

    data_modules = {"glue": GlueDataModule, "twitter": TwitterDataModule, "bias_detection": BiasDetectionDataModule}

    data = data_modules[args.task](
        dataset_percentage = 1,
        augmentors = [],
        batch_size = args.batch_size
    )

    data.prepare_data()
    data.setup("fit")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data1 = list(data.get_dataset_text()).to(device)
    train_data2 = train_data1.copy()

    augmentor.set_augmentation_percentage(1000) # So guaranteed augmentation
    print("Start augmenting!")
    start_time = time.time()
    train_data2 = augmentor.augment_dataset(train_data2).to(device)
    print("Finish augmenting! Time taken: " + str(time.time() - start_time))

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(device)

    embeddings1 = model.encode(train_data1)
    embeddings2 = model.encode(train_data2)

    plot_and_compare_emb(embeddings1, embeddings2, args.task + '.png')




