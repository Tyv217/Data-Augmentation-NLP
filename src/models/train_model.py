import torch, time, random
from torchtext.datasets import AG_NEWS as agnews
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
from torch.utils.tensorboard import SummaryWriter
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, early_stopping, ModelCheckpoint
from argparse import ArgumentParser
from ..helpers import EnglishPreProcessor, Logger, parse_augmentors, set_seed
from .text_classifier import TextClassifierEmbeddingModel
from .seq2seq_translator import Seq2SeqTranslator
from ..data import TranslationDataModule, AGNewsDataModule, GlueDataModule, TwitterDataModule, BiasDetectionDataModule, IMDBDataModule, TrecDataModule, DBPediaDataModule, FewShotTextClassifyWrapperModule
from pytorch_lightning.loggers import TensorBoardLogger
from .better_text_classifier import Better_Text_Classifier
from .data_augmentors import Synonym_Replacer, Back_Translator, Insertor, Deletor, CutOut, CutMix, MixUp
from pytorch_lightning.plugins.environments import SLURMEnvironment
import signal, os

def seq2seq_translate():
    MODEL_NAME = "t5-small"
    parser = ArgumentParser(conflict_handler = 'resolve')

    # add PROGRAM level args
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning_rate", type=str, default="5e-5")
    parser.add_argument("--augmentors", type=str, default="")
    parser.add_argument("--dataset_percentage", type=int, default=100)
    parser.add_argument("--augmentation_params", type=str, default="")
    parser.add_argument("--N_samples", type=int, default=256 * 10)
    parser.add_argument("--N_valid_size", type=int, default=32 * 10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--embed_size", type=int, default=32)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--deterministic", type=bool, default=True)
    parser.add_argument("--pretrain", default=True, action="store_false")
    parser.add_argument("--no_pretrain",  dest='pretrain', action="store_false")
    parser.set_defaults(pretrain=False)
    # parser.add_argument("--deterministic", type=bool, default=True)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    set_seed(args.seed)
    args.task = "translate"
    augmentator_mapping = {"sr": Synonym_Replacer("english"), "bt": Back_Translator("en"), "in": Insertor("english"), "de": Deletor(), "co": CutOut(), "cm": CutMix(), "mu": MixUp()}
    word_augmentors, embed_augmentors = parse_augmentors(args, augmentator_mapping)
    try:
        learning_rate = float(args.learning_rate)
    except ValueError:
        raise Exception("Learning rate argument should be a float")
    data = TranslationDataModule(
        model_name = MODEL_NAME,
        dataset_percentage = args.dataset_percentage / 100,
        augmentors = word_augmentors,
        batch_size=args.batch_size
    )

    data.prepare_data()
    data.setup("fit")
    filename = "translate_" + args.augmentors + "_data=" + str(args.dataset_percentage) + "_seed=" + str(args.seed)
    
    try:
        os.remove("runs_translate/" + filename + ".ckpt")
    except FileNotFoundError:
        pass
    
    logger = TensorBoardLogger(
        "runs_translate", name=filename
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    early_stop_callback = early_stopping.EarlyStopping(
        monitor='validation_loss_epoch',
        min_delta=0,
        patience=3,
        mode='min',
    )
    print(args)

    checkpoint_callback = ModelCheckpoint(
        monitor='validation_bleu',
        dirpath='runs_translate',
        save_last=True,
        save_top_k=1,
        save_weights_only=True,
        filename=filename,
        auto_insert_metric_name=False,
    )

    trainer = pl.Trainer.from_argparse_args(
        args, logger=logger, replace_sampler_ddp=False, callbacks=[lr_monitor, early_stop_callback, checkpoint_callback] , plugins=[SLURMEnvironment(requeue_signal=signal.SIGUSR1)]
    )  # , distributed_backend='ddp_cpu')
    
    # for batch_idx, batch in enumerate(data.split_and_pad_data(data.dataset['train'])):
    #     input_, output = batch
    #     print(input_['src_len'])
    
    model = Seq2SeqTranslator(
        model_name = MODEL_NAME,
        max_epochs = args.max_epochs,
        tokenizer = data.tokenizer,
        steps_per_epoch = int(len(data.train_dataloader())),
        pretrain = args.pretrain,
        augmentors = embed_augmentors,
        learning_rate = learning_rate
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # most basic trainer, uses good defaults (1 gpu)
    trainer.fit(model, data)
    trainer.test(model, dataloaders = data.test_dataloader())
    
    print("Seed:", args.seed)
    print("Augmentors:", args.augmentors)
    print("Augmentation params:", args.augmentation_params)
    print("Dataset Percentage:", args.dataset_percentage)

def better_text_classify():
    parser = ArgumentParser(conflict_handler = 'resolve')

    # add PROGRAM level args
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning_rate", type=str, default="5e-5")
    parser.add_argument("--task", type=str, default="bias_detection")
    parser.add_argument("--augmentors", type=str, default="")
    parser.add_argument("--augmentation_params", type=str, default="")
    parser.add_argument("--dataset_percentage", type=int, default=100)
    parser.add_argument("--N_samples", type=int, default=256 * 10)
    parser.add_argument("--N_valid_size", type=int, default=32 * 10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--embed_size", type=int, default=32)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--deterministic", type=bool, default=True)
    parser.add_argument("--train", default=True)
    parser.add_argument("--no_train",  dest='train', action="store_false")
    parser.set_defaults(train=True)
    parser.add_argument("--pretrain", default=True, action="store_false")
    parser.add_argument("--no_pretrain",  dest='pretrain', action="store_false")
    parser.set_defaults(pretrain=True)
    parser.add_argument("--samples_per_class", type=int)
    args = parser.parse_args()
    set_seed(args.seed)
    augmentator_mapping = {"sr": Synonym_Replacer("english"), "bt": Back_Translator("en"), "in": Insertor("english"), "de": Deletor(), "co": CutOut(), "cm": CutMix(), "mu": MixUp()}
    word_augmentors, embed_augmentors = parse_augmentors(args, augmentator_mapping)
    try:
        learning_rate = float(args.learning_rate)
    except ValueError:
        raise Exception("Learning rate argument should be a float")
    data_modules = {"glue": GlueDataModule, "twitter": TwitterDataModule, "bias_detection": BiasDetectionDataModule, "ag_news": AGNewsDataModule, "imdb": IMDBDataModule, "trec": TrecDataModule, "dbpedia": DBPediaDataModule}

    if args.samples_per_class is not None:
        args.dataset_percentage = 100

    data = data_modules[args.task](
        dataset_percentage = args.dataset_percentage / 100,
        augmentors = word_augmentors,
        batch_size = args.batch_size
    )

    if args.samples_per_class is not None:
        data = FewShotTextClassifyWrapperModule(data, args.samples_per_class)

    data.prepare_data()
    data.setup("fit")

    filename = args.task + "_" + args.augmentors + "_data=" + str(args.dataset_percentage) + "seed=" + str(args.seed)

    try:
        os.remove("runs_better_text_classify/" + filename + ".ckpt")
    except FileNotFoundError:
        pass

    logger = TensorBoardLogger(
        "runs_better_text_classify", name=filename
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    early_stop_callback = early_stopping.EarlyStopping(
        monitor='validation_loss',
        min_delta=0,
        patience=3,
        mode='min',
    )
    print(args)

    checkpoint_callback = ModelCheckpoint(
            monitor='validation_accuracy',
            dirpath='runs_better_text_classify',
            save_top_k=1,
            save_weights_only=True,
            filename=filename,
            auto_insert_metric_name=False,
            reset_on_train_end=True  # Reset the callback between trials
        )

    trainer = pl.Trainer.from_argparse_args(
        args, logger=logger, replace_sampler_ddp=False, callbacks=[lr_monitor, early_stop_callback, checkpoint_callback]
    )  # , distributed_backend='ddp_cpu')
    
    # for batch_idx, batch in enumerate(data.split_and_pad_data(data.dataset['train'])):
    #     input_, output = batch
    #     print(input_['src_len'])    

    # id2label = {0: "WORLD", 1: "SPORTS", 2: "BUSINESS", 3: "SCIENCE"}
    # label2id = {"WORLD": 0, "SPORTS": 1, "BUSINESS": 2, "SCIENCE": 3}

    model = Better_Text_Classifier(
        learning_rate = learning_rate,
        max_epochs = args.max_epochs,
        tokenizer = data.tokenizer,
        steps_per_epoch = int(len(data.train_dataloader())),
        num_labels = len(data.id2label),
        id2label = data.id2label,
        label2id = data.label2id,
        pretrain = args.pretrain,
        augmentors = embed_augmentors
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    # most basic trainer, uses good defaults (1 gpu)
    if args.train:
        trainer.fit(model, data)
    trainer.test(model, dataloaders = data.test_dataloader())

    print("Seed:", args.seed)
    print("Augmentors:", args.augmentors)
    print("Augmentation params:", args.augmentation_params)
    if args.samples_per_class is not None:
        print("FewShot Training Used. Samples per class:", args.samples_per_class)
    else:
        print("Dataset Percentage:", args.dataset_percentage)

    print("Auto LR Finder Used:", args.auto_lr_find)