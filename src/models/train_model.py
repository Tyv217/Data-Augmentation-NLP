import torch, time, random
from torchtext.datasets import AG_NEWS as agnews
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
from torch.utils.tensorboard import SummaryWriter
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, early_stopping, ModelCheckpoint
from ..helpers import EnglishPreProcessor, Logger, parse_augmentors, set_seed, plot_saliency_scores
from .seq2seq_translator import Seq2SeqTranslator
from ..data import TranslationDataModule, AGNewsDataModule, GlueDataModule, TwitterDataModule, BabeDataModule, IMDBDataModule, TrecDataModule, DBPediaDataModule, FewShotTextClassifyWrapperModule, WikiText2DataModule
from pytorch_lightning.loggers import TensorBoardLogger
from .text_classifier import Text_Classifier
from .text_classifier_with_saliency import Text_Classifier_With_Saliency
import signal, os

def train_model(args):
    if args.task == 'classify':
        text_classify(args)
    elif args.task == 'translate':
        seq2seq_translate(args)
    elif args.task == 'language_model':
        language_model(args)
    else:
        raise Exception("Unknown Task")

def seq2seq_translate(args):
    MODEL_NAME = "t5-small"
    word_augmentors, embed_augmentors = parse_augmentors(args)
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
    
    logger = TensorBoardLogger(
        args.logger_dir, name=filename
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
        dirpath=args.logger_dir,
        save_last=True,
        save_top_k=1,
        save_weights_only=True,
        filename=filename,
        auto_insert_metric_name=False,
    )

    trainer = pl.Trainer.from_argparse_args(
        args, logger=logger, replace_sampler_ddp=False, callbacks=[lr_monitor, early_stop_callback, checkpoint_callback] #, plugins=[SLURMEnvironment(requeue_signal=signal.SIGUSR1)]
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
        
    try:
        os.remove(args.logger_dir + "/" + filename + ".ckpt")
    except FileNotFoundError:
        pass
    
    print("Seed:", args.seed)
    print("Augmentors:", args.augmentors)
    print("Augmentation params:", args.augmentation_params)
    print("Dataset Percentage:", args.dataset_percentage)

def text_classify(args):
    word_augmentors, embed_augmentors = parse_augmentors(args)
    try:
        learning_rate = float(args.learning_rate)
    except ValueError:
        raise Exception("Learning rate argument should be a float")
    
    data_modules = {"glue": GlueDataModule, "twitter": TwitterDataModule, "babe": BabeDataModule, "ag_news": AGNewsDataModule, "imdb": IMDBDataModule, "trec": TrecDataModule, "dbpedia": DBPediaDataModule}

    if args.samples_per_class is not None:
        args.dataset_percentage = 100

    data = data_modules[args.dataset](
        dataset_percentage = args.dataset_percentage / 100,
        augmentors = word_augmentors,
        batch_size = args.batch_size
    )

    if args.samples_per_class is not None:
        data = FewShotTextClassifyWrapperModule(data, args.samples_per_class)

    data.prepare_data()
    data.setup("fit")

    filename = args.task + "_" + args.augmentors + "_data=" + str(args.dataset_percentage) + "seed=" + str(args.seed)

    logger = TensorBoardLogger(
        args.logger_dir, name=filename
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
            monitor='validation_accuracy',
            dirpath=args.logger_dir,
            save_top_k=1,
            save_weights_only=True,
            filename=filename,
            auto_insert_metric_name=False
        )

    trainer = pl.Trainer.from_argparse_args(
        args, logger=logger, replace_sampler_ddp=False, callbacks=[lr_monitor, early_stop_callback, checkpoint_callback]
    )  # , distributed_backend='ddp_cpu')
    
    # for batch_idx, batch in enumerate(data.split_and_pad_data(data.dataset['train'])):
    #     input_, output = batch
    #     print(input_['src_len'])    

    # id2label = {0: "WORLD", 1: "SPORTS", 2: "BUSINESS", 3: "SCIENCE"}
    # label2id = {"WORLD": 0, "SPORTS": 1, "BUSINESS": 2, "SCIENCE": 3}

    model = Text_Classifier(
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
    trainer.fit(model, data)
    trainer.test(model, dataloaders = data.test_dataloader())
     
    try:
        os.remove(args.logger_dir + "/" + filename + ".ckpt")
    except FileNotFoundError:
        pass
    
    print("Seed:", args.seed)
    print("Augmentors:", args.augmentors)
    print("Augmentation params:", args.augmentation_params)
    if args.samples_per_class is not None:
        print("FewShot Training Used. Samples per class:", args.samples_per_class)
    else:
        print("Dataset Percentage:", args.dataset_percentage)

def text_classify_with_saliency(args):
    word_augmentors, embed_augmentors = parse_augmentors(args)
    try:
        learning_rate = float(args.learning_rate)
    except ValueError:
        raise Exception("Learning rate argument should be a float")
    
    data_modules = {"glue": GlueDataModule, "twitter": TwitterDataModule, "babe": BabeDataModule, "ag_news": AGNewsDataModule, "imdb": IMDBDataModule, "trec": TrecDataModule, "dbpedia": DBPediaDataModule}

    if args.samples_per_class is not None:
        args.dataset_percentage = 100

    data = data_modules[args.dataset](
        dataset_percentage = args.dataset_percentage / 100,
        augmentors = word_augmentors,
        batch_size = args.batch_size
    )

    if args.samples_per_class is not None:
        data = FewShotTextClassifyWrapperModule(data, args.samples_per_class)

    data.prepare_data()
    data.setup("fit")

    filename = args.task + "_" + args.augmentors + "_data=" + str(args.dataset_percentage) + "seed=" + str(args.seed)

    logger = TensorBoardLogger(
        args.logger_dir, name=filename
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
            monitor='validation_accuracy',
            dirpath=args.logger_dir,
            save_top_k=1,
            save_weights_only=True,
            filename=filename,
            auto_insert_metric_name=False
        )

    trainer = pl.Trainer.from_argparse_args(
        args, logger=logger, replace_sampler_ddp=False, callbacks=[lr_monitor, early_stop_callback, checkpoint_callback]
    )  # , distributed_backend='ddp_cpu')
    
    # for batch_idx, batch in enumerate(data.split_and_pad_data(data.dataset['train'])):
    #     input_, output = batch
    #     print(input_['src_len'])    

    # id2label = {0: "WORLD", 1: "SPORTS", 2: "BUSINESS", 3: "SCIENCE"}
    # label2id = {"WORLD": 0, "SPORTS": 1, "BUSINESS": 2, "SCIENCE": 3}

    model = Text_Classifier(
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
    trainer.fit(model, data)
    trainer.test(model, dataloaders = data.test_dataloader())
     
    try:
        os.remove(args.logger_dir + "/" + filename + ".ckpt")
    except FileNotFoundError:
        pass
    
    print("Seed:", args.seed)
    print("Augmentors:", args.augmentors)
    print("Augmentation params:", args.augmentation_params)
    if args.samples_per_class is not None:
        print("FewShot Training Used. Samples per class:", args.samples_per_class)
    else:
        print("Dataset Percentage:", args.dataset_percentage)
    
    saliency_scores = model.saliency_scores
    keys = list(saliency_scores.keys())
    for i in range(len(keys)):
        words = keys[i]
        scores = saliency_scores[keys[i]]
        plot_saliency_scores(words, scores, "saliency_fig_" + str(i) + ".png")

def language_model(args):
    
    set_seed(args.seed)
    word_augmentors, embed_augmentors = parse_augmentors(args)
    try:
        learning_rate = float(args.learning_rate)
    except ValueError:
        raise Exception("Learning rate argument should be a float")
    
    if args.samples_per_class is not None:
        args.dataset_percentage = 100

    data = WikiText2DataModule(
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
        monitor='validation_loss_epoch',
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
            auto_insert_metric_name=False
        )

    trainer = pl.Trainer.from_argparse_args(
        args, logger=logger, replace_sampler_ddp=False, callbacks=[lr_monitor, early_stop_callback, checkpoint_callback]
    )  # , distributed_backend='ddp_cpu')
    
    # for batch_idx, batch in enumerate(data.split_and_pad_data(data.dataset['train'])):
    #     input_, output = batch
    #     print(input_['src_len'])    

    # id2label = {0: "WORLD", 1: "SPORTS", 2: "BUSINESS", 3: "SCIENCE"}
    # label2id = {"WORLD": 0, "SPORTS": 1, "BUSINESS": 2, "SCIENCE": 3}

    model = Text_Classifier(
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