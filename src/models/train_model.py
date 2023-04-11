import torch, time, random
from torchtext.datasets import AG_NEWS as agnews
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
from torch.utils.tensorboard import SummaryWriter
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, early_stopping, ModelCheckpoint
from ..helpers import EnglishPreProcessor, Logger, parse_augmentors, set_seed, plot_saliency_scores
from .seq2seq_translator import TranslatorModule
from ..data import IWSLT17DataModule, AGNewsDataModule, ColaDataModule, MNLIDataModule, SST2DataModule, TwitterDataModule, BabeDataModule, IMDBDataModule, TrecDataModule, DBPediaDataModule, FewShotTextClassifyWrapperModule, WikiText2DataModule
from pytorch_lightning.loggers import TensorBoardLogger
from .text_classifier import TextClassifierModule
from .text_classifier_with_saliency import TextClassifierSaliencyModule
from .language_model import LanguageModelModule
import signal, os
import statistics

def train_model(args):
    if args.task == 'classify':
        text_classify(args)
    elif args.task == 'classify_saliency':
        text_classify_with_saliency(args)
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
    
    data = IWSLT17DataModule(
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
        args, deterministic=True, logger=logger, replace_sampler_ddp=False, callbacks=[lr_monitor, early_stop_callback, checkpoint_callback] #, plugins=[SLURMEnvironment(requeue_signal=signal.SIGUSR1)]
    )  # , distributed_backend='ddp_cpu')
    
    # for batch_idx, batch in enumerate(data.split_and_pad_data(data.dataset['train'])):
    #     input_, output = batch
    #     print(input_['src_len'])
    
    model = TranslatorModule(
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
    
    data_modules = {"cola": ColaDataModule, "mnli": MNLIDataModule, "sst2": SST2DataModule, "twitter": TwitterDataModule, "babe": BabeDataModule, "ag_news": AGNewsDataModule, "imdb": IMDBDataModule, "trec": TrecDataModule, "dbpedia": DBPediaDataModule}

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
        args, deterministic=True, logger=logger, replace_sampler_ddp=False, callbacks=[lr_monitor, early_stop_callback, checkpoint_callback]
    )  # , distributed_backend='ddp_cpu')
    
    # for batch_idx, batch in enumerate(data.split_and_pad_data(data.dataset['train'])):
    #     input_, output = batch
    #     print(input_['src_len'])    

    # id2label = {0: "WORLD", 1: "SPORTS", 2: "BUSINESS", 3: "SCIENCE"}
    # label2id = {"WORLD": 0, "SPORTS": 1, "BUSINESS": 2, "SCIENCE": 3}

    model = TextClassifierModule(
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

    with open('results_' + args.dataset + '.txt', 'a') as f:
        f.write("Seed:", args.seed)
        if args.samples_per_class is not None:
            f.write("FewShot Training Used. Samples per class:", args.samples_per_class)
        else:
            f.write("Dataset Percentage:", args.dataset_percentage)
        f.write("Test accuracy", trainer.callback_metrics['test_accuracy'])
        f.write("\n\n\n")

def text_classify_with_saliency(args):
    word_augmentors, embed_augmentors = parse_augmentors(args)
    try:
        learning_rate = float(args.learning_rate)
    except ValueError:
        raise Exception("Learning rate argument should be a float")
    

    data_modules = {"cola": ColaDataModule, "mnli": MNLIDataModule, "sst2": SST2DataModule,  "twitter": TwitterDataModule, "babe": BabeDataModule, "ag_news": AGNewsDataModule, "imdb": IMDBDataModule, "trec": TrecDataModule, "dbpedia": DBPediaDataModule}

    if args.samples_per_class is not None:
        args.dataset_percentage = 100

    data = data_modules[args.dataset](
        dataset_percentage = args.dataset_percentage / 100,
        batch_size = args.batch_size,
        tokenize = False
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
        args, deterministic=True, logger=logger, replace_sampler_ddp=False, callbacks=[lr_monitor, early_stop_callback, checkpoint_callback]
    )  # , distributed_backend='ddp_cpu')
    
    # for batch_idx, batch in enumerate(data.split_and_pad_data(data.dataset['train'])):
    #     input_, output = batch
    #     print(input_['src_len'])    

    # id2label = {0: "WORLD", 1: "SPORTS", 2: "BUSINESS", 3: "SCIENCE"}
    # label2id = {"WORLD": 0, "SPORTS": 1, "BUSINESS": 2, "SCIENCE": 3}

    model = TextClassifierSaliencyModule(
        learning_rate = learning_rate,
        max_epochs = args.max_epochs,
        tokenizer = data.tokenizer,
        steps_per_epoch = int(len(data.train_dataloader())),
        num_labels = len(data.id2label),
        id2label = data.id2label,
        label2id = data.label2id,
        pretrain = args.pretrain,
        word_augmentors = word_augmentors,
        embed_augmentors = embed_augmentors
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
    
    saliency_scores_per_word = model.saliency_scores_per_word
    mean_saliency = {key: statistics.mean(values) for key, values in saliency_scores_per_word.items() if len(values) > 10}
    highest_means = sorted(mean_saliency.items(), key=lambda x: x[1], reverse=True)[:10]
    lowest_means = sorted(mean_saliency.items(), key=lambda x: x[1])[:10]
    print('Top 10 highest means:', highest_means)
    print('Top 10 lowest means:', lowest_means)

    def batch(iterable, n=1):
        """Batch elements of an iterable into lists of size n."""
        it = iter(iterable)
        while True:
            chunk = []
            for i in range(n):
                try:
                    chunk.append(next(it))
                except StopIteration:
                    break
            if chunk:
                yield chunk
            else:
                return

    saliency_scores = model.saliency_scores
    keys = list(saliency_scores.keys())
    batch_size = 1000
    fig_count = 0
    for i, key_batch in enumerate(batch(keys, batch_size)):
        score_batch = [saliency_scores[key] for key in key_batch]
        for j in range(len(key_batch)):
            words = key_batch[j]
            scores = score_batch[j]
            if fig_count < 100:
                plot_saliency_scores(words, scores, f"saliency_fig_{i*batch_size+j}.png")
                fig_count += 1

    saliency_scores_per_word = model.saliency_scores_per_word
    batch_size = 1000
    highest = []
    lowest = []
    for i, items in enumerate(batch(saliency_scores_per_word.items(), batch_size)):
        mean_saliency = {key: statistics.mean(value) for key, value in items if len(value) > 10}
        highest_means = sorted(mean_saliency.items(), key=lambda x: x[1], reverse=True)[:10]
        lowest_means = sorted(mean_saliency.items(), key=lambda x: x[1])[:10]
        highest += highest_means
        lowest += lowest_means

    highest_means = sorted(highest, key=lambda x: x[1], reverse=True)[:10]
    lowest_means = sorted(lowest, key=lambda x: x[1])[:10]
    print('Top 10 highest means:', highest_means)
    print('Top 10 lowest means:', lowest_means)


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
        os.remove("runs_language_model/" + filename + ".ckpt")
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
            monitor='validation_perplexity',
            dirpath='runs_language_model',
            save_top_k=1,
            save_weights_only=True,
            filename=filename,
            auto_insert_metric_name=False
        )

    trainer = pl.Trainer.from_argparse_args(
        args, deterministic=True, logger=logger, replace_sampler_ddp=False, callbacks=[lr_monitor, early_stop_callback, checkpoint_callback]
    )  # , distributed_backend='ddp_cpu')
    
    # for batch_idx, batch in enumerate(data.split_and_pad_data(data.dataset['train'])):
    #     input_, output = batch
    #     print(input_['src_len'])    

    # id2label = {0: "WORLD", 1: "SPORTS", 2: "BUSINESS", 3: "SCIENCE"}
    # label2id = {"WORLD": 0, "SPORTS": 1, "BUSINESS": 2, "SCIENCE": 3}

    model = LanguageModelModule(
        learning_rate = learning_rate,
        max_epochs = args.max_epochs,
        tokenizer = data.tokenizer,
        steps_per_epoch = int(len(data.train_dataloader())),
        augmentors = embed_augmentors
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    # most basic trainer, uses good defaults (1 gpu)
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