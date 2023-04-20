import torch, time, random
from torchtext.datasets import AG_NEWS as agnews
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
from torch.utils.tensorboard import SummaryWriter
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, early_stopping
from argparse import ArgumentParser
from ..helpers import EnglishPreProcessor, Logger, parse_augmentors, parse_augmentors_int, set_seed, PyTorchLightningPruningCallback
from .translator import TranslatorModule
from ..data import IWSLT17DataModule, AGNewsDataModule, ColaDataModule, TwitterDataModule, BabeDataModule, IMDBDataModule, TrecDataModule, DBPediaDataModule, FewShotTextClassifyWrapperModule, QNLIDataModule, SST2DataModule
from pytorch_lightning.loggers import TensorBoardLogger
from .text_classifier import TextClassifierModule
from .text_classifier_policies import TextClassifierPolicyModule
from .data_augmentors import AUGMENTOR_LIST_SINGLE, AUGMENTOR_LIST
from pytorch_lightning.plugins.environments import SLURMEnvironment
import signal
import optuna
from pytorch_lightning.callbacks import ModelCheckpoint
import os
from hyperopt import hp
from sklearn.model_selection import StratifiedShuffleSplit
from .train_model import text_classify
from ray import tune
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.integration.pytorch_lightning import TuneReportCallback, \
    TuneReportCheckpointCallback
import numpy as np
from torch.utils.data import DataLoader
from optuna import distributions

def hyper_param_search(args):
    if args.task == 'classify':
        text_classify_search(args)
    elif args.task == 'translate':
        seq2seq_translate_search(args)
    else:
        raise Exception("Unknown Task")

def text_classify_search(args):
    if args.to_search == "lr":
        text_classify_search_lr(args)
    elif args.to_search == "aug":
        text_classify_search_aug(args)
    elif args.to_search == "policy":
        text_classify_search_policy(args)
    else:
        raise Exception("Unknown Search Parameter")

def seq2seq_translate_search(args):
    if args.to_search == "lr":
        seq2seq_translate_search_lr(args)
    elif args.to_search == "aug":
        seq2seq_translate_search_aug(args)
    else:
        raise Exception("Unknown Search Parameter")

def print_trial_stats(study):
    pruned_trials = study.get_trials(deepcopy = False, states=[optuna.trial.TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy = False, states=[optuna.trial.TrialState.COMPLETE])
    print("Study statistics:")
    print(" Number of finished trials: ", len(study.trials))
    print(" Number of pruned tirals: ", len(pruned_trials))
    print(" Number of complete trials: ", len(complete_trials))
    for complete_trial in complete_trials:
        try:
            print(f"Value in trial number {complete_trial.number}: {complete_trial.value:.4f}")
        except:
            print("Value in trial number", str(complete_trial.number) + ":", complete_trial.value)
        print("Hyperparameters in trial number", str(complete_trial.number) + ":")
        try:
            for key, value in complete_trial.params.items():
                print(f"    {key}: {value}")
        except:
            print(complete_trial.params)

    print(f"Best value: {study.best_value:.4f}")
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")

def seq2seq_translate_search_aug(args):
    
    
    def objective(trial: optuna.Trial, args):

        MODEL_NAME = "t5-small"
        augmentation_param_range = {"sr": (0,50), "bt": (0,500), "in": (0,50), "de": (0,50), "co": (0,100), "cm": (0,100), "mu": (0,100)}
        augmentation_params = []
        augmentor_names = list(filter(lambda x: x != "", (args.augmentors.split(","))))
        for name in augmentor_names:
            param_range = augmentation_param_range[name]
            augmentation_params.append(trial.suggest_int(f"{name} augmentation param", param_range[0], param_range[1]))
        word_augmentors, embed_augmentors = parse_augmentors_int(augmentor_names, augmentation_params)
        try:
            learning_rate = float(args.learning_rate)
        except ValueError:
            raise Exception("Learning rate argument should be a float")
        
        data = IWSLT17DataModule(
            model_name = MODEL_NAME,
            dataset_percentage = 1,
            augmentors = word_augmentors,
            batch_size=args.batch_size
        )
        data.prepare_data()
        data.setup("fit")
        filename = "translate_" + args.augmentors + "_data=" + str(args.dataset_percentage) + "_seed=" + str(args.seed) + "_augmentation_params=" + str(augmentation_params)
        
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

        checkpoint_callback = ModelCheckpoint(
            monitor='validation_bleu',
            dirpath=args.logger_dir,
            save_top_k=1,
            save_weights_only=True,
            filename=filename,
            auto_insert_metric_name=False,
        )
        
        early_pruning_callback = PyTorchLightningPruningCallback(trial, monitor="validation_bleu")

        trainer = pl.Trainer.from_argparse_args(
            args, deterministic=True,  logger=logger, replace_sampler_ddp=False, callbacks=[lr_monitor, early_stop_callback, early_pruning_callback, checkpoint_callback], plugins=[SLURMEnvironment(requeue_signal=signal.SIGUSR1)]
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
        test_bleu = trainer.callback_metrics["test_bleu"].item()

        try:
            os.remove(args.logger_dir + "/" + filename + ".ckpt")
        except FileNotFoundError:
            raise Exception("Could not reset checkpoint files across trials.")
        
        return test_bleu

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, args), n_trials = 10, timeout = 129600)
    print_trial_stats(study)


def seq2seq_translate_search_lr(args):

    def objective(trial, args):
        MODEL_NAME = "t5-small"
        learning_rate = trial.suggest_float("learning_rate", 4e-5, 1e-3, log=True)
        data = IWSLT17DataModule(
            model_name = MODEL_NAME,
            dataset_percentage = 1,
            batch_size=args.batch_size
        )
        data.prepare_data()
        data.setup("fit")
        filename = "translate_" + args.augmentors + "_data=" + str(args.dataset_percentage) + "_seed=" + str(args.seed) + "_lr=" + str(learning_rate)

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

        checkpoint_callback = ModelCheckpoint(
            monitor='validation_loss_epoch',
            dirpath=args.logger_dir,
            save_top_k=1,
            save_weights_only=True,
            filename=filename,
            auto_insert_metric_name=False,
        )
        
        early_pruning_callback = PyTorchLightningPruningCallback(trial, monitor="validation_bleu")

        trainer = pl.Trainer.from_argparse_args(
            args, deterministic=True, logger=logger, replace_sampler_ddp=False, callbacks=[lr_monitor, early_stop_callback, early_pruning_callback, checkpoint_callback], plugins=[SLURMEnvironment(requeue_signal=signal.SIGUSR1)]
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
            learning_rate = learning_rate
        ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        # most basic trainer, uses good defaults (1 gpu)
        trainer.fit(model, data)
        trainer.test(model, dataloaders = data.test_dataloader())
        test_bleu = trainer.callback_metrics["test_bleu"].item()
        
        try:
            os.remove(args.logger_dir + "/" + filename + ".ckpt")
        except FileNotFoundError:
            raise Exception("Could not reset checkpoint files across trials.")
        
        return test_bleu

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, args), timeout = 43200)
    print_trial_stats(study)


def text_classify_search_aug(args):
    
    def objective(trial, args):
        augmentation_param_range = {"sr": (0,50), "bt": (0,500), "in": (0,50), "de": (0,50), "co": (0,100), "cm": (0,100), "mu": (0,100)}
        try:
            learning_rate = float(args.learning_rate)
        except ValueError:
            raise Exception("Learning rate argument should be a float")
        augmentation_params = []
        augmentor_names = list(filter(lambda x: x != "", (args.augmentors.split(","))))
        for name in augmentor_names:
            param_range = augmentation_param_range[name]
            augmentation_params.append(trial.suggest_int(f"{name} augmentation param", param_range[0], param_range[1]))
        word_augmentors, embed_augmentors = parse_augmentors_int(augmentor_names, augmentation_params)
        data_modules = {"cola": ColaDataModule, "twitter": TwitterDataModule, "babe": BabeDataModule, "ag_news": AGNewsDataModule, "imdb": IMDBDataModule, "trec": TrecDataModule, "dbpedia": DBPediaDataModule, "qnli": QNLIDataModule, "sst2": SST2DataModule}
        
        if args.samples_per_class is not None:
            args.dataset_percentage = 100

        data = data_modules[args.dataset](
            dataset_percentage = 1,
            augmentors = word_augmentors,
            batch_size = args.batch_size
        )
        data.setup("fit")

        filename = str(args.dataset) + "_" + args.augmentors + "_data=" + str(args.dataset_percentage) + "_seed=" + str(args.seed) + "_augmentation_params=" +  str(augmentation_params)
        
        logger = TensorBoardLogger(
            args.logger_dir, name=filename
        )


        checkpoint_callback = ModelCheckpoint(
            monitor='validation_accuracy',
            dirpath=args.logger_dir,
            save_top_k=1,
            save_weights_only=True,
            filename=filename,
            auto_insert_metric_name=False,
        )

        lr_monitor = LearningRateMonitor(logging_interval="step")
        early_stop_callback = early_stopping.EarlyStopping(
            monitor='validation_loss_epoch',
            min_delta=0,
            patience=3,
            mode='min',
        )
        early_pruning_callback = PyTorchLightningPruningCallback(trial, monitor="validation_accuracy")

        plugins = []

        trainer = pl.Trainer.from_argparse_args(
            args, deterministic=True, logger=logger, replace_sampler_ddp=False, callbacks=[lr_monitor, early_stop_callback, early_pruning_callback, checkpoint_callback], plugins=plugins
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
        test_accuracy = trainer.callback_metrics["test_accuracy"].item()

        
        try:
            os.remove(args.logger_dir + "/" + filename + ".ckpt")
        except FileNotFoundError:
            raise Exception("Could not reset checkpoint files across trials.")
        
        return test_accuracy

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, args), n_trials = 10, timeout = 129600)

    print_trial_stats(study)

def text_classify_search_lr(args):

    def objective(trial, args):
        lr = trial.suggest_float("learning_rate", 4e-5, 1e-2, log=True)
        data_modules = {"cola": ColaDataModule, "twitter": TwitterDataModule, "babe": BabeDataModule, "ag_news": AGNewsDataModule, "imdb": IMDBDataModule, "trec": TrecDataModule, "dbpedia": DBPediaDataModule}
        
        if args.samples_per_class is not None:
            args.dataset_percentage = 100

        data = data_modules[args.dataset](
            dataset_percentage = args.dataset_percentage / 100,
            batch_size = args.batch_size
        )

        if args.samples_per_class is not None:
            data = FewShotTextClassifyWrapperModule(data, args.samples_per_class)

        data.prepare_data()
        data.setup("fit")
        filename = args.task + "_" + args.augmentors + "_data=" + str(args.dataset_percentage) + "seed=" + str(args.seed) + "lr=" + str(lr)
        logger = TensorBoardLogger(
            args.logger_dir, name=filename
        )

        checkpoint_callback = ModelCheckpoint(
            monitor='validation_accuracy',
            dirpath=args.logger_dir,
            save_top_k=1,
            save_weights_only=True,
            filename=filename,
            auto_insert_metric_name=False,
        )

        lr_monitor = LearningRateMonitor(logging_interval="step")
        early_stop_callback = early_stopping.EarlyStopping(
            monitor='validation_loss_epoch',
            min_delta=0,
            patience=3,
            mode='min',
        )
        early_pruning_callback = PyTorchLightningPruningCallback(trial, monitor="validation_accuracy")

        trainer = pl.Trainer.from_argparse_args(
            args, deterministic=True, logger=logger, replace_sampler_ddp=False, callbacks=[lr_monitor, early_stop_callback, early_pruning_callback, checkpoint_callback]
        )  # , distributed_backend='ddp_cpu')
        
        # for batch_idx, batch in enumerate(data.split_and_pad_data(data.dataset['train'])):
        #     input_, output = batch
        #     print(input_['src_len'])    

        # id2label = {0: "WORLD", 1: "SPORTS", 2: "BUSINESS", 3: "SCIENCE"}
        # label2id = {"WORLD": 0, "SPORTS": 1, "BUSINESS": 2, "SCIENCE": 3}

        model = TextClassifierModule(
            learning_rate = lr,
            max_epochs = args.max_epochs,
            tokenizer = data.tokenizer,
            steps_per_epoch = int(len(data.train_dataloader())),
            num_labels = len(data.id2label),
            id2label = data.id2label,
            label2id = data.label2id,
            pretrain = args.pretrain,
            word_augmentors = [],
            embed_augmentors = []
        ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        # most basic trainer, uses good defaults (1 gpu)
        trainer.fit(model, data)
        trainer.test(model, dataloaders = data.test_dataloader())
        test_accuracy = trainer.callback_metrics["test_accuracy"].item()
        return test_accuracy

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, args), n_trials = 50, timeout = 14400)
    print_trial_stats(study)

def text_classify_search_policy(args):
    data_modules = {"cola": ColaDataModule, "twitter": TwitterDataModule, "babe": BabeDataModule, "ag_news": AGNewsDataModule, "imdb": IMDBDataModule, "trec": TrecDataModule, "dbpedia": DBPediaDataModule, "qnli": QNLIDataModule, "sst2": SST2DataModule}


    final_policies = []

    data = data_modules[args.dataset](
        dataset_percentage = args.dataset_percentage,
        augmentors = [],
        batch_size = args.batch_size
    )
    data.setup("fit")

    n_splits = args.n_splits

    train_samples, train_labels = data.format_data(data.train_dataset)
    valid_samples, valid_labels = data.format_data(data.valid_dataset)

    train_samples = np.array(train_samples)
    valid_samples = np.array(valid_samples)

    train_samples = np.concatenate((train_samples, valid_samples))
    train_labels = np.concatenate((train_labels, valid_labels))

    reward_attr = 'valid_loss'

    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.1, random_state=args.seed)
    for i, (train_index, test_index) in enumerate(sss.split(train_samples, train_labels)):

        splitted_train = train_samples[train_index]
        splitted_train_labels = train_labels[train_index]
        test = train_samples[test_index]
        test_labels = train_labels[test_index]

        train_model_split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=args.seed)
        for train_index, test_index in train_model_split.split(splitted_train, splitted_train_labels):
            model_train, model_train_labels = splitted_train[train_index], splitted_train_labels[train_index]
            model_valid, model_valid_labels = splitted_train[test_index], splitted_train_labels[test_index]
        
        train_dataloader = DataLoader(data.split_and_tokenize((model_train, model_train_labels), format = False) , batch_size=data.batch_size, shuffle = True)
        valid_dataloader = DataLoader(data.split_and_tokenize((model_valid, model_valid_labels), format = False), batch_size=data.batch_size)
        test_dataloader = DataLoader(data.split_and_tokenize((test, test_labels), format = False), batch_size=data.batch_size)

        try:
            learning_rate = float(args.learning_rate)
        except ValueError:
            raise Exception("Learning rate argument should be a float")
        
        data_modules = {"cola": ColaDataModule, "qnli": QNLIDataModule, "sst2": SST2DataModule, "twitter": TwitterDataModule, "babe": BabeDataModule, "ag_news": AGNewsDataModule, "imdb": IMDBDataModule, "trec": TrecDataModule, "dbpedia": DBPediaDataModule}

        if args.samples_per_class is not None:
            args.dataset_percentage = 100

        filename = args.task + "_" + args.augmentors + "_data=" + str(args.dataset_percentage) + "seed=" + str(args.seed) + "_fast_aa_search_" + str(i)

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

        model = TextClassifierPolicyModule(
            learning_rate = learning_rate,
            max_epochs = args.max_epochs,
            tokenizer = data.tokenizer,
            steps_per_epoch = int(len(data.train_dataloader())),
            num_labels = len(data.id2label),
            id2label = data.id2label,
            label2id = data.label2id,
            pretrain = args.pretrain
        ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        # most basic trainer, uses good defaults (1 gpu)
        trainer.fit(model, train_dataloader = train_dataloader, valid_dataloader = valid_dataloader)

        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: train_and_eval(trial, args, test_dataloader, model, trainer), n_trials = 1, timeout = 30000)
        for key, value in study.best_params.items():
            print(f"    {key}: {value}")
        
        import pdb
        pdb.set_trace()





    def train_and_eval(trial, args, dataloader, model, trainer):
        policies = suggest_policies(trial, args)
        
        validation_policy = suggest_policies(trial, args) 
        model.set_validation_policy(validation_policy)

        trainer.validate(model, dataloader)

        return trainer.callback_metrics['validation_loss_epoch']

    def suggest_policies(trial, args):
        policies = []

        for i in range(args.num_policy):
            augmentors = []
            for j in range(args.num_op):
                # Sample from a categorical distribution that represents a possible augmentation method
                aug_dist = distributions.CategoricalDistribution(AUGMENTOR_LIST)

                augmentor = trial.suggest_categorical(f"augmentor_{i}", aug_dist)

                lam = trial.suggest_float(f"{str(augmentor)}_prob", distributions.UniformDistribution(0, 1))
                
                # Append the selected augmentation method and its associated probability to the list
                augmentor.set_augmentation_percentage(lam)
                augmentors.append(augmentor)
            
            policies.append(augmentors)
            
        return policies
        

            


        





