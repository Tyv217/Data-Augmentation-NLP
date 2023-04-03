import torch, time, random
from torchtext.datasets import AG_NEWS as agnews
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
from torch.utils.tensorboard import SummaryWriter
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, early_stopping
from argparse import ArgumentParser
from ..helpers import EnglishPreProcessor, Logger, parse_augmentors, parse_augmentors_int, set_seed, PyTorchLightningPruningCallback
from .text_classifier import TextClassifierEmbeddingModel
from .seq2seq_translator import Seq2SeqTranslator
from ..data import TranslationDataModule, AGNewsDataModule, GlueDataModule, TwitterDataModule, BiasDetectionDataModule, IMDBDataModule, TrecDataModule, DBPediaDataModule, FewShotTextClassifyWrapperModule
from pytorch_lightning.loggers import TensorBoardLogger
from .better_text_classifier import Better_Text_Classifier
from .data_augmentors import Synonym_Replacer, Back_Translator, Insertor, Deletor, CutOut, CutMix, MixUp
from pytorch_lightning.plugins.environments import SLURMEnvironment
import signal
import optuna
from pytorch_lightning.callbacks import ModelCheckpoint
import os


def seq2seq_translate_search_aug():
    
    
    def objective(trial: optuna.Trial):

        MODEL_NAME = "t5-small"

        # add PROGRAM level args
        parser = ArgumentParser(conflict_handler = 'resolve')

        # add PROGRAM level args
        parser = pl.Trainer.add_argparse_args(parser)
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--task", type=str, default="translate")
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
        parser.add_argument("--use_high_lr", type=bool, default=False)
        parser.add_argument("--pretrain", action="store_true")
        parser.add_argument("--no_pretrain",  dest='pretrain', action="store_false")
        parser.set_defaults(pretrain=False)
        # parser.add_argument("--deterministic", type=bool, default=True)
        parser = pl.Trainer.add_argparse_args(parser)
        arguments = parser.parse_args()
        set_seed(arguments.seed)
        augmentator_mapping = {"sr": Synonym_Replacer("english"), "bt": Back_Translator("en"), "in": Insertor("english"), "de": Deletor(), "co": CutOut(), "cm": CutMix()}
        augmentation_param_range = {"sr": (0,50), "bt": (0,500), "in": (0,50), "de": (0,50), "co": (0,100), "cm": (0,100), "mu": (0,100)}
        augmentation_params = []
        augmentor_names = list(filter(lambda x: x != "", (arguments.augmentors.split(","))))
        for name in augmentor_names:
            param_range = augmentation_param_range[name]
            augmentation_params.append(trial.suggest_int(f"{name} augmentation param", param_range[0], param_range[1]))
        word_augmentors, embed_augmentors = parse_augmentors(arguments, augmentator_mapping)
        data = TranslationDataModule(
            model_name = MODEL_NAME,
            dataset_percentage = 1,
            augmentors = word_augmentors,
            batch_size=arguments.batch_size
        )
        data.prepare_data()
        data.setup("fit")
        filename = "translate_" + arguments.augmentors + "_data=" + str(arguments.dataset_percentage) + "_seed=" + str(arguments.seed)
        logger = TensorBoardLogger(
            "search_translate", name=dir
        )

        lr_monitor = LearningRateMonitor(logging_interval="step")
        early_stop_callback = early_stopping.EarlyStopping(
            monitor='validation_loss',
            min_delta=0,
            patience=3,
            mode='min',
        )

        checkpoint_callback = ModelCheckpoint(
            dirpath='search_translate',
            save_last=True,
            save_top_k=1,
            save_weights_only=True,
            filename=filename,
            auto_insert_metric_name=False,
            reset_on_train_end=True  # Reset the callback between trials
        )
        
        early_pruning_callback = PyTorchLightningPruningCallback(trial, monitor="validation_bleu")
        
        print(arguments)

        trainer = pl.Trainer.from_argparse_args(
            arguments, logger=logger, replace_sampler_ddp=False, callbacks=[lr_monitor, early_stop_callback, early_pruning_callback, checkpoint_callback], plugins=[SLURMEnvironment(requeue_signal=signal.SIGUSR1)]
        )  # , distributed_backend='ddp_cpu')
        
        # for batch_idx, batch in enumerate(data.split_and_pad_data(data.dataset['train'])):
        #     input_, output = batch
        #     print(input_['src_len'])
        
        model = Seq2SeqTranslator(
            model_name = MODEL_NAME,
            max_epochs = arguments.max_epochs,
            tokenizer = data.tokenizer,
            steps_per_epoch = int(len(data.train_dataloader())),
            pretrain = arguments.pretrain,
            augmentors = embed_augmentors
        ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        # most basic trainer, uses good defaults (1 gpu)
        trainer.fit(model, data)
        trainer.test(model, dataloaders = data.test_dataloader())
        test_accuracy = trainer.callback_metrics["test_bleu"].item()
        return test_accuracy

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, timeout = 43200)
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

def seq2seq_translate_search_lr():
    MODEL_NAME = "t5-small"

    # add PROGRAM level args
    parser = ArgumentParser(conflict_handler = 'resolve')

    # add PROGRAM level args
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--seed", type=int, default=0)
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
    parser.add_argument("--use_high_lr", type=bool, default=False)
    parser.add_argument("--pretrain", action="store_true")
    parser.add_argument("--no_pretrain",  dest='pretrain', action="store_false")
    parser.set_defaults(pretrain=False)
    # parser.add_argument("--deterministic", type=bool, default=True)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    set_seed(args.seed)

    def objective(trial):
        lr = trial.suggest_float("learning_rate", 4e-5, 1e-3, log=True)
        data = TranslationDataModule(
            model_name = MODEL_NAME,
            dataset_percentage = 1,
            batch_size=args.batch_size
        )
        data.prepare_data()
        data.setup("fit")
        filename = "translate_" + args.augmentors + "_data=" + str(args.dataset_percentage) + "_seed=" + str(args.seed)
        logger = TensorBoardLogger(
            "search_translate", name=dir
        )

        args.default_root_dir = "search_translate/" + dir

        lr_monitor = LearningRateMonitor(logging_interval="step")
        early_stop_callback = early_stopping.EarlyStopping(
            monitor='validation_loss',
            min_delta=0,
            patience=3,
            mode='min',
        )

        checkpoint_callback = ModelCheckpoint(
            dirpath='search_translate',
            save_last=True,
            save_top_k=1,
            save_weights_only=True,
            filename=filename,
            auto_insert_metric_name=False,
            reset_on_train_end=True  # Reset the callback between trials
        )
        
        early_pruning_callback = PyTorchLightningPruningCallback(trial, monitor="validation_bleu")
        
        print(args)

        trainer = pl.Trainer.from_argparse_args(
            args, logger=logger, replace_sampler_ddp=False, callbacks=[lr_monitor, early_stop_callback, early_pruning_callback, checkpoint_callback], plugins=[SLURMEnvironment(requeue_signal=signal.SIGUSR1)]
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
            learning_rate = lr
        ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        # most basic trainer, uses good defaults (1 gpu)
        trainer.fit(model, data)
        trainer.test(model, dataloaders = data.test_dataloader())
        test_bleu = trainer.callback_metrics["test_bleu"].item()
        return test_bleu

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, timeout = 43200)
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


def better_text_classify_search_aug():
    
    def objective(trial):
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
        arguments = parser.parse_args()
        set_seed(arguments.seed)
        augmentator_mapping = {"sr": Synonym_Replacer("english"), "bt": Back_Translator("en"), "in": Insertor("english"), "de": Deletor(), "co": CutOut(), "cm": CutMix(), "mu": MixUp()}
        augmentation_param_range = {"sr": (0,50), "bt": (0,500), "in": (0,50), "de": (0,50), "co": (0,100), "cm": (0,100), "mu": (0,100)}
        try:
            learning_rate = float(arguments.learning_rate)
        except ValueError:
            raise Exception("Learning rate argument should be a float")
        augmentation_params = []
        augmentor_names = list(filter(lambda x: x != "", (arguments.augmentors.split(","))))
        for name in augmentor_names:
            param_range = augmentation_param_range[name]
            augmentation_params.append(trial.suggest_int(f"{name} augmentation param", param_range[0], param_range[1]))
        word_augmentors, embed_augmentors = parse_augmentors(arguments, augmentator_mapping)
        data_modules = {"glue": GlueDataModule, "twitter": TwitterDataModule, "bias_detection": BiasDetectionDataModule, "ag_news": AGNewsDataModule, "imdb": IMDBDataModule, "trec": TrecDataModule, "dbpedia": DBPediaDataModule}
        data = data_modules[arguments.task](
            dataset_percentage = 1,
            augmentors = word_augmentors,
            batch_size = arguments.batch_size
        )
        data.setup("fit")

        filename = str(arguments.task) + "_" + arguments.augmentors + "_data=" + str(arguments.dataset_percentage) + "_seed=" + str(arguments.seed)
        logger = TensorBoardLogger(
            "runs_hyperparam_search_better_text_classify", name=dir
        )

        checkpoint_callback = ModelCheckpoint(
            dirpath='runs_hyperparam_search_better_text_classify',
            save_last=True,
            save_top_k=1,
            save_weights_only=True,
            filename=filename,
            auto_insert_metric_name=False,
            reset_on_train_end=True  # Reset the callback between trials
        )

        lr_monitor = LearningRateMonitor(logging_interval="step")
        early_stop_callback = early_stopping.EarlyStopping(
            monitor='validation_loss',
            min_delta=0,
            patience=3,
            mode='min',
        )
        early_pruning_callback = PyTorchLightningPruningCallback(trial, monitor="validation_accuracy")
        
        print(arguments)

        trainer = pl.Trainer.from_argparse_args(
            arguments, logger=logger, replace_sampler_ddp=False, callbacks=[lr_monitor, early_stop_callback, early_pruning_callback, checkpoint_callback], plugins=[SLURMEnvironment(requeue_signal=signal.SIGUSR1)]
        )  # , distributed_backend='ddp_cpu')
                
        # for batch_idx, batch in enumerate(data.split_and_pad_data(data.dataset['train'])):
        #     input_, output = batch
        #     print(input_['src_len'])    

        # id2label = {0: "WORLD", 1: "SPORTS", 2: "BUSINESS", 3: "SCIENCE"}
        # label2id = {"WORLD": 0, "SPORTS": 1, "BUSINESS": 2, "SCIENCE": 3}

        model = Better_Text_Classifier(
            learning_rate = learning_rate,
            max_epochs = arguments.max_epochs,
            tokenizer = data.tokenizer,
            steps_per_epoch = int(len(data.train_dataloader())),
            num_labels = len(data.id2label),
            id2label = data.id2label,
            label2id = data.label2id,
            pretrain = arguments.pretrain,
            augmentors = embed_augmentors
        ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        # most basic trainer, uses good defaults (1 gpu)
        trainer.fit(model, data)
        trainer.test(model, dataloaders = data.test_dataloader())
        test_accuracy = trainer.callback_metrics["test_accuracy"].item()
        return test_accuracy

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials = 10, timeout = 32400)

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

def better_text_classify_search_lr():
    parser = ArgumentParser(conflict_handler = 'resolve')

    # add PROGRAM level args
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
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
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    set_seed(args.seed)

    def objective(trial):
        lr = trial.suggest_float("learning_rate", 4e-5, 1e-2, log=True)
        data_modules = {"glue": GlueDataModule, "twitter": TwitterDataModule, "bias_detection": BiasDetectionDataModule, "ag_news": AGNewsDataModule, "imdb": IMDBDataModule, "trec": TrecDataModule, "dbpedia": DBPediaDataModule}
        
        if args.samples_per_class is not None:
            args.dataset_percentage = 100

        data = data_modules[args.task](
            dataset_percentage = args.dataset_percentage / 100,
            batch_size = args.batch_size
        )

        if args.samples_per_class is not None:
            data = FewShotTextClassifyWrapperModule(data, args.samples_per_class)

        data.prepare_data()
        data.setup("fit")
        dir = args.task + "_" + args.augmentors + "_data=" + str(args.dataset_percentage) + "seed=" + str(args.seed) + "lr=" + str(lr)
        logger = TensorBoardLogger(
            "runs_hyperparam_search_better_text_classify", name=dir
        )

        checkpoint_callback = ModelCheckpoint(
            dirpath="runs_hyperparam_search_better_text_classify/" + dir,
            filename='my_model-{epoch:02d}-{val_loss:.2f}',
            monitor=os.environ.get('SLURM_JOB_ID', None),
            save_top_k=1,
            mode='min'
        )

        lr_monitor = LearningRateMonitor(logging_interval="step")
        early_stop_callback = early_stopping.EarlyStopping(
            monitor='validation_loss',
            min_delta=0,
            patience=3,
            mode='min',
        )
        early_pruning_callback = PyTorchLightningPruningCallback(trial, monitor="validation_accuracy")
        
        print(args)

        trainer = pl.Trainer.from_argparse_args(
            args, logger=logger, replace_sampler_ddp=False, callbacks=[lr_monitor, early_stop_callback, early_pruning_callback, checkpoint_callback]
        )  # , distributed_backend='ddp_cpu')
        
        # for batch_idx, batch in enumerate(data.split_and_pad_data(data.dataset['train'])):
        #     input_, output = batch
        #     print(input_['src_len'])    

        # id2label = {0: "WORLD", 1: "SPORTS", 2: "BUSINESS", 3: "SCIENCE"}
        # label2id = {"WORLD": 0, "SPORTS": 1, "BUSINESS": 2, "SCIENCE": 3}

        model = Better_Text_Classifier(
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
    study.optimize(objective, n_trials = 50, timeout = 14400)

    print(f"Best value: {study.best_value:.4f}")
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")