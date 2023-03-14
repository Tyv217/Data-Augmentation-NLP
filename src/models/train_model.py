import torch, time, random
from torchtext.datasets import AG_NEWS as agnews
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
from torch.utils.tensorboard import SummaryWriter
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, early_stopping
from argparse import ArgumentParser
from ..helpers import EnglishPreProcessor, Logger, parse_augmentors, set_seed
from .text_classifier import TextClassifierEmbeddingModel
from .seq2seq_translator import Seq2SeqTranslator
from ..data import TranslationDataModule, AGNewsDataModule, GlueDataModule, TwitterDataModule, BiasDetectionDataModule
from pytorch_lightning.loggers import TensorBoardLogger
from .better_text_classifier import Better_Text_Classifier
from .data_augmentors import Synonym_Replacer, Back_Translator, Insertor, Deletor, CutOut, CutMix
from pytorch_lightning.plugins.environments import SLURMEnvironment
import signal

def train_model_text_classifier(dataloader, model, loss_fn, optimizer, epoch_number, logger, writer):
    model.train()
    accuracy, count = 0, 0
    log_interval = 500
    start_time = time.time()

    for index, (label, text, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model(text, offsets) # Calls forward function

        loss = loss_fn(predicted_label, label)
        writer.add_scalar("Loss/train", loss, epoch_number)
        loss.backward() 

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()

        accuracy += (predicted_label.argmax(1) == label).sum().item()
        count += label.size(0)

        if(index % log_interval == 0 and index > 0):
            elapsed = time.time() - start_time
            logger.log_batch(epoch_number, index, len(dataloader), accuracy, count)
            accuracy, count = 0, 0
            start_time = time.time()
        
        writer.flush()

def eval_model_text_classifier(dataloader, model, loss_fn, epoch_number, logger):
    model.eval()
    accuracy, count = 0, 0
    with torch.no_grad():
        for index, (label, text, offsets) in enumerate(dataloader):
            predicted_label = model(text, offsets)
            loss = loss_fn(predicted_label, label)
            accuracy += (predicted_label.argmax(1) == label).sum().item()
            count += label.size(0)

    return accuracy / count

# Use pytorch lightning

def run_model_text_classifier(model, train_iter, preprocessor, augmentation_percentage, augmentors, learning_rate):
    EPOCHS = 10
    LEARNING_RATE = learning_rate
    BATCH_SIZE = 64
    
    writer = SummaryWriter()
    
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma = 0.1)

    total_accuracy = None
    _, test_iter = agnews()



    test_dataset = to_map_style_dataset(test_iter)

    logger = Logger()
    test_dataloader = preprocessor.get_dataloader(test_dataset, batch_size = BATCH_SIZE, shuffle = True)

    for epoch_number in range(1, EPOCHS + 1):
        torch.cuda.empty_cache()
        start_time = time.time()
        if augmentors is not None and augmentation_percentage is not None:
            model.eval()
            with torch.no_grad():
                for augmentor in augmentors:
                    augmented_train_iter = augmentor.augment_dataset(train_iter, augmentation_percentage, preprocessor)
                print("Augmented!")
        else:
            augmented_train_iter = train_iter
        train_dataset = to_map_style_dataset(augmented_train_iter)
        num_train = int(len(train_dataset) * 0.95)
        split_train, split_valid = random_split(train_dataset, [num_train, len(train_dataset) - num_train])
        train_dataloader = preprocessor.get_dataloader(split_train, batch_size = BATCH_SIZE, shuffle = True)
        valid_dataloader = preprocessor.get_dataloader(split_valid, batch_size = BATCH_SIZE, shuffle = True)
        train_model_text_classifier(train_dataloader, model, loss_fn, optimizer, epoch_number, logger, writer)
        curr_accuracy = eval_model_text_classifier(valid_dataloader, model, loss_fn, epoch_number, logger)
        if total_accuracy is not None and total_accuracy > curr_accuracy:
            scheduler.step()
            logger.log_epoch(epoch_number, time.time() - start_time, total_accuracy)
        else:
            total_accuracy = curr_accuracy
            logger.log_epoch(epoch_number, time.time() - start_time, curr_accuracy)
    

    test_accuracy = eval_model_text_classifier(test_dataloader, model, loss_fn, epoch_number, logger)
    logger.log_test_result(test_accuracy)
    return test_accuracy
    writer.close()

def text_classify(augmentors, learning_rate, augmentation_percentage = 0, dataset_percentage = 1):
    # print("Gets to here 1")
    # parser = ArgumentParser()
    # print("Gets to here 2")
    # parser.add_argument("--augmentation_prcentage", type=int, default=0)
    # print("Gets to here 3")
    # args = parser.parse_args()
    # print("Gets to here 4")
    print("Gets to here")
    train_iter = list(agnews(split='train'))
    random.shuffle(train_iter)
    train_iter = train_iter[:int(len(train_iter) * dataset_percentage)]
    print("Augmentation Percentage:", str(augmentation_percentage * 100) + "%")
    eng_pre_processor_train = EnglishPreProcessor(train_iter)
    num_class = len(set([label for (label, text) in train_iter]))
    vocab_size = len(eng_pre_processor_train.get_vocab())
    embed_size = 64
    model = TextClassifierEmbeddingModel(vocab_size, embed_size, num_class).to(eng_pre_processor_train.get_device())
    accuracy = run_model_text_classifier(model, train_iter, eng_pre_processor_train, augmentation_percentage, augmentors, learning_rate)
    # with open("/home/xty20/Data-Augmentation-NLP/accuracies.txt", "a") as f:
    #     f.write("Percentage: " + str(augmentation_percentage * 100) + "%, accuracy: " + "{0:.3g}\n".format(accuracy))
    return accuracy

def seq2seq_translate():
    MODEL_NAME = "t5-small"
    parser = ArgumentParser(conflict_handler = 'resolve')

    # add PROGRAM level args
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
    # parser.add_argument("--deterministic", type=bool, default=True)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    set_seed(args.seed)
    augmentator_mapping = {"sr": Synonym_Replacer("english"), "bt": Back_Translator("en"), "in": Insertor("english"), "de": Deletor(), "co": CutOut(), "cm": CutMix()}
    augmentors = parse_augmentors(args, augmentator_mapping)

    data = TranslationDataModule(
        model_name = MODEL_NAME,
        dataset_percentage = args.dataset_percentage / 100,
        augmentors = augmentors,
        batch_size=args.batch_size
    )
    data.prepare_data()
    data.setup("fit")
    dir = "translate_" + args.augmentors + "_data=" + str(args.dataset_percentage) + "_seed=" + str(args.seed)
    logger = TensorBoardLogger(
        "runs_translate", name=dir
    )

    args.default_root_dir = "runs_translate/" + dir

    lr_monitor = LearningRateMonitor(logging_interval="step")
    early_stop_callback = early_stopping.EarlyStopping(
        monitor='validation_loss',
        min_delta=0,
        patience=3,
        mode='min',
    )
    print(args)

    trainer = pl.Trainer.from_argparse_args(
        args, logger=logger, replace_sampler_ddp=False, callbacks=[lr_monitor], plugins=[SLURMEnvironment(requeue_signal=signal.SIGUSR1)]
    )  # , distributed_backend='ddp_cpu')
    
    # for batch_idx, batch in enumerate(data.split_and_pad_data(data.dataset['train'])):
    #     input_, output = batch
    #     print(input_['src_len'])
    
    try:
        use_high_lr = args.use_high_lr
    except:
        use_high_lr = False

    model = Seq2SeqTranslator(
        use_high_lr,
        model_name = MODEL_NAME,
        max_epochs = args.max_epochs,
        tokenizer = data.tokenizer,
        steps_per_epoch = int(len(data.train_dataloader()))
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
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    set_seed(args.seed)
    augmentator_mapping = {"sr": Synonym_Replacer("english"), "bt": Back_Translator("en"), "in": Insertor("english"), "de": Deletor(), "co": CutOut(), "cm": CutMix()}
    augmentors = parse_augmentors(args, augmentator_mapping)

    data_modules = {"glue": GlueDataModule, "twitter": TwitterDataModule, "bias_detection": BiasDetectionDataModule, "ag_news": AGNewsDataModule,}

    data = data_modules[args.task](
        dataset_percentage = args.dataset_percentage / 100,
        augmentors = augmentors,
        batch_size = args.batch_size
    )

    data.prepare_data()
    data.setup("fit")

    logger = TensorBoardLogger(
        "runs_better_text_classify", name=args.task + "_" + args.augmentors + "_data=" + str(args.dataset_percentage) + "seed=" + str(args.seed)
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    early_stop_callback = early_stopping.EarlyStopping(
        monitor='validation_loss',
        min_delta=0,
        patience=3,
        mode='min',
    )
    print(args)

    trainer = pl.Trainer.from_argparse_args(
        args, logger=logger, replace_sampler_ddp=False, callbacks=[lr_monitor, early_stop_callback]
    )  # , distributed_backend='ddp_cpu')
    
    # for batch_idx, batch in enumerate(data.split_and_pad_data(data.dataset['train'])):
    #     input_, output = batch
    #     print(input_['src_len'])    

    # id2label = {0: "WORLD", 1: "SPORTS", 2: "BUSINESS", 3: "SCIENCE"}
    # label2id = {"WORLD": 0, "SPORTS": 1, "BUSINESS": 2, "SCIENCE": 3}

    model = Better_Text_Classifier(
        learning_rate = args.learning_rate,
        max_epochs = args.max_epochs,
        steps_per_epoch = int(len(data.train_dataloader())),
        num_labels = len(data.id2label),
        id2label = data.id2label,
        label2id = data.label2id,
        pretrain = args.pretrain
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    # most basic trainer, uses good defaults (1 gpu)
    if args.train:
        trainer.fit(model, data)
    trainer.test(model, dataloaders = data.test_dataloader())

    print("Seed:", args.seed)
    print("Augmentors:", args.augmentors)
    print("Augmentation params:", args.augmentation_params)
    print("Dataset Percentage:", args.dataset_percentage)
    print("Auto LR Finder Used:", args.auto_lr_find)
