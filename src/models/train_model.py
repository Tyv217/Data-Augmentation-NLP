import torch, time, random
from torchtext.datasets import AG_NEWS as agnews
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
from torch.utils.tensorboard import SummaryWriter
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from argparse import ArgumentParser
from ..helpers import EnglishPreProcessor, Logger
from .text_classifier import TextClassifierEmbeddingModel
from .seq2seq_translator import Seq2SeqTranslator
from ..data import TranslationDataModule
from pytorch_lightning.loggers import TensorBoardLogger


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

def run_model_text_classifier(model, train_iter, augmentation_percentage, augmentor):
    EPOCHS = 7
    LEARNING_RATE = 0.1
    MOMENTUM = 0.9
    BATCH_SIZE = 16
    
    writer = SummaryWriter()
    
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = LEARNING_RATE, momentum = MOMENTUM)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma = 0.1)

    total_accuracy = None
    _, test_iter = agnews()



    test_dataset = to_map_style_dataset(test_iter)


    preprocessor = EnglishPreProcessor(train_iter)
    logger = Logger()
    test_dataloader = preprocessor.get_dataloader(test_dataset, batch_size = BATCH_SIZE, shuffle = True)

    for epoch_number in range(1, EPOCHS + 1):
        torch.cuda.empty_cache()
        start_time = time.time()
        train_iter = augmentor.augment_dataset(train_iter, augmentation_percentage)
        print("Augmented!")
        train_dataset = to_map_style_dataset(train_iter)
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

def text_classify(augmentation_percentage, augmentor):
    train_iter = list(agnews(split='train'))
    random.shuffle(train_iter)
    # train_iter = train_iter[:len(train_iter) // ]
    print("Augmentation Percentage:", str(augmentation_percentage * 100) + "%")
    eng_pre_processor_train = EnglishPreProcessor(train_iter)
    num_class = len(set([label for (label, text) in train_iter]))
    vocab_size = len(eng_pre_processor_train.get_vocab())
    embed_size = 64
    model = TextClassifierEmbeddingModel(vocab_size, embed_size, num_class).to(eng_pre_processor_train.get_device())
    accuracy = run_model_text_classifier(model, train_iter, augmentation_percentage, augmentor)
    return accuracy

def seq2seq_translate():
    parser = ArgumentParser()

    # add PROGRAM level args
    parser.add_argument("--N_samples", type=int, default=256 * 10)
    parser.add_argument("--N_valid_size", type=int, default=32 * 10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--embed_size", type=int, default=32)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    data = TranslationDataModule(batch_size=args.batch_size)
    data.prepare_data()
    data.setup("fit")

    logger = TensorBoardLogger(
        "runs", name="fit"
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    print(args)
    trainer = pl.Trainer.from_argparse_args(
        args, logger=logger, replace_sampler_ddp=False, callbacks=[lr_monitor]
    )  # , distributed_backend='ddp_cpu')
    
    # for batch_idx, batch in enumerate(data.split_and_pad_data(data.dataset['train'])):
    #     input_, output = batch
    #     print(input_['src_len'])
    
    model = Seq2SeqTranslator(
        tokenizer = data.tokenizer,
        steps_per_epoch = int(len(data.train_dataloader()))
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # most basic trainer, uses good defaults (1 gpu)
    # trainer.tune(model, data)
    trainer.fit(model, data)