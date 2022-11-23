import torch, time
from torchtext.datasets import AG_NEWS as agnews
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
from torch.utils.tensorboard import SummaryWriter
from .helpers import EnglishPreProcessor, Logger
from .text_classifier import TextClassifierEmbeddingModel


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

def run_model_text_classifier(model):
    EPOCHS = 50
    LEARNING_RATE = 5
    BATCH_SIZE = 64
    
    writer = SummaryWriter()    

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma = 0.1)

    total_accuracy = None
    train_iter, test_iter = agnews()
    print(train_iter)

    train_dataset = to_map_style_dataset(train_iter)
    print(train_dataset)
    test_dataset = to_map_style_dataset(test_iter)
    num_train = int(len(train_dataset) * 0.95)
    split_train, split_valid = random_split(train_dataset, [num_train, len(train_dataset) - num_train], \
        generator = torch.Generator().manual_seed(0))

    preprocessor = EnglishPreProcessor(train_iter)
    logger = Logger()

    train_dataloader = preprocessor.get_dataloader(split_train, batch_size = BATCH_SIZE, shuffle = True)
    valid_dataloader = preprocessor.get_dataloader(split_valid, batch_size = BATCH_SIZE, shuffle = True)
    test_dataloader = preprocessor.get_dataloader(test_dataset, batch_size = BATCH_SIZE, shuffle = True)

    for epoch_number in range(1, EPOCHS + 1):
        start_time = time.time()
        train_model_text_classifier(train_dataloader, model, loss_fn, optimizer, epoch_number, logger, writer)
        curr_accuracy = eval_model_text_classifier(valid_dataloader, model, loss_fn, epoch_number, logger)
        if total_accuracy is not None and total_accuracy > curr_accuracy:
            scheduler.step()
        else:
            total_accuracy = curr_accuracy
            logger.log_epoch(epoch_number, time.time() - start_time, curr_accuracy)

    test_accuracy = eval_model(test_dataloader, model, loss_fn, epoch_number, logger)
    logger.log_test_result(test_accuracy)
    writer.close()

def text_classify():
    train_iter = agnews(split='train')
    eng_pre_processor_train = EnglishPreProcessor(train_iter)
    num_class = len(set([label for (label, text) in train_iter]))
    vocab_size = len(eng_pre_processor_train.get_vocab())
    embed_size = 64
    model = TextClassifierEmbeddingModel(vocab_size, embed_size, num_class).to(eng_pre_processor_train.get_device())
    run_model_text_classifier(model)