import torch, time
from torchtext.datasets import AG_NEWS as agnews
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
from models import EnglishPreProcessor, TextClassifierModel

def text_classifier_main():
    training_iter = iter(agnews(split = 'train'))
    eng_pre_processor_train = EnglishPreProcessor(training_iter)
    num_class = len(set([label for (label, text) in training_iter]))
    vocab_size = len(eng_pre_processor_train.get_vocab())
    embed_size = 64
    model = TextClassifierModel(vocab_size, embed_size, num_class).to(eng_pre_processor_train.get_device())

def train_model(dataloader, model, optimizer):
    model.train()
    accuracy, count = 0, 0
    log_interval = 500
    start_time = time.time()
    for index, (label, text, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model(text, offsets) # Calls forward function


# Use pytorch lightning


def run_model(model):
    EPOCHS = 10
    LEARNING_RATE = 5
    BATCH_SIZE = 64

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model)