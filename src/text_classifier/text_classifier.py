import torch
from torchtext.datasets import AG_NEWS as agnews
from data_preprocessor import EnglishPreProcessor

training_iter = iter(agnews(split = 'train'))
eng_pre_processor_train = EnglishPreProcessor(training_iter)

print(eng_pre_processor_train.text_pipeline('here is the an example'))
print(eng_pre_processor_train.get_dataloader())