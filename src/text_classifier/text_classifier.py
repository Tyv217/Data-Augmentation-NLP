import torch
from torchtext.datasets import AG_NEWS as agnews

training_iter = iter(agnews(split = 'train'))