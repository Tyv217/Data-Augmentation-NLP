import torch
import numpy as np
import pytorch_lightning as pl
from transformers import DistilBertForMaskedLM, AutoConfig, DistilBertTokenizer
from torcheval.metrics.text import Perplexity

class LanguageModelModule(pl.LightningModule):
    def __init__(self, model, data, learning_rate, max_epochs, tokenizer, steps_per_epoch, augmentors = []):
        super().__init__()
        self.initial_child_model = model
        self.child_model = model

    def initialize_child_model(self):
        self.child_model = self.initial_child_model

    def forward(self, input_id, attention_mask, label):
        return self.model(input_id, attention_mask = attention_mask, labels = label)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr = self.learning_rate)
        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr = self.learning_rate,
                steps_per_epoch = self.steps_per_epoch,
                epochs = self.max_epochs,
                anneal_strategy = "linear",
                final_div_factor = 1000,
                pct_start = 0.01,
            ),
            "name": "learning_rate",
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [lr_scheduler]
    
    def training_step(self, batch, batch_idx):
        input = batch['input_id']
        attention_mask = batch['attention_mask']
        label = batch['label']
        
        inputs_embeds = self.model.distilbert.embeddings(input)

        for augmentor in self.augmentors:
            inputs_embeds, attention_mask, label = augmentor.augment_dataset(inputs_embeds, attention_mask, label)

        loss = self.model(inputs_embeds = inputs_embeds, attention_mask = attention_mask, labels = label).loss

        self.log(
            "training_loss",
            loss.item(),
            on_step = True,
            on_epoch = True,
            prog_bar = True,
            logger = True,
            sync_dist = True,
        )

        return loss
    
    def validation_step(self, batch, batch_idx):
        output = self.forward(input_id = batch['input_id'], attention_mask = batch['attention_mask'], label = batch['label'])
        loss = output.loss
        logits = output.logits
        self.metric.update(logits, batch['input_id'])
        
        self.log(
            "validation_loss",
            loss.item(),
            on_step = True,
            on_epoch = True,
            prog_bar = True,
            logger = True,
            sync_dist = True,
        )

        return loss
        
    def on_validation_epoch_end(self):
        perplexity = self.metric.compute()
        self.metric.reset()

        self.log(
            "validation_perplexity",
            perplexity.item(),
            on_step = False,
            on_epoch = True,
            prog_bar = True,
            logger = True,
            sync_dist = True,
        )

        return perplexity
    
    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            output = self.forward(input_id = batch['input_id'], attention_mask = batch['attention_mask'], label = batch['label'])
        logits = output.logits
        self.metric.update(logits, batch['input_id'])
        
    def on_test_epoch_end(self):
        perplexity = self.metric.compute()
        self.metric.reset()

        self.log(
            "test_perplexity",
            perplexity.item(),
            on_step = False,
            on_epoch = True,
            prog_bar = True,
            logger = True,
            sync_dist = True,
        )

        return perplexity