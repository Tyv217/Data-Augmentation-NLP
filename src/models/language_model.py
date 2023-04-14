import torch
import numpy as np
import pytorch_lightning as pl
from transformers import DistilBertForMaskedLM, AutoConfig, DistilBertTokenizer
import math

class LanguageModelModule(pl.LightningModule):
    def __init__(self, learning_rate, max_epochs, tokenizer, steps_per_epoch, augmentors = []):
        super().__init__()
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        MODEL_NAME = "distilbert-base-uncased"
        self.pretrain = True
        self.tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME, do_lower_case=True)
        self.config = AutoConfig.from_pretrained(MODEL_NAME)
        self.model = DistilBertForMaskedLM(self.config)
        self.steps_per_epoch = steps_per_epoch
        self.augmentors = augmentors
        # self.metric = Perplexity(ignore_index = -100).to("cuda" if torch.cuda.is_available() else "cpu")

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

        batch_size, seq_len, num_classes = logits.shape
        logits = logits.view(batch_size * seq_len, num_classes)
        labels = batch['label'].view(batch_size * seq_len)

        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)

        perplexity = math.exp(loss)
        
        self.log(
            "validation_loss",
            loss.item(),
            on_step = True,
            on_epoch = True,
            prog_bar = True,
            logger = True,
            sync_dist = True,
        )

        self.log(
            "validation_perplexity",
            perplexity,
            on_step = False,
            on_epoch = True,
            prog_bar = True,
            logger = True,
            sync_dist = True,
        )

        return loss, perplexity
        
    # def on_validation_epoch_end(self):
    #     perplexity = self.metric.compute()
    #     self.metric.reset()

    #     self.log(
    #         "validation_perplexity",
    #         perplexity.item(),
    #         on_step = False,
    #         on_epoch = True,
    #         prog_bar = True,
    #         logger = True,
    #         sync_dist = True,
    #     )

    #     return perplexity
    
    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            output = self.forward(input_id = batch['input_id'], attention_mask = batch['attention_mask'], label = batch['label'])
        logits = output.logits
        batch_size, seq_len, num_classes = logits.shape
        logits = logits.view(batch_size * seq_len, num_classes)
        labels = batch['label'].view(batch_size * seq_len)

        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)

        perplexity = math.exp(loss)

        self.log(
            "test_perplexity",
            perplexity,
            on_step = False,
            on_epoch = True,
            prog_bar = True,
            logger = True,
            sync_dist = True,
        )

        return loss, perplexity
        
    # def on_test_epoch_end(self):
    #     perplexity = self.metric.compute()
    #     self.metric.reset()

    #     self.log(
    #         "test_perplexity",
    #         perplexity.item(),
    #         on_step = False,
    #         on_epoch = True,
    #         prog_bar = True,
    #         logger = True,
    #         sync_dist = True,
    #     )

    #     return perplexity