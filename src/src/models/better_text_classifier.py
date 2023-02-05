import torch
import numpy as np
import pytorch_lightning as pl
from transformers import AutoModelForSequenceClassification

class Better_Text_Classifier(pl.LightningModule):
    def __init__(self, max_epochs, steps_per_epoch):
        super().__init__()
        self.learning_rate = 5e-5
        self.max_epochs = max_epochs
        id2label = {0: "WORLD", 1: "SPORTS", 2: "BUSINESS", 3: "SCIENCE"}
        label2id = {"WORLD": 0, "SPORTS": 1, "BUSINESS": 2, "SCIENCE": 3}
        self.model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels = 4, id2label = id2label, label2id = label2id)
        self.steps_per_epoch = steps_per_epoch

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
        loss = self.forward(batch['input_id'], batch['attention_mask'], batch['label']).loss

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
        pred_flat = torch.argmax(logits, axis=1).flatten()
        labels_flat = batch['label'].flatten()
        acc = torch.sum(pred_flat == labels_flat) / len(labels_flat)

        self.log(
            "validation_accuracy",
            acc.item(),
            on_step = False,
            on_epoch = True,
            prog_bar = True,
            logger = True,
            sync_dist = True,
        )

        self.log(
            "validation_loss",
            loss.item(),
            on_step = True,
            on_epoch = True,
            prog_bar = True,
            logger = True,
            sync_dist = True,
        )

        return loss, acc
    
    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            output = self.forward(input_id = batch['input_id'], attention_mask = batch['attention_mask'], label = batch['label'])
        logits = output.logits
        pred_flat = torch.argmax(logits, axis=1).flatten()
        labels_flat = batch['label'].flatten()
        acc = torch.sum(pred_flat == labels_flat) / len(labels_flat)

        self.log(
            "test_accuracy",
            acc.item(),
            on_step = False,
            on_epoch = True,
            prog_bar = True,
            logger = True,
            sync_dist = True,
        )

        return acc