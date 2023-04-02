import torch
import numpy as np
import pytorch_lightning as pl
from transformers import AutoModelForSequenceClassification

class Better_Text_Classifier_With_Saliency(pl.LightningModule):
    def __init__(self, learning_rate, max_epochs, steps_per_epoch, num_labels, id2label, label2id, pretrain = True, word_augmentors = [], embed_augmentors = []):
        super().__init__()
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.id2label = id2label
        self.label2id = label2id
        self.model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels = num_labels, id2label = id2label, label2id = label2id, problem_type="multi_label_classification")
        if not pretrain:
            self.model.init_weights()
        self.steps_per_epoch = steps_per_epoch
        self.word_augmentors = word_augmentors
        self.embed_augmentors = embed_augmentors
        self.saliency_scores = {}
        self.new_saliency_scores = {}

    def forward(self, input_id, attention_mask, label):
        label = label.to(torch.float)
        return self.model(input_id, attention_mask = attention_mask, labels = label, output_attentions = True)

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
        input_lines = batch['input_lines']
        saliency_scores = [self.saliency_scores[input_line] for input_line in input_lines]
        for augmentor in self.word_augmentors:
                input_lines, _, labels = augmentor.augment_dataset_with_saliency(input_lines, None, labels, saliency_scores)
        input_encoding = self.tokenizer.batch_encode_plus(
            input_lines,
            add_special_tokens = True,
            max_length = 400,
            padding = "max_length",
            truncation = True,
            return_attention_mask = True,
            return_tensors = "pt",
        )
        input_ids, attention_masks = input_encoding.input_ids, input_encoding.attention_mask
        inputs_embeds = self.model.distilbert.embeddings(input_ids)

        for augmentor in self.embed_augmentors:
            inputs_embeds, attention_masks, label = augmentor.augment_dataset(inputs_embeds, attention_masks, label)

        output = self.model(inputs_embeds = inputs_embeds, attention_mask = attention_masks, labels = label)

        loss = output.loss
        attention_weights = output.attentions
        saliency_scores = attention_weights.sum(dim=1).sum(dim=1)
        # input_tokens = self.tokenizer.decode(encoded_input['input_ids'])

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
    
    def on_train_epoch_end(self):
        self.saliency_scores = self.new_saliency_scores
        self.new_saliency_scores = {}
    
    def validation_step(self, batch, batch_idx):
        input_lines = batch['input_lines']
        input_encoding = self.tokenizer.batch_encode_plus(
            input_lines,
            add_special_tokens = True,
            max_length = 400,
            padding = "max_length",
            truncation = True,
            return_attention_mask = True,
            return_tensors = "pt",
        )
        input_ids, attention_masks = input_encoding.input_ids, input_encoding.attention_mask
        with torch.no_grad():
            output = self.forward(input_id = input_ids, attention_mask = attention_masks, label = batch['label'])
        loss = output.loss
        logits = output.logits
        pred_flat = torch.argmax(logits, axis=1).flatten()
        labels_flat = torch.argmax(batch['label'], axis=1).flatten()
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
        input_lines = batch['input_lines']
        input_encoding = self.tokenizer.batch_encode_plus(
            input_lines,
            add_special_tokens = True,
            max_length = 400,
            padding = "max_length",
            truncation = True,
            return_attention_mask = True,
            return_tensors = "pt",
        )
        input_ids, attention_masks = input_encoding.input_ids, input_encoding.attention_mask
        with torch.no_grad():
            output = self.forward(input_id = input_ids, attention_mask = attention_masks, label = batch['label'])
        logits = output.logits
        pred_flat = torch.argmax(logits, axis=1).flatten()
        labels_flat = torch.argmax(batch['label'], axis=1).flatten()
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