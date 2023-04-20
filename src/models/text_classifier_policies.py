import torch
import numpy as np
import pytorch_lightning as pl
from transformers import AutoModelForSequenceClassification
import re

class TextClassifierPolicyModule(pl.LightningModule):
    def __init__(self, learning_rate, max_epochs, tokenizer, steps_per_epoch, num_labels, id2label, label2id, pretrain = True, training_policies = [], embed_augmentors = []):
        super().__init__()
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.tokenizer = tokenizer
        self.id2label = id2label
        self.label2id = label2id
        self.model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels = num_labels, id2label = id2label, label2id = label2id, problem_type="multi_label_classification", output_attentions=True)
        if not pretrain:
            self.model.init_weights()
        self.steps_per_epoch = steps_per_epoch
        self.training_policies = training_policies
        self.validation_policies = []
        self.embed_augmentors = embed_augmentors
        self.saliency_scores = {}
        self.saliency_scores_per_word = {}

    def set_validation_policy(self, validation_policy):
        self.validation_policy = validation_policy

    def forward(self, input_id, attention_mask, label):
        label = label.to(torch.float)
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
        original_lines = batch['input_lines']
        label = batch['label'].to(torch.float)
        
        augmentors_on_words = [[] for i in range(len(original_lines))]
        augmentors_on_embeddings = [[] for i in range(len(original_lines))]
        if len(self.training_policies) > 0:
            policies = np.random.choice(self.training_policies, len(original_lines))
            for i, row in enumerate(policies):
                for augmentor in row:
                    if augmentor.operate_on_embeddings:
                        augmentors_on_embeddings[i].append(augmentor)
                    else:
                        augmentors_on_words[i].append(augmentor)

        new_lines = []
        for line, augmentors in zip(original_lines, augmentors_on_words):
            new_line = line
            for augmentor in augmentors:
                new_line = augmentor.augment_one_sample(new_line)
            new_lines.append(new_line)

        input_encoding = self.tokenizer.batch_encode_plus(
            new_lines,
            add_special_tokens = True,
            max_length = 400,
            padding = "max_length",
            truncation = True,
            return_attention_mask = True,
            return_tensors = "pt",
        )
        input_ids, attention_masks = input_encoding.input_ids.to(self.device), input_encoding.attention_mask.to(self.device)
        inputs_embeds = self.model.distilbert.embeddings(input_ids)

        new_samples = []
        all_samples = zip(inputs_embeds, attention_masks, label)

        for sample, augmentors in zip(all_samples, augmentors_on_embeddings):
            sentence, attention_mask, label = sample
            new_samples_curr = []
            for augmentor in augmentors:
                new_lines = augmentor.augment_one_sample(sentence, attention_mask, label)
                if(new_lines is not None):
                    if(len(new_lines) == 1): # Cutout
                        sentence, attention_mask, label = new_lines[0]
                    else: # Mixup or CutMix
                        new_samples_curr.append(new_lines[1])
                else:
                    continue

            new_samples_curr.append((sentence, attention_mask, label))
            new_samples.extend(new_samples_curr)
        
        inputs_embeds, attention_masks, label = zip(*new_samples)

        output = self.model(inputs_embeds = inputs_embeds, attention_mask = attention_masks, labels = label)

        loss = output.loss

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
        original_lines = batch['input_lines']
        label = batch['label'].to(torch.float)
        
        augmentors_on_words = [[] for i in range(len(original_lines))]
        augmentors_on_embeddings = [[] for i in range(len(original_lines))]
        has_embedding_augmentors = False
        if len(self.validation_policies) > 0:
            policies = np.random.choice(self.validation_policies, len(original_lines))
            for i, row in enumerate(policies):
                for augmentor in row:
                    if augmentor.operate_on_embeddings:
                        has_embedding_augmentors = True
                        augmentors_on_embeddings[i].append(augmentor)
                    else:
                        augmentors_on_words[i].append(augmentor)

        new_lines = []
        for line, augmentors in zip(original_lines, augmentors_on_words):
            new_line = line
            for augmentor in augmentors:
                new_line = augmentor.augment_one_sample(new_line)
            new_lines.append(new_line)

        input_encoding = self.tokenizer.batch_encode_plus(
            new_lines,
            add_special_tokens = True,
            max_length = 400,
            padding = "max_length",
            truncation = True,
            return_attention_mask = True,
            return_tensors = "pt",
        )
        input_ids, attention_masks = input_encoding.input_ids.to(self.device), input_encoding.attention_mask.to(self.device)
        inputs_embeds = self.model.distilbert.embeddings(input_ids)

        new_samples = []
        all_samples = zip(inputs_embeds, attention_masks, label)

        for sample, augmentors in zip(all_samples, augmentors_on_embeddings):
            sentence, attention_mask, label = sample
            new_samples_curr = []
            for augmentor in augmentors:
                new_lines = augmentor.augment_one_sample(sentence, attention_mask, label)
                if(new_lines is not None):
                    if(len(new_lines) == 1): # Cutout
                        sentence, attention_mask, label = new_lines[0]
                    else: # Mixup or CutMix
                        new_samples_curr.append(new_lines[1])
                else:
                    continue

            new_samples_curr.append((sentence, attention_mask, label))
            new_samples.extend(new_samples_curr)
        
        inputs_embeds, attention_masks, label = zip(*new_samples)
        inputs_embeds = torch.stack(inputs_embeds)
        attention_masks = torch.stack(attention_masks)
        label = torch.stack(label)

        import pdb
        pdb.set_trace()

        output = self.model(inputs_embeds = inputs_embeds, attention_mask = attention_masks, labels = label)
        loss = output.loss

        self.log(
            "validation_loss",
            loss.item(),
            on_step = True,
            on_epoch = True,
            prog_bar = True,
            logger = True,
            sync_dist = True,
        )

        if has_embedding_augmentors:
            return loss
        else:
            logits = output.logits
            pred_flat = torch.argmax(logits, axis=1).flatten()
            labels_flat = torch.argmax(batch['label'], axis=1).flatten()
            acc = torch.sum(pred_flat == labels_flat) / len(labels_flat)
            self.log(
                "validation_accuracy",
                acc.item(),
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
        input_ids, attention_masks = input_encoding.input_ids.to(self.device), input_encoding.attention_mask.to(self.device)
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