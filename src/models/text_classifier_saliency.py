import torch
import numpy as np
import pytorch_lightning as pl
from transformers import AutoModelForSequenceClassification
import re

class TextClassifierSaliencyModule(pl.LightningModule):
    def __init__(self, learning_rate, max_epochs, tokenizer, steps_per_epoch, num_labels, id2label, label2id, invert_saliency, pretrain = True, word_augmentors = [],  embed_augmentors = []):
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
        self.word_augmentors = word_augmentors
        self.embed_augmentors = embed_augmentors
        self.saliency_scores = {}
        self.saliency_scores_per_word = {}
        self.invert_saliency = False if invert_saliency == 0 else True

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
    
    def get_saliency_scores(self, input_lines, input_ids, attentions):
        tokenized_inputs = self.tokenizer.convert_ids_to_tokens(input_ids)
        # print(encoded_inputs)
        # print(tokenized_inputs)
        # print(attention_masks)
        input_lines = re.sub(' +', ' ', input_lines).lstrip().rstrip()
        words = input_lines.split(" ")
        tokens = [token.lstrip('▁').replace("##", "", 1) for token in tokenized_inputs]
        special_tokens = np.logical_and(np.char.startswith(np.array(tokens), '['), np.char.endswith(np.array(tokens), ']'))
        non_special_indices = np.nonzero(~special_tokens)
        
        attentions = np.array(attentions)[non_special_indices]
        tokens = np.array(tokens)[non_special_indices]
        num_words = len(words)
        word_weights = np.empty(num_words, dtype=float)
        token_index = 0
        try:
            for i in range(num_words):
                curr_tokens = tokens[token_index]
                word_weights[i] = attentions[token_index]
                while(curr_tokens != words[i].lower()):
                    token_index += 1
                    curr_tokens += tokens[token_index]
                    word_weights[i] += attentions[token_index]
                token_index += 1
        except:
            return []
        
        if self.invert_saliency:
            word_weights = 1 / word_weights

        return word_weights / np.sum(word_weights)
    
    def adapt_scores_to_original_line(self, original_line, input_line, scores):
        original_words = original_line.split(" ")
        input_words = input_line.split(" ")
        if len(original_words) == len(input_words):
            return scores
        else:
            new_scores = []
            for i in range(len(original_words)):
                original_word = original_words[i]
                if original_word in input_words:
                    index = input_words.index(original_word)
                    new_scores.append(scores[index])
                else:
                    continue
            new_scores = np.array(new_scores)
            return new_scores / np.sum(new_scores)
    
    def training_step(self, batch, batch_idx):
        original_lines = batch['input_lines']
        label = batch['label'].to(torch.float)
        saliency_scores = [self.saliency_scores.get(input_line, np.array([])) for input_line in original_lines]
        for augmentor in self.word_augmentors:
                input_lines, _, label = augmentor.augment_dataset_with_saliency(original_lines, None, label, saliency_scores)
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
        inputs_embeds = self.model.distilbert.embeddings(input_ids)

        for augmentor in self.embed_augmentors:
            inputs_embeds, attention_masks, label = augmentor.augment_dataset(inputs_embeds, attention_masks, label)

        output = self.model(inputs_embeds = inputs_embeds, attention_mask = attention_masks, labels = label)

        loss = output.loss
        attention_weights = output.attentions

        saliency_scores_tokens = attention_weights[0].detach().sum(dim=1).sum(dim=1)
        
        for lines, original_line, ids, attentions in zip(input_lines, original_lines, input_ids, saliency_scores_tokens):
            saliency_scores_words = self.get_saliency_scores(lines, ids.detach().cpu(), attentions.detach().cpu())
            saliency_scores_words = self.adapt_scores_to_original_line(original_line, lines, saliency_scores_words )
            self.saliency_scores[original_line] = saliency_scores_words
            for word, score in zip(re.sub(' +', ' ', lines).lstrip().rstrip()\
                                   .split(" "), saliency_scores_words):
                if word in self.saliency_scores_per_word.keys():
                    self.saliency_scores_per_word[word].append(score)
                else:
                    self.saliency_scores_per_word[word] = [score]

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
        input_ids, attention_masks = input_encoding.input_ids.to(self.device), input_encoding.attention_mask.to(self.device)
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