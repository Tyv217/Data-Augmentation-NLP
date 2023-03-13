import torch, random, evaluate
import pytorch_lightning as pl
import torchmetrics.functional as plfunc
from transformers import AutoConfig, T5ForConditionalGeneration
from datasets import load_metric

class Seq2SeqTranslator(pl.LightningModule):
    def __init__(self, use_high_lr, model_name, max_epochs, tokenizer, steps_per_epoch):
        super().__init__()
        self.learning_rate = 1e-4
        if use_high_lr:
            self.learning_rate = 1e-3
        self.max_epochs = max_epochs
        self.tokenizer = tokenizer
        self.config = AutoConfig.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration(self.config)
        self.steps_per_epoch = steps_per_epoch
        self.model.resize_token_embeddings(len(tokenizer))
    
    def forward(self, input_id, attention_mask, label):
        return self.model(input_ids = input_id, attention_mask = attention_mask, labels = label)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr = self.learning_rate)
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
        label_ids = batch['label']
        output = self.forward(batch['input_id'], batch['attention_mask'], label_ids)
        loss = output.loss
        logits = output.logits
        pred_output_ids = torch.argmax(logits, axis = 2)
        pred_outputs = self.tokenizer.batch_decode(pred_output_ids, skip_special_tokens = True)
        label_ids[label_ids == -100] = 0
        labels = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        references = [[label] for label in labels]
        # bleu score needs two arguments
        # first: predicted_ids - list of predicted sequences as a list of predicted ids
        # second: target_ids - list of references (can be many, list)
        
        bleu_metric = evaluate.load("sacrebleu")
        bleu_score = bleu_metric.compute(predictions = pred_outputs, references = references)['score']
        loss = self.forward(input_id = batch['input_id'], attention_mask = batch['attention_mask'], label = batch['label']).loss
        # torch.unsqueeze(trg_batchT,1).tolist())
        
        self.log(
            "bleu_score",
            bleu_score,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        self.log(
            "eval_loss",
            loss.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return loss, bleu_score

    def test_step(self, batch, batch_idx):
        pred_output_ids = self.model.generate(input_ids = batch['input_id'], attention_mask = batch['attention_mask'], do_sample=False, max_length = 512)
        pred_outputs = self.tokenizer.batch_decode(pred_output_ids, skip_special_tokens = True)
        label_ids = batch['label']
        label_ids[label_ids == -100] = 0
        labels = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        references = [[label] for label in labels]
        # bleu score needs two arguments
        # first: predicted_ids - list of predicted sequences as a list of predicted ids
        # second: target_ids - list of references (can be many, list)
        
        bleu_metric = evaluate.load("sacrebleu")
        bleu_score = bleu_metric.compute(predictions = pred_outputs, references = references)['score']
        # torch.unsqueeze(trg_batchT,1).tolist())
        
        self.log(
            "bleu_score",
            bleu_score,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return bleu_score

    def predict_step(self, batch, batch_idx):
        pred_output_ids = self.model.generate(input_ids = batch['input_id'], attention_mask = batch['attention_mask'], do_sample=False, max_length = 512)
        pred_outputs = self.tokenizer.batch_decode(pred_output_ids, skip_special_tokens = True)

        return pred_outputs
