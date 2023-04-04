import torch, random, evaluate
import pytorch_lightning as pl
import torchmetrics.functional as plfunc
from transformers import AutoConfig, T5ForConditionalGeneration, T5Model
from datasets import load_metric

class Seq2SeqTranslator(pl.LightningModule):
    def __init__(self, model_name, max_epochs, tokenizer, steps_per_epoch, pretrain, augmentors = [], learning_rate = 1e-4):
        super().__init__()
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.tokenizer = tokenizer
        if pretrain:
            self.model = T5Model.from_pretrained()
        else:
            self.config = AutoConfig.from_pretrained(model_name)
            self.model = T5ForConditionalGeneration(self.config)
        self.steps_per_epoch = steps_per_epoch
        self.model.resize_token_embeddings(len(tokenizer))
        self.augmentors = augmentors
        
        num_params = sum(p.numel() for p in self.model.parameters())

        print(num_params)
    
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
        input = batch['input_id']
        attention_mask = batch['attention_mask']
        label = batch['label']
        for augmentor in self.augmentors:
            input, attention_mask, label = augmentor.augment_dataset(input, attention_mask, label)
            
        loss = self.forward(input, attention_mask, label).loss

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
        input = batch['input_id']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        output = self.forward(input, attention_mask, labels)
        loss = output.loss
        logits = output.logits
        pred_output_ids = torch.argmax(logits, axis = 2)
        pred_outputs = self.tokenizer.batch_decode(pred_output_ids, skip_special_tokens = True)
        labels[labels == -100] = 0
        labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        references = [[label] for label in labels]
        # bleu score needs two arguments
        # first: predicted_ids - list of predicted sequences as a list of predicted ids
        # second: target_ids - list of references (can be many, list)
        
        bleu_metric = evaluate.load("sacrebleu")
        bleu_score = bleu_metric.compute(predictions = pred_outputs, references = references)['score']
        # torch.unsqueeze(trg_batchT,1).tolist())
        
        self.log(
            "validation_bleu",
            bleu_score,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        self.log(
            "validation_loss",
            loss.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return loss, bleu_score

    def test_step(self, batch, batch_idx):
        input = batch['input_id']
        attention_mask = batch['attention_mask']
        pred_output_ids = self.model.generate(input_ids = input, attention_mask = attention_mask, do_sample=False, max_length = 512)
        pred_outputs = self.tokenizer.batch_decode(pred_output_ids, skip_special_tokens = True)
        labels = batch['label']
        labels[labels == -100] = 0
        labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        references = [[label] for label in labels]
        # bleu score needs two arguments
        # first: predicted_ids - list of predicted sequences as a list of predicted ids
        # second: target_ids - list of references (can be many, list)
        
        bleu_metric = evaluate.load("sacrebleu")
        bleu_score = bleu_metric.compute(predictions = pred_outputs, references = references)['score']
        # torch.unsqueeze(trg_batchT,1).tolist())
        
        self.log(
            "test_bleu",
            bleu_score,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return bleu_score