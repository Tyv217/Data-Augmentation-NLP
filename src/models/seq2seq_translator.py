import torch, random
import pytorch_lightning as pl
import torchmetrics.functional as plfunc
from transformers import T5ForConditionalGeneration

class Seq2SeqTranslator(pl.LightningModule):
    def __init__(self, tokenizer, steps_per_epoch):
        super().__init__()

        self.learning_rate = 3e-4
        self.max_epochs = 50
        self.tokenizer = tokenizer
        self.model = T5ForConditionalGeneration.from_pretrained('t5-base')
        self.steps_per_epoch = steps_per_epoch
        self.model.init_weights()
        self.model.resize_token_embeddings(len(tokenizer))
    
    def forward(self, input_id, attention_mask, label):
        return self.model(input_ids = input_id, attention_mask = attention_mask, labels = label)

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
        )

        return loss

    def validation_step(self, batch, batch_idx):
        pred_output_ids = self.model.generate(input_ids = batch['input_id'], attention_mask = batch['attention_mask'], do_sample=False)
        pred_outputs = [self.tokenizer.decode(output_id, skip_special_tokens = True, clean_up_tokenization_spaces = True) for output_id in pred_output_ids]
        label_ids = batch['label']
        label_ids[label_ids == -100] = 0
        labels = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        targets = [[label] for label in labels]
        # bleu score needs two arguments
        # first: predicted_ids - list of predicted sequences as a list of predicted ids
        # second: target_ids - list of references (can be many, list)
        import pdb
        pdb.set_trace()
        bleu_score = plfunc.bleu_score(pred_outputs, targets, n_gram = 3).to(
            self.device
        )  # torch.unsqueeze(trg_batchT,1).tolist())

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