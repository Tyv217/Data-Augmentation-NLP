import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler, RandomSampler, BatchSampler
import pytorch_lightning as pl
import pytorch_lightning.metrics.functional as plfunc
from pytorch_lightning.loggers import TensorBoardLogger

class Encoder(pl.LightningModule):

    def __init__(self, input_size, embed_size, enc_hid_size, dec_hid_size, dropout):
        super().__init__()

        self.embedding = torch.nn.Embedding(input_size, embed_size)
        self.gru = torch.nn.GRU(embed_size, enc_hid_size, bidirectional = True)
        self.fc = torch.nn.Linear(enc_hid_size * 2, dec_hid_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, input_lengths):
        embedded = self.dropout(self.embedding(input))
        packed_embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths.to('cpu'))

        packed_outputs, hidden = self.gru(packed_embedded)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_outputs)

        #hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        #outputs are always from the last layer
        
        #hidden [-2, :, : ] is the last of the forwards RNN 
        #hidden [-1, :, : ] is the last of the backwards RNN
        
        #initial decoder hidden is final hidden state of the forwards and backwards 
        #  encoder RNNs fed through a linear layer
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))

        return outputs, hidden

class Attention(pl.LightningModule):
    def __init__(self, enc_hid_size, dec_hid_size):
        super().__init__()

        # Hidden state of forward & backward of encoder + hidden state of decoder -> hidden state of decoder
        self.attn = nn.Linear((enc_hid_size * 2) + dec_hid_size, dec_hid_size)
        self.v = nn.Linear(enc_hid_size, 1, bias = False)

    def forward(self, hidden, encoder_outputs, mask):
        batch_size = encoder_outputs.shape[1]
        input_lengths = encoder_outputs.shape[0]
    
        hidden = hidden.unsqueeze(1).repeat(1, input_lengths, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2)))
        attention = self.v(energy).squeeze(2)
        attention = attention.masked_fill(mask == 0, -1e10)
        
        return torch.nn.functional.softmax(attention, dim = 1)

class Decoder(pl.LightningModule):
    def __init__(self, output_size, embed_size, enc_hid_size, dec_hid_size, dropout, attention):
        super().__init__()

        self.output_size = output_size
        self.attention = attention
        
        self.embedding = nn.Embedding(output_size, embed_size)
        self.gru = nn.GRU((enc_hid_size * 2) + embed_size, dec_hid_size)
        self.fc = nn.Linear((enc_hid_size * 2) + dec_hid_size + embed_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs, mask):
        input = input.unsqueeze(0)
        
        embedded = self.dropout(self.embedding(input))
        
        attn = self.attention(hidden, encoder_outputs, mask)
        attn = attn.unsqueeze(1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted_encoder_outputs = torch.bmm(attn, encoder_outputs)
        weighted_encoder_outputs = weighted_encoder_outputs.permute(1, 0, 2)
        
        gru_input = torch.cat((embedded, weighted_encoder_outputs), dim = 2)
        
        output, hidden = self.gru(gru_input, hidden.unsqueeze(0))
        
        #output = [seq len, batch size, dec hid dim * n directions]
        #hidden = [n layers * n directions, batch size, dec hid dim]
        
        #seq len, n layers and n directions will always be 1 in this decoder, therefore:
        #output = [1, batch size, dec hid dim]
        #hidden = [1, batch size, dec hid dim]
        #this also means that output == hidden
        assert (output == hidden).all()
        
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted_encoder_outputs = weighted_encoder_outputs.squeeze(0)
        
        prediction = self.fc(torch.cat((output, weighted, embedded), dim = 1))

        return prediction, hidden.squeeze(0), attn.squeeze(1)
        
class Seq2SeqTranslator(pl.LightningModule):
    def __init__(self, input_vocab_size, output_vocab_size, embed_size, hidden_size, dropout, input_padding_index):
        super().__init__()
        self.input_size = input_vocab_size
        self.output_size = output_vocab_size
        self.enc_emb_size = embed_size
        self.dec_emb_size = embed_size
        self.enc_hid_size = hidden_size
        self.dec_hid_size = hidden_size
        self.enc_dropout = dropout
        self.dec_dropout = dropout
        self.input_padding_index = input_padding_index

        self.save_hyperparameters()

        self.loss = torch.nn.CrossEntropyLoss(ignore_index = self.input_padding_index)
        self.learning_rate = 0.0005
        self.max_epochs = 50

        self.encoder = Encoder(
            self.input_size, 
            self.enc_emb_size, 
            self.enc_hid_size, 
            self.dec_hid_size, 
            self.enc_dropout
        )

        self.attention = Attention(self.enc_hid_size, self.dec_hid_size)

        self.decoder = Decoder(
            self.output_size, 
            self.dec_emb_size, 
            self.enc_hid_size, 
            self.dec_hid_size, 
            self.dec_dropout, 
            self.attention
        )

        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                torch.nn.init.normal_(param.data, mean=0, std=0.01)
            else:
                torch.nn.init.constant_(param.data, 0)

    def create_mask(self, input_):
        mask = (input_ != self.input_padding_index).permute(1, 0)
        return mask
    
    def forward(self, input_, input_lengths, training, teaching_forcing_ratio = 0.5):
        batch_size = input_.shape[1]
        training_size = training.shape[0]
        training_vocab_size = self.decoder.output_size

        decoder_outputs = torch.zeros(training_size, batch_size, training_vocab_size)

        encoder_outputs, hidden = self.encoder(input_, input_lengths)

        decoder_input = training[0,:]

        mask = self.create_mask(input_)

        for t in range(1, training_length):
            
            decoder_output, hidden, _ = self.decoder(decoder_input, hidden, encoder_outputs, mask)
            
            #place predictions in a tensor holding predictions for each token
            decoder_outputs[t] = decoder_output
            
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            #get the highest predicted token from our predictions
            top_predicted_token = decoder_output.argmax(1) 
            
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            decoder_input = training[t] if teacher_force else top_predicted_token
            
        return decoder_outputs

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr = self.learning_rate)
        lr_scheduler = {
            "scheduler": optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr = self.learning_rate,
                steps_per_epoch = int(len(self.train_dataloader())),
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

    def training_step(batch, batch_idx):
        input_batch, training_batch = batch
        input_, input_lengths = input_batch

        output = self.forward(input_, input_lengths, training_batch)

        output_dim = output.shape[-1]
        output = output[1:].view(-1, self.output_size)
        training = training[1:].view(-1)

        loss = self.loss(output, training)

        self.log(
            "training_loss",
            loss.item(),
            on_step = True,
            on_epoch = True,
            prog_bar = True,
            logger = rue,
        )

        return loss

    def validation_step(batch, batch_idx):
        input_batch, training_batch = batch
        input_, input_lengths = input_batch

        output = self.forward(input_, input_lengths, training_batch, 0)

        output_dim = output.shape[-1]
        output = output[1:].view(-1, self.output_size)
        training = training[1:].view(-1)

        loss = self.loss(output, training)

        self.log(
            "validation_loss",
            loss.item(),
            on_step = True,
            on_epoch = True,
            prog_bar = True,
            logger = rue,
        )

        pred_seq = outputs[1:].argmax(2)

        # change layout: seq_len * batch_size -> batch_size * seq_len
        pred_seq = pred_seq.T

        # compere list of predicted ids for all sequences in a batch to targets
        acc = plfunc.accuracy(pred_seq.reshape(-1), training_batch.reshape(-1))

        # need to cast to list of predicted sequences (as list of token ids)   [ [seq1_tok1, seq1_tok2, ...seq1_tokN],..., [seqK_tok1, seqK_tok2, ...seqK_tokZ]]
        predicted_ids = pred_seq.tolist()

        # need to add additional dim to each target reference sequence in order to
        # convert to format needed by bleu_score function [ seq1=[ [reference1], [reference2] ], seq2=[ [reference1] ] ]
        target_ids = torch.unsqueeze(trg_batch, 1).tolist()

        # bleu score needs two arguments
        # first: predicted_ids - list of predicted sequences as a list of predicted ids
        # second: target_ids - list of references (can be many, list)
        bleu_score = plfunc.nlp.bleu_score(predicted_ids, target_ids, n_gram=3).to(
            self.device
        )  # torch.unsqueeze(trg_batchT,1).tolist())

        return loss, acc, bleu_score

    



