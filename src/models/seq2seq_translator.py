import torch, random
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler, RandomSampler, BatchSampler
import pytorch_lightning as pl
import torchmetrics.functional as plfunc
from torchnlp.encoders.text.static_tokenizer_encoder import StaticTokenizerEncoder
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torchnlp.encoders.text.text_encoder import TextEncoder
# from torchtext.legacy.data import Field, BucketIterator
import pdb

class Encoder(pl.LightningModule):

    def __init__(self, input_size, embed_size, enc_hid_size, dec_hid_size, dropout):
        super().__init__()
        
        self.embedding = torch.nn.Embedding(input_size, embed_size)
        self.gru = torch.nn.GRU(embed_size, enc_hid_size, num_layers = 1, bidirectional = True)
        self.fc = torch.nn.Linear(enc_hid_size * 2, dec_hid_size)
        self.dropout = torch.nn.Dropout(dropout, inplace = True)

    def forward(self, input, input_lengths):
        embedded = self.dropout(self.embedding(input))
        packed_embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths.cpu(), enforce_sorted = False)
        # embedded = [seq len, 32 = batch size, 256 = embedding feature size]

        packed_outputs, hidden = self.gru(packed_embedded)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_outputs, total_length = input.shape[0])

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
        self.attn = torch.nn.Linear((enc_hid_size * 2) + dec_hid_size, dec_hid_size)
        self.v = torch.nn.Linear(enc_hid_size, 1, bias = False)

    def forward(self, hidden, encoder_outputs, mask):
        # batch_size = encoder_outputs.shape[1]
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
        
        self.embedding = torch.nn.Embedding(output_size, embed_size)
        self.gru = torch.nn.GRU((enc_hid_size * 2) + embed_size, dec_hid_size)
        self.fc = torch.nn.Linear((enc_hid_size * 2) + dec_hid_size + embed_size, output_size)
        self.dropout = torch.nn.Dropout(dropout)

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
        
        prediction = self.fc(torch.cat((output, weighted_encoder_outputs, embedded), dim = 1))

        return prediction, hidden.squeeze(0), attn.squeeze(1)
        
class Seq2SeqTranslator(pl.LightningModule):
    def __init__(self, input_vocab_size, output_vocab_size, embed_size, hidden_size, dropout, input_padding_index, input_tokenizer, output_tokenizer):
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

        self.input_tokenizer = input_tokenizer
        self.output_tokenizer = output_tokenizer
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

        for t in range(1, training_size):
            
            decoder_output, hidden, _ = self.decoder(decoder_input, hidden, encoder_outputs, mask)
            
            #place predictions in a tensor holding predictions for each token
            decoder_outputs[t] = decoder_output
            
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teaching_forcing_ratio
            
            #get the highest predicted token from our predictions
            top_predicted_token = decoder_output.argmax(1) 
            
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            decoder_input = training[t] if teacher_force else top_predicted_token
            
        return decoder_outputs

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr = self.learning_rate)
        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr = self.learning_rate,
                steps_per_epoch = self.trainer.accumulate_grad_batches,
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
        torch.cuda.empty_cache()

        input_batch, target_batch = batch
        input_seq = input_batch["src"].transpose(0,1)
        input_lengths = input_batch["src_len"]

        target_seq = target_batch["trg"].transpose(0,1)

        outputs = self.forward(input_seq, input_lengths, target_seq)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        output = outputs[1:].view(-1, self.output_size).to(device)
        target = target_seq[1:].reshape(-1).to(device)

        loss = self.loss(output, target)

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
        torch.cuda.empty_cache()

        input_batch, target_batch = batch

        input_seq = input_batch["src"].transpose(0,1)
        input_lengths = input_batch["src_len"]

        target_seq = target_batch["trg"].transpose(0,1)

        outputs = self.forward(input_seq, input_lengths, target_seq, 0)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        output = outputs[1:].view(-1, self.output_size).to(device)
        target = target_seq[1:].reshape(-1).to(device)

        loss = self.loss(output, target)

        prediction = outputs[1:].argmax(2).T

        target_batch = target_seq[1:].T

        # acc = plfunc.accuracy(prediction.reshape(-1), target_batch.reshape(-1))

        predicted_seq = prediction.tolist()

        target_seq_bleu = torch.unsqueeze(target_batch, 1).tolist()

        predicted_seq_words = []
        target_seq_words = []
        for i in range(len(predicted_seq)):
            targets = []
            for target in target_seq_bleu[i]:
                targets.append(self.output_tokenizer.decode(target))
            target_seq_words.append(targets)
            predicted_seq_words.append(self.output_tokenizer.decode(predicted_seq[i]))
            
        # bleu score needs two arguments
        # first: predicted_ids - list of predicted sequences as a list of predicted ids
        # second: target_ids - list of references (can be many, list)
        # pdb.set_trace()
        bleu_score = plfunc.bleu_score(predicted_seq_words, target_seq_words, n_gram = 3).to(
            self.device
        )  # torch.unsqueeze(trg_batchT,1).tolist())

        self.log(
            "validation_loss",
            loss.item(),
            on_step = True,
            on_epoch = True,
            prog_bar = True,
            logger = True,
        )

        self.log(
            "val_bleu_idx",
            bleu_score,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return loss, bleu_score

class Seq2SeqDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32):
        super().__init__()

        self.dataset = load_dataset("iwslt2017", "iwslt2017-en-de")
        self.batch_size = batch_size

    def build_vocab(self):
        en_lines = []
        de_lines = []
        for line in self.dataset['train']:
            en_lines.append(line['translation']['en'])
            de_lines.append(line['translation']['de'])

        self.en_tokenizer = CustomTokenizerEncoder(en_lines, append_eos = True)
        self.de_tokenizer = CustomTokenizerEncoder(de_lines, append_eos = True)
        self.padding_index = self.en_tokenizer.padding_index
        
        self.input_vocab_size = self.en_tokenizer.vocab_size
        self.output_vocab_size = self.de_tokenizer.vocab_size

    def split_and_pad_data(self, data):
        en_index_lines = []
        de_index_lines = []
        en_lengths = []
        de_lengths = []

        for line in data:
            en_line = self.en_tokenizer.encode(line['translation']['en'])
            en_index_lines.append(en_line)
            en_lengths.append(len(en_line))
            de_line = self.de_tokenizer.encode(line['translation']['de'])
            de_index_lines.append(de_line)
            de_lengths.append(len(de_line))
        en_index_lines = pad_sequence(en_index_lines, batch_first = True)
        de_index_lines = pad_sequence(de_index_lines, batch_first = True)

        data_seq = []
        for i in range(len(en_index_lines)):
            if(en_lengths[i] > 0 and de_lengths[i] > 0):
                input = {'src': en_index_lines[i], 'src_len': en_lengths[i]}
                output = {'trg': de_index_lines[i], 'trg_len': de_lengths[i]}
                data_seq.append((input,output))
        return data_seq

    def setup(self, stage: str):
        self.build_vocab() 
        self.train_iterator = self.split_and_pad_data(self.dataset['train'])
        self.valid_iterator = self.split_and_pad_data(self.dataset['validation'])
        self.test_iterator = self.split_and_pad_data(self.dataset['test'])

    def train_dataloader(self):
        return DataLoader(self.train_iterator, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.valid_iterator, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_iterator, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.test_iterator, batch_size=self.batch_size)

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        pass


class CustomTokenizerEncoder(StaticTokenizerEncoder):
    def decode_index_to_token(self, index):
        if(index > self.vocab_size):
            return '<unk>'
        else:
            return self.index_to_token[index]

    def decode(self, encoded):
        """ Decodes a tensor into a sequence.

        Args:
            encoded (torch.Tensor): Encoded sequence.

        Returns:
            str: Sequence decoded from ``encoded``.
        """
        encoded = TextEncoder.decode(self, encoded)
        tokens = [self.decode_index_to_token(index) for index in encoded]
        return self.detokenize(tokens)