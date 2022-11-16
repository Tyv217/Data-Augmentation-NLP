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
    def __init__(self, encoder, decoder, input_padding_index, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.input_padding_index = input_padding_index
        self.device = device

    def create_mask(self, input_):
        mask = (input_ != self.input_padding_index).permute(1, 0)
        return mask
    
    def forward(self, input_, input_lengths, training, teaching_forcing_ratio = 0.5):
        batch_size = input_.shape[1]
        training_size = training.shape[0]
        training_vocab_size = self.decoder.output_size

        outputs = torch.zeros(training_size, batch_size, training_vocab_size).to(self.device)

        encoder_outputs, hidden = self.encoder(input_, input_lengths)

        decoder_input = training[0,:]

        mask = self.create_mask(input_)

        for t in range(1, training_length):
            
            output, hidden, _ = self.decoder(decoder_input, hidden, encoder_outputs, mask)
            
            #place predictions in a tensor holding predictions for each token
            outputs[t] = output
            
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            #get the highest predicted token from our predictions
            top_predicted_token = output.argmax(1) 
            
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            decoder_input = training[t] if teacher_force else top_predicted_token
            
        return outputs