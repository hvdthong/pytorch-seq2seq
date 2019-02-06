import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator

import spacy

import random
import math
import os

import nl_core_news_sm
import en_core_web_sm


def tokenize_de(text):
    """
    Tokenizes German text from a string into a list of strings and reverses it
    """
    return [tok.text for tok in spacy_de.tokenizer(text)][::-1]


def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [sent len, batch size]

        embedded = self.dropout(self.embedding(src.cuda() if torch.cuda.is_available() else src))
        # embedded = self.dropout(self.embedding(src))

        # embedded = [sent len, batch size, emb dim]

        outputs, (hidden, cell) = self.rnn(embedded)

        # outputs = [sent len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # outputs are always from the top hidden layer

        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)

        self.out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # n directions in the decoder will both always be 1, therefore:
        # hidden = [n layers, batch size, hid dim]
        # context = [n layers, batch size, hid dim]

        input = input.unsqueeze(0)

        # input = [1, batch size]
        embedded = self.dropout(self.embedding(input.cuda() if torch.cuda.is_available() else input))
        # embedded = self.dropout(self.embedding(input))

        # embedded = [1, batch size, emb dim]

        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        # output = [sent len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # sent len and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, hid dim]
        # hidden = [n layers, batch size, hid dim]
        # cell = [n layers, batch size, hid dim]

        check_output = output.squeeze(0)

        prediction = self.out(output.squeeze(0))

        # prediction = [batch size, output dim]

        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim, "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = [sent len, batch size]
        # trg = [sent len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)

        # first input to the decoder is the <sos> tokens
        input = trg[0, :]

        for t in range(1, max_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            input = (trg[t] if teacher_force else top1)

        return outputs


def train_model(model, iterator, optimizer, criterion, clip):
    model.train()

    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()

        output = model(src, trg)

        # trg = [sent len, batch size]
        # output = [sent len, batch size, output dim]

        # reshape to:
        # trg = [(sent len - 1) * batch size]
        # output = [(sent len - 1) * batch size, output dim]

        out_src, out_trg = output[1:].view(-1, output.shape[2]), trg[1:].view(-1)
        loss = criterion(out_src.cuda() if torch.cuda.is_available() else out_src,
                         out_trg.cuda() if torch.cuda.is_available() else out_trg)

        # loss = criterion(output[1:].view(-1, output.shape[2]), trg[1:].view(-1))
        print('Loss value:', torch.tensor(loss).float())
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            output = model(src, trg, 0)  # turn off teacher forcing

            out_src, out_trg = output[1:].view(-1, output.shape[2]), trg[1:].view(-1)
            loss = criterion(out_src.cuda() if torch.cuda.is_available() else out_src,
                             out_trg.cuda() if torch.cuda.is_available() else out_trg)

            # loss = criterion(output[1:].view(-1, output.shape[2]), trg[1:].view(-1))

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


if __name__ == '__main__':
    SEED = 1234

    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    Multi30k.download('data')

    spacy_de = nl_core_news_sm.load()
    spacy_en = en_core_web_sm.load()

    SRC = Field(tokenize=tokenize_de, init_token='<sos>', eos_token='<eos>', lower=True)
    TRG = Field(tokenize=tokenize_en, init_token='<sos>', eos_token='<eos>', lower=True)

    train, valid, test = TranslationDataset.splits(
        path='./data/multi30k/',
        exts=['.de', '.en'],
        fields=[('src', SRC), ('trg', TRG)],
        train='train',
        validation='val',
        test='test2016')

    print("Number of training examples:", len(train.examples))
    print("Number of validation examples:", len(valid.examples))
    print("Number of testing examples:", len(test.examples))
    print(vars(train.examples[0]))

    SRC.build_vocab(train, min_freq=2)
    TRG.build_vocab(train, min_freq=2)
    print("Unique tokens in source (de) vocabulary:", len(SRC.vocab))
    print("Unique tokens in target (en) vocabulary:", len(TRG.vocab))

    BATCH_SIZE = 128
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train, valid, test), batch_size=BATCH_SIZE, repeat=False)

    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 2
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

    model = Seq2Seq(enc, dec, device).to(device)
    optimizer = optim.Adam(model.parameters())

    pad_idx = TRG.vocab.stoi['<pad>']

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    N_EPOCHS = 25
    CLIP = 10
    SAVE_DIR = 'save'
    MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'tut1_model.pt')

    best_valid_loss = float('inf')

    if not os.path.isdir('SAVE_DIR'):
        os.makedirs('SAVE_DIR')

    for epoch in range(N_EPOCHS):

        train_loss = train_model(model, train_iterator, optimizer, criterion, CLIP)
        valid_loss = evaluate(model, valid_iterator, criterion)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

        print('| Epoch: | Train Loss: | Train PPL: '
              '| Val. Loss: | Val. PPL: |', epoch + 1, train_loss, math.exp(train_loss), valid_loss,
              math.exp(valid_loss))

    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    test_loss = evaluate(model, test_iterator, criterion)
    print('| Test Loss:  | Test PPL: |', test_loss, math.exp(test_loss))
