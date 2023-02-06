import torch.nn as nn
import torch

class MenmaEncoder(nn.Module):
    def __init__(self, vocab_size, embedded_size, num_hidden, num_layers, dropout = 0) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedded_size)
        self.rnn = nn.GRU(embedded_size, num_hidden, num_layers, dropout=dropout)

    def forward(self, X: torch.Tensor):
        X = self.embedding(X)
        # (num_steps, batch_size, embedded_size)
        X = X.permute(1, 0, 2)
        # output (num_steps, batch_size, num_hiddens)
        output, state = self.rnn(X)
        # state (num_layers, batch_size, num_hiddens)
        return output, state

class MenmaDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers,
                          dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs):
        return enc_outputs[1]

    def forward(self, X, state):
        # X (num_steps, batch_size, embedded_size)
        X = self.embedding(X).permute(1, 0, 2)
        # 广播context，使其具有与X相同的num_steps
        context = state[-1].repeat(X.shape[0], 1, 1)
        # X_and_context (num_steps, batch_size, embedded_size + num_hiddens)
        X_and_context = torch.cat((X, context), 2)
        
        output, state = self.rnn(X_and_context, state)
        output = self.dense(output).permute(1, 0, 2)
        # output的形状:(batch_size,num_steps,vocab_size)
        # state的形状:(num_layers,batch_size,num_hiddens)
        return output, state


class MenmaEncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X):
        enc_outputs = self.encoder(enc_X)
        dec_state = self.decoder.init_state(enc_outputs)
        return self.decoder(dec_X, dec_state)