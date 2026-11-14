# -*- coding: utf-8 -*-
"""手写 LSTM 中文语言模型。"""
import torch
import torch.nn as nn


class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=80, hidden_dim=160, num_layers=1,
                 dropout=0.2, block_size=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.block_size = block_size
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        emb = self.embedding(x)
        out, hidden = self.lstm(emb, hidden)
        out = self.drop(out)
        logits = self.head(out)
        return logits, hidden

    def init_hidden(self, batch_size, device):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return (h0, c0)

    @property
    def device(self):
        return next(self.parameters()).device


def count_params(model):
    return sum(p.numel() for p in model.parameters())
