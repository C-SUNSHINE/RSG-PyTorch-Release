#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import torch
import torch.nn.functional as F
from jactorch.nn import TorchApplyRecorderMixin
from torch import nn


class DQN(TorchApplyRecorderMixin):
    def __init__(
        self, in_dim, out_dim, h_dim=256,
        input_encoder_gen=lambda: None,
        label_encoder_gen=lambda: None,
        use_lstm=False,
        activation='linear',
        normalize=False,
        log_normalize=False,
        add_dim=0,
        playroom_add=False,
    ):
        super().__init__()
        self.input_encoder = input_encoder_gen()
        self.in_dim = in_dim if in_dim is not None else self.input_encoder.out_dim
        self.label_encoder = label_encoder_gen()
        self.out_dim = out_dim
        self.h_dim = h_dim
        self.use_lstm = use_lstm
        self.add_dim = add_dim
        self.playroom_add = playroom_add
        if use_lstm:
            self.lstm = nn.LSTM(self.in_dim + self.add_dim + self.label_encoder.out_dim, h_dim, batch_first=True)

        self.decoder = (
            nn.Sequential(
                nn.Linear(self.in_dim + self.add_dim + self.label_encoder.out_dim, h_dim),
                nn.ReLU(),
                nn.Linear(h_dim, h_dim),
                nn.ReLU(),
                nn.Linear(h_dim, out_dim),
            )
            if not use_lstm
            else nn.Sequential(nn.Linear(h_dim, h_dim), nn.ReLU(), nn.Linear(h_dim, out_dim))
        )
        if activation == 'linear':
            self.activation = lambda x: x
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'logsigmoid':
            self.activation = nn.LogSigmoid()
        else:
            raise ValueError('Invalid activation %s' % activation)
        self.normalize = normalize
        self.log_normalize = log_normalize
        assert not (normalize and log_normalize)

    def get_zero_lstm_state(self, x):
        h = torch.zeros(1, x.size(0), self.h_dim, device=x.device)
        c = torch.zeros(1, x.size(0), self.h_dim, device=x.device)
        return h, c

    def forward(self, x, labels=None, embeddings=None, hidden=None, add=None):
        """
        Args:
            x: [batch * length * state_dim]
            labels: list of labels of length batch
            embeddings: [batch * embedding_dim]
            hidden: (h, c) each [batch * 1 * hidden_dim]
            add: [batch * length * add_dim]

        Returns: action_score[batch * length * n_actions] (, hidden)

        """
        if self.playroom_add:
            assert add.size(0) == x.size(0) and add.size(1) == x.size(1) and add.size(2) == 3
            add = add + x[:, :, :3]
            add2 = add[:, :, 2:3]
            add2 = add2 + add2.lt(math.pi).type(torch.float) * math.pi
            add2 = add2 - add2.ge(math.pi).type(torch.float) * math.pi
            add = torch.cat((add[:, :, :2], add2), dim=2)
            x = torch.cat((add, x[:, :, 3:]), dim=2)
            add = None
        batch, length = x.size()[:2]
        feature = self.input_encoder(x.reshape(batch * length, *x.size()[2:])).view(batch, length, -1)
        if labels is not None:
            assert embeddings is None
            embeddings = self.label_encoder(labels)
        if embeddings is not None:
            feature = torch.cat((feature, embeddings.unsqueeze(1).repeat(1, feature.size(1), 1)), dim=2)
        if self.add_dim != 0:
            feature = torch.cat((feature, add), dim=2)
        else:
            assert add is None
        if self.use_lstm:
            assert hidden is not None
            feature, hidden = self.lstm(feature, hidden)
            res = self.activation(self.decoder(feature))
            if self.normalize:
                res = F.softmax(res, dim=-1)
            elif self.log_normalize:
                res = F.log_softmax(res, dim=-1)
            return res, hidden
        else:
            assert hidden is None
            res = self.activation(self.decoder(feature))
            if self.normalize:
                res = F.softmax(res, dim=-1)
            elif self.log_normalize:
                res = F.log_softmax(res, dim=-1)
            return res
