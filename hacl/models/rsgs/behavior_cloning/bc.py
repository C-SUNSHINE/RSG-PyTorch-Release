#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from jactorch.nn import TorchApplyRecorderMixin
from torch import nn


class BehaviorNet(TorchApplyRecorderMixin):
    def __init__(
        self,
        in_dim,
        out_dim,
        h_dim=256,
        input_encoder_gen=lambda: None,
        emb_dim=0,
        use_lstm=False,
        continuous=False,
    ):
        super().__init__()
        self.input_encoder = input_encoder_gen()
        self.in_dim = in_dim if in_dim is not None else self.input_encoder.out_dim
        self.out_dim = out_dim
        self.h_dim = h_dim
        self.use_lstm = use_lstm
        self.continuous = continuous
        self.emb_dim = emb_dim
        if use_lstm:
            self.lstm = nn.LSTM(self.in_dim + emb_dim, h_dim, batch_first=True)
            self.base = nn.Linear(h_dim, h_dim)
        else:
            self.base = nn.Sequential(
                nn.Linear(self.in_dim + emb_dim, h_dim), nn.ReLU(), nn.Linear(h_dim, h_dim)
            )

        if continuous:
            self.mu = nn.Sequential(nn.Linear(h_dim, out_dim), nn.Tanh())
            self.var = nn.Sequential(nn.Linear(h_dim, out_dim), nn.Sigmoid())
        else:
            self.actor = nn.Sequential(
                nn.Linear(h_dim, out_dim),
                nn.LogSoftmax(dim=-1)
            )
        self.stop = nn.Sequential(nn.Linear(h_dim, 2), nn.LogSoftmax(dim=-1))

    def get_zero_lstm_state(self, x):
        h = torch.zeros(1, x.size(0), self.h_dim, device=x.device)
        c = torch.zeros(1, x.size(0), self.h_dim, device=x.device)
        return h, c

    def forward(self, x, hidden=None, embeddings=None):
        """
        Args:
            x: [batch * length * state_dim]
            hidden: (h, c) each [batch * 1 * hidden_dim]
            embeddings: [batch * emb_dim]

        Returns: action_score[batch * length * n_actions], value[batch * length](, hidden) if not normal_action
                 (mu[batch * length * action_dim], var[batch * length * action_dim]), value[batch * length] (, hidden) if normal_action

        """
        batch, length = x.size()[:2]
        feature = self.input_encoder(x.reshape(batch * length, *x.size()[2:])).view(batch, length, -1)
        if self.emb_dim != 0:
            assert embeddings.size(1) == self.emb_dim
            feature = torch.cat((feature, embeddings.unsqueeze(1).repeat(1, feature.size(1), 1)), dim=2)
        else:
            assert embeddings is None
        if self.use_lstm:
            assert hidden is not None
            feature, hidden = self.lstm(feature, hidden)
        else:
            assert hidden is None
        feature = self.base(feature)
        stop = self.stop(feature)
        if self.continuous:
            mu = self.mu(feature)
            var = self.var(feature)
            if self.use_lstm:
                return (mu, var), stop, hidden
            else:
                return (mu, var), stop
        else:
            score = self.actor(feature)
            if self.use_lstm:
                return score, stop, hidden
            else:
                return score, stop
