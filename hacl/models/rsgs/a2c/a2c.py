#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from jactorch.nn import TorchApplyRecorderMixin
from torch import nn


class A2C(TorchApplyRecorderMixin):
    def __init__(
        self,
        in_dim,
        out_dim,
        h_dim=256,
        input_encoder_gen=lambda: None,
        label_encoder_gen=lambda: None,
        use_lstm=False,
        continuous=False,
    ):
        super().__init__()
        self.input_encoder = input_encoder_gen()
        self.in_dim = in_dim if in_dim is not None else self.input_encoder.out_dim
        self.label_encoder = label_encoder_gen()
        self.out_dim = out_dim
        self.h_dim = h_dim
        self.use_lstm = use_lstm
        self.continuous = continuous
        if use_lstm:
            self.lstm = nn.LSTM(self.in_dim + self.label_encoder.out_dim, h_dim, batch_first=True)
            self.base = nn.Linear(h_dim, h_dim)
        else:
            self.base = nn.Sequential(
                nn.Linear(self.in_dim + self.label_encoder.out_dim, h_dim), nn.ReLU(), nn.Linear(h_dim, h_dim)
            )

        if continuous:
            self.mu = nn.Sequential(nn.Linear(h_dim, out_dim), nn.Tanh())
            self.var = nn.Sequential(nn.Linear(h_dim, out_dim), nn.Sigmoid())
            self.value = nn.Linear(h_dim, 1)
        else:
            self.actor = nn.Sequential(
                nn.Linear(h_dim, out_dim),
            )
            self.value = nn.Linear(h_dim, 1)

    def get_zero_lstm_state(self, x):
        h = torch.zeros(1, x.size(0), self.h_dim, device=x.device)
        c = torch.zeros(1, x.size(0), self.h_dim, device=x.device)
        return h, c

    def forward(self, x, labels=None, embeddings=None, hidden=None):
        """
        Args:
            x: [batch * length * state_dim]
            labels: list of labels of length batch
            embeddings: [batch * embedding_dim]
            hidden: (h, c) each [batch * 1 * hidden_dim]

        Returns: action_score[batch * length * n_actions], value[batch * length](, hidden) if not normal_action
                 (mu[batch * length * action_dim], var[batch * length * action_dim]), value[batch * length] (, hidden) if normal_action

        """
        batch, length = x.size()[:2]
        feature = self.input_encoder(x.reshape(batch * length, *x.size()[2:])).view(batch, length, -1)
        if labels is not None:
            assert embeddings is None
            embeddings = self.label_encoder(labels)
        if embeddings is not None:
            feature = torch.cat((feature, embeddings.unsqueeze(1).repeat(1, feature.size(1), 1)), dim=2)
        if self.use_lstm:
            assert hidden is not None
            feature, hidden = self.lstm(feature, hidden)
        else:
            assert hidden is None
        feature = self.base(feature)
        value = self.value(feature).squeeze(-1)
        if self.continuous:
            mu = self.mu(feature)
            var = torch.clip(self.var(feature), 1e-5, 1)
            if self.use_lstm:
                return (mu, var), value, hidden
            else:
                return (mu, var), value
        else:
            score = self.actor(feature)
            if self.use_lstm:
                return score, value, hidden
            else:
                return score, value
