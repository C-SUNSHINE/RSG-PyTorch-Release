#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from hacl.models.rsgs.evaluators.evaluator import Evaluator
from hacl.models.rsgs.encoders.traj_encoder import TrajEncoder
from hacl.models.rsgs.encoders.label_encoder import LabelEncoder


class LSTMEvaluator(Evaluator):
    _EXTRA_DICT_KEY = [
        'env_name',
    ]

    def __init__(self, env_name='gridworld', env_args=None, add_labels=None, h_dim=128):
        super().__init__()
        if env_args is None:
            env_args = dict()
        self.env_args = env_args
        self.env_name = env_name
        self.h_dim = h_dim
        self.traj_encoder = TrajEncoder(env_name, env_args, s_dim=h_dim, h_dim=h_dim)
        self.traj_lstm = nn.LSTM(input_size=h_dim, hidden_size=h_dim, bidirectional=True, batch_first=True)

        self.label_encoder = LabelEncoder(add_labels, h_dim=h_dim)

        self.decoder = nn.Sequential(nn.Linear(h_dim * 4, h_dim), nn.Sigmoid(), nn.Linear(h_dim, 1))
        # init_weights(self)

    def get_training_parameters(self):
        return self.parameters()

    @property
    def qvalue_based(self):
        return False

    def extra_state_dict(self):
        return {key: getattr(self, key) for key in self._EXTRA_DICT_KEY}

    def load_extra_state_dict(self, extra_dict):
        for key in self._EXTRA_DICT_KEY:
            setattr(self, key, extra_dict[key])

    def forward(self, trajs, labels=None, training=False, progress=None, **kwargs):
        n = len(trajs)
        # Compute traj_features
        # trajs = [(traj[0][-1:], tuple()) for traj in trajs] # TODO restore it
        seqs = self.traj_encoder(trajs, **kwargs)
        # print('seqs=', seqs)
        # print('traj=', (trajs[0][0][-1:], tuple()))
        # input()
        lens = [int(seq.size(0)) for seq in seqs]
        max_len = max(lens)
        seqs_tensor = torch.stack([F.pad(seq, (0, 0, 0, max_len - seq.size(0))) for seq in seqs], dim=0)
        packed = nn.utils.rnn.pack_padded_sequence(seqs_tensor, lens, batch_first=True, enforce_sorted=False)
        h, _ = self.traj_lstm(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True)
        # h: size(n, max_length, 2*self.h_dim)
        assert h.size() == torch.Size((n, max_len, 2 * self.h_dim))
        traj_features = h.sum(1) / torch.tensor(lens, device=h.device).view(n, 1)

        # Compute label_features
        label_features = self.label_encoder(labels)

        # Compute score
        res = self.decoder(
            torch.cat(
                (
                    traj_features.view(n, 1, -1).repeat(1, len(labels), 1),
                    label_features.view(1, len(labels), -1).repeat(n, 1, 1),
                ),
                dim=2,
            )
        ).view(n, len(labels))
        return res

    def train(self, mode=True):
        super(LSTMEvaluator, self).train(mode=mode)
