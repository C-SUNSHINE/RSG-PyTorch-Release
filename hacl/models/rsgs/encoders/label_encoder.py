#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn
import torch.nn.functional as F
from jactorch.nn import TorchApplyRecorderMixin


class LabelEncoder(TorchApplyRecorderMixin):
    _EXTRA_DICT_KEY = ['words', 'word2index', 'h_dim']
    SYMBOLS = '&|>()'

    def __init__(self, labels, h_dim=128, out_dim=None):
        super().__init__()
        self.h_dim = h_dim
        self.words = self.get_words_from_labels(labels)
        self.word2index = {w: i for i, w in enumerate(self.words)}
        self.word_embedding = nn.Parameter(torch.eye(len(self.words)), requires_grad=False)  # nn.Embedding(len(self.words), h_dim)
        self.lang_lstm = nn.LSTM(input_size=len(self.words), hidden_size=h_dim, bidirectional=True, batch_first=True)
        if out_dim is None:
            self.decoder = None
            self.out_dim = 2 * self.h_dim
        else:
            self.decoder = nn.Linear(2 * self.h_dim, out_dim)
            self.out_dim = out_dim

    def get_words_from_labels(self, labels):
        words = list(self.SYMBOLS)
        for label in labels:
            s = label
            for c in self.SYMBOLS:
                s = s.replace(c, ' ')
            for l in s.split(' '):
                if len(l) > 0 and l not in words:
                    words.append(l)
        words = list(sorted(set(words)))
        return words

    def extra_state_dict(self):
        return {key: getattr(self, key) for key in self._EXTRA_DICT_KEY}

    def load_extra_state_dict(self, extra_dict):
        for key in self._EXTRA_DICT_KEY:
            setattr(self, key, extra_dict[key])

    def label_to_word_idxs(self, label):
        s = label
        for c in self.SYMBOLS:
            s = s.replace(c, ' ')
        words = []
        ptr = 0
        while ptr < len(label):
            pos = s.find(' ', ptr)
            if pos == -1:
                words.append(label[ptr:])
                break
            else:
                if ptr < pos:
                    words.append(label[ptr:pos])
                words.append(label[pos: pos + 1])
                ptr = pos + 1
        return [self.word2index[word] for word in words]

    def forward(self, labels):
        assert not isinstance(labels, str)
        n = len(labels)
        # seqs = [ self.word_embedding(torch.LongTensor(self.label_to_word_idxs(label)).to(self.device)) for label in labels]
        seqs = [self.word_embedding.index_select(dim=0, index=torch.LongTensor(self.label_to_word_idxs(label)).to(self.device)) for label in labels]
        lens = [seq.size(0) for seq in seqs]
        max_len = max(lens)
        seqs_tensor = torch.stack([F.pad(seq, (0, 0, 0, max_len - seq.size(0))) for seq in seqs], dim=0)
        packed = nn.utils.rnn.pack_padded_sequence(seqs_tensor, lens, batch_first=True, enforce_sorted=False)
        h, _ = self.lang_lstm(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True)
        # h: size(n, max_length, 2*self.h_dim)
        assert h.size() == torch.Size((n, max_len, 2 * self.h_dim))
        res = h.sum(1) / torch.tensor(lens, device=h.device).view(n, 1)
        if self.decoder is not None:
            res = self.decoder(res)
        return res
