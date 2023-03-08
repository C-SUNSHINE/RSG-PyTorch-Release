#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from torch import nn


class ValuePredictor(nn.Module):
    def __init__(self, n_states):
        super().__init__()
        self.n_states = n_states
        self.optimizer = None

    def get_values(self, *args, **kwargs):
        raise NotImplementedError
