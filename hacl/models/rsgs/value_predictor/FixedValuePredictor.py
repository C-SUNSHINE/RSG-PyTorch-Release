#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from .ValuePredictorBase import ValuePredictor


class FixedValuePredictor(ValuePredictor):
    def __init__(self, n_states):
        super().__init__(n_states)
        self.values = nn.Parameter(torch.zeros(n_states), requires_grad=True)

    def forward(self):
        return self.values

    def get_values(self):
        return self()
