#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from hacl.models.rsgs.init_value_models.craftingworld_init_value_net import CraftingWorldStateEncoder


class CraftingWorldValueNet(nn.Module):
    def __init__(self, h_dim=128, agg_method='max'):
        super().__init__()
        self.acts = []
        self.h_dim = h_dim
        self.agg_method = agg_method
        self.net = CraftingWorldStateEncoder(h_dim=h_dim, out_dim=h_dim, agg_method=self.agg_method)
        self.decoder = nn.Linear(h_dim, 1)
        self.is_zero = False

    def forward(self, x):
        # if not self.is_zero and torch.abs(self.decoder.weight.data).sum() < 1e-5:
        #     self.is_zero = True
        #     print('set a value net to zero')
        if self.is_zero:
            return torch.randn(*x.size()[:-1], device=x.device)
        y = self.net(x)
        y = self.decoder(y).squeeze(-1)
        return y

    def visualize(self, **kwargs):
        raise NotImplementedError()
