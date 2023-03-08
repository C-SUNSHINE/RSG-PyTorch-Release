#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from hacl.nn.utils import init_weights
from hacl.models.rsgs.encoders.craftingworld_state_encoder import CraftingWorldStateEncoder


class CraftingWorldInitValueNet(nn.Module):
    def __init__(self, h_dim=128, agg_method='max', use_not_goal=False, **kwargs):
        super().__init__()
        self.acts = []
        self.h_dim = h_dim
        self.agg_method = agg_method
        self.net = nn.ModuleDict()
        self.use_not_goal = use_not_goal
        print('CraftingWorldInitValueNet, h_dim = %d, agg_method=%s' % (h_dim, agg_method))

    def __contains__(self, item):
        return item in self.acts

    def add_act(self, act):
        self.acts.append(act)
        if not (self.use_not_goal and act.startswith('pre_')):
            self.net.add_module(
                act,
                CraftingWorldStateEncoder(self.h_dim, 1, activation='logsigmoid', agg_method=self.agg_method,
                                          add_negative=self.use_not_goal)
            )
            init_weights(self.net[act])

    def forward(self, act, x, ignore_pre=False):
        assert act in self.acts
        if self.use_not_goal and act.startswith('pre_'):
            eff_act = 'eff_' + act[4:]
            res = self.net[eff_act](x, use_neg=True).squeeze(-1)
            if ignore_pre:
                res = torch.zeros_like(res)
                # print('ignored')
            return res
        return self.net[act](x).squeeze(-1)
