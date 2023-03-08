#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import torch
from torch import nn as nn




class InitValueExpressionWrapper(nn.Module):
    priority = {'&': 2, '|': 1}

    def __init__(self, net, *args, ptrajonly=False, **kwargs):
        super().__init__()
        print('InitValueExpressionWrapper,', 'ptrajonly=', ptrajonly)
        self.net = net
        self.init_value = net(*args, **kwargs)
        self.ptrajonly = ptrajonly
        # for label in reward_labels:
        #     s = label
        #     for c in "&|() ":
        #         s = s.replace(c, ' ')
        #     for l in s.split(' '):
        #         if len(l) > 0:
        #             self.atomic_labels.add(l)
        # self.init_value = net(self.atomic_labels, *args, **kwargs)

    def __contains__(self, item):
        return item in self.init_value

    def add_edge_labels(self, edge_labels):
        for label in edge_labels:
            s = label
            for c in "&|() ":
                s = s.replace(c, ' ')
            for l in s.split(' '):
                if len(l) > 0 and l not in self.init_value.acts:
                    self.init_value.add_act(l)

    def get_atomic_values(self, states):
        res = {}
        for al in self.init_value.acts:
            res[al] = self.init_value(al, states)
        return res

    def forward(self, act, states, atomic_value=None, ignore_pre=False):
        if '&' in act:
            output = torch.stack(
                [self(a.strip(), states, atomic_value=atomic_value, ignore_pre=ignore_pre) for a in act.split('&')], dim=0
            )
            if self.ptrajonly:
                return output.sum(0)
            else:
                return output.min(dim=0)[0]
        else:
            if atomic_value is not None and act in atomic_value:
                output = atomic_value[act]
            else:
                output = self.init_value(act, states, ignore_pre=ignore_pre)
                if atomic_value is not None:
                    atomic_value[act] = output
        return output
