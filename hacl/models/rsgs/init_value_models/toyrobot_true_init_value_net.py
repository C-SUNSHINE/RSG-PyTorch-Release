#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn
from .toyrobot_init_value_net import PointCentric
from hacl.nn.utils import init_weights, TimesConstant


class ToyRobotTrueInitValueNet(nn.Module):
    def __init__(self, pspace=None, s_dim=3, **kwargs):
        super().__init__()
        self.acts = []
        self.in_dim = s_dim
        self.net = nn.ModuleDict()
        self.region_center = {}
        for name in pspace.regions:
            self.region_center[name] = pspace.regions[name].reference

    def _build_true_init_value_net(self, name, pre=False):
        net = PointCentric()
        net.set_target(self.region_center[name].x, self.region_center[name].y)
        for param in net.parameters():
            param.requires_grad = False
        if pre:
            return nn.Sequential(net, TimesConstant(-0.1))
        return net

    def add_act(self, act):
        self.acts.append(act)
        if act.startswith('eff_'):
            self.net.add_module(act, self._build_true_init_value_net(act[4:]))
        elif act.startswith('pre_'):
            self.net.add_module(act, self._build_true_init_value_net(act[4:], pre=True))
        init_weights(self.net[act])

    def forward(self, act, x):
        """
        :param act: a string describing sub-action predicate such as 'no_A' and 'has_A' or act1 & act2
        :param x: tensor with last one dimension [3] representing a symbolic state.
        :return: eliminate the last two dimensions to single value.
        """
        assert act in self.acts
        siz = x.size()[:-1]
        y = x.view(-1, x.size(-1))
        assert y.size(1) == self.in_dim
        output = self.net[act](y[:, :2])
        if len(siz) == 0:
            output = output[0]
        else:
            output = output.reshape(*siz)
        return output
