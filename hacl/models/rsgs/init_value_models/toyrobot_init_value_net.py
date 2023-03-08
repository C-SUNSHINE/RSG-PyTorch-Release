#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn

from hacl.nn.utils import init_weights


class PointCentric(nn.Module):
    def __init__(self, parameterized=True, alpha=1.0, radius=1.0, **kwargs):
        super().__init__()
        if parameterized:
            self.target = nn.Parameter(torch.tensor([10., 10.], dtype=torch.float32, requires_grad=True))
        else:
            self.target = None
        self.force_target = False
        self.alpha = alpha
        self.radius = radius
        init_weights(self)

    def set_target(self, x, y, weight=None, force_target=None):
        if self.target is None:
            self.target = nn.Parameter(torch.tensor([10., 10.], dtype=torch.float32, requires_grad=False))
        self.target.data[0] = x
        self.target.data[1] = y
        if force_target is not None:
            self.set_force_target(force_target)

    def set_force_target(self, mode=True):
        self.force_target = mode

    def forward(self, x, targets=None):
        if targets is None or self.force_target:
            assert self.target is not None
            targets = self.target.unsqueeze(0)
        res = torch.nn.LogSigmoid()(
            self.alpha * (self.radius ** 2 - ((x - targets) ** 2).sum(-1))
        )
        return res


class ToyRobotInitValueNet(nn.Module):
    _n_multi_points = 10

    def __init__(self, s_dim=None, net_type='mlp'):
        super().__init__()
        self.acts = []
        self.in_dim = s_dim
        self.h_dim = 128
        self.net = nn.ModuleDict()
        self.net_type = net_type

    def add_act(self, act):
        self.acts.append(act)
        if self.net_type == 'mlp':
            self.net.add_module(
                act,
                nn.Sequential(
                    nn.Linear(self.in_dim, self.h_dim),
                    nn.ReLU(),
                    nn.Linear(self.h_dim, 1),
                    nn.LogSigmoid(),
                ),
            )
        elif self.net_type == 'point':
            if self.in_dim is None:
                self.net.add_module(act, nn.Sequential(PointCentric()))
            else:
                self.net.add_module(act, nn.ModuleList([
                    nn.Linear(self.in_dim, 2),
                    PointCentric(parameterized=False, alpha=1.0, radius=0.5),
                ]))
        elif self.net_type == 'multi-point':
            n = self._n_multi_points
            self.net.add_module(act, nn.ModuleList([
                nn.Linear(self.in_dim, n * 2),
                PointCentric(parameterized=False, alpha=1.0, radius=1.5),
            ]))
        else:
            raise ValueError('Invalid net type %s' % self.net_type)
        # self.set_optimal_parameter()

    def set_optimal_parameter(self, ratio=0):
        for act in self.net:
            if act in self.net:
                pos = None
                if act == 'eff_MusicOn':
                    pos = 20
                elif act == 'eff_MusicOff':
                    pos = 17
                elif act == 'eff_Monkey':
                    pos = 14
                elif act == 'eff_LightOn':
                    pos = 11
                elif act == 'eff_Bell':
                    pos = 8
                elif act == 'eff_Ball':
                    pos = 5
                else:
                    continue
                self.net[act][0].weight.data[:] *= ratio
                self.net[act][0].bias.data[:] *= ratio
                self.net[act][0].weight.data[0, pos + 1] += (1 - ratio)
                self.net[act][0].weight.data[1, pos + 2] += (1 - ratio)
            self.net[act].requires_grad_(False)

    def forward(self, act, x):
        """
        :param act: a string describing sub-action predicate such as 'no_A' and 'has_A' or act1 & act2
        :param x: tensor with last one dimension [3] representing a symbolic state.
        :return: eliminate the last two dimensions to single value.
        """

        # for act_ in ['eff_Ball', 'eff_Bell', 'eff_LightOn', 'eff_Monkey', 'eff_MusicOff', 'eff_MusicOn']:
        #     if act_ in self.net:
        #         pos = {
        #             'agent': -1,
        #             'eff_Ball': 5,
        #             'eff_Bell': 8,
        #             'eff_LightOn': 11,
        #             'eff_Monkey': 14,
        #             'eff_MusicOff': 17,
        #             'eff_MusicOn': 20,
        #         }
        #         print(act_)
        #         print(self.net[act_][0].weight.data[0].detach().cpu())
        #         print(self.net[act_][0].weight.data[1].detach().cpu())
        #         print(self.net[act_][0].bias.data.detach().cpu())
        #         for k in pos:
        #             print('%s <- %s: (%.3f, %.3f)' % (act_, k, float(self.net[act_][0].weight.data[0, pos[k] + 1]), float(self.net[act_][0].weight.data[1, pos[k] + 2])))
        # # exit()

        assert act in self.acts
        siz = x.size()[:-1]
        y = x.view(-1, x.size(-1))
        if self.net_type == 'mlp':
            output = self.net[act](y)
        elif self.net_type == 'point':
            if self.in_dim is None:
                output = self.net[act](y)
            else:
                # print(self.net[act][0].weight.device, y.device)
                # input()
                output = self.net[act][0](y)
                output = self.net[act][1](y[:, :2], output)
        elif self.net_type == 'multi-point':
            n = self._n_multi_points
            output = self.net[act][0](y)
            output = output.view(output.size(0), n, 2)
            # print(y.unsqueeze(1).size(), output.size())
            output = self.net[act][1](y[:, :2].unsqueeze(1), output).mean(1)
        else:
            raise ValueError('Invalid net type %s' % str(self.net_type))
        if len(siz) == 0:
            output = output[0]
        else:
            output = output.reshape(*siz)
        return output
