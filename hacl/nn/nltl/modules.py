#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import jactorch
import torch
import torch.nn as nn

from hacl.nn.nlm import MLPLogic
from .functional import interval_pooling, transition_pooling_2d1d

__all__ = ['TemporalLogicMachineDP2D']


class TemporalLogicMachineDP2D(nn.Module):
    def __init__(self, input_dim, output_dim, nr_steps, hidden_dims=None, residual_input=True, until=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nr_steps = nr_steps
        self.hidden_dims = hidden_dims if hidden_dims is not None else [input_dim for _ in range(nr_steps - 1)]
        assert len(self.hidden_dims) == nr_steps - 1
        self.residual_input = residual_input
        self.until = until

        self.step_linears = nn.ModuleList()
        relation_dim = self.input_dim * (2 if until else 1)
        current_dim = relation_dim
        if self.nr_steps == 0:
            assert input_dim == output_dim
        else:
            for i in range(self.nr_steps - 1):
                self.step_linears.append(MLPLogic(current_dim + relation_dim, self.hidden_dims[i], []))
                current_dim = self.hidden_dims[i] + (relation_dim if self.residual_input else 0)
            self.step_linears.append(MLPLogic(current_dim, self.output_dim, []))

    def forward(self, a):
        """
        :param a: [batch, t, t, h_dim]
        :return: multi-step temporal quantification result f[batch, t, t, :] f[:, i, j, :] is the result for [i, j]
         for i <= j
        """
        t = a.size(1)
        if self.until:
            r = torch.cat((interval_pooling(a, reduction='max'), interval_pooling(a, reduction='min')), dim=-1)
        else:
            r = interval_pooling(a, reduction='max')

        f = torch.zeros(r.size(0), r.size(1) + 1, r.size(2) + 1, r.size(3))
        f[:, :-1, 1:] = r
        r = f
        for step in range(self.nr_steps - 1):
            linear_input = torch.cat((jactorch.add_dim(f, 3, t + 1), jactorch.add_dim(r, 1, t + 1)), dim=-1)
            f = transition_pooling_2d1d(self.step_linears[step](linear_input))
            if self.residual_input:
                f = torch.cat((f, r), dim=-1)
        f = self.step_linears[-1](f)
        return f[:, :-1, 1:]
