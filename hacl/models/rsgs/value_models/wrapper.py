#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import optim
from torch import nn
import os


class ValueEstimatorStateMachineWrapper(nn.Module):
    def __init__(self, state_machine, net, *args, optimizer_lr=0.01, **kwargs):
        super(ValueEstimatorStateMachineWrapper, self).__init__()
        self.state_machine = state_machine
        self.net = nn.ModuleList()
        for node in self.state_machine.nodes:
            self.net.append(net(*args, **kwargs))
        self.loss_func = nn.MSELoss(reduction='mean')
        if not hasattr(self.net[0], 'tune'):
            self.set_optimizer(lr=optimizer_lr)

    def set_optimizer(self, lr=0.01):
        parameters = list(filter(lambda p: p.requires_grad, self.net.parameters()))
        if len(parameters) == 0:
            self.optimizer = None
        else:
            self.optimizer = optim.Adam(parameters, lr=lr)

    def forward(self, states):
        """
        :param states: batch of states
        :return: [V(s, node|L) for node in state_machine.nodes]
        """
        res = [self.net[index](states) for index in range(len(self.state_machine.nodes))]
        return torch.stack(res, dim=-1)

    def tune(self, states, values):
        if hasattr(self.net[0], 'tune'):
            for i in range(len(self.net)):
                self.net[i].tune(states, values[:, i], index=i)
        else:
            outputs = self(states.detach())
            loss = self.loss_func(outputs, values.detach())
            # print('value net loss =', loss)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 5)
            self.optimizer.step()

    def visualize_nodes(self, save_dir=None, **kwargs):
        for i, node in enumerate(self.state_machine.nodes):
            node_save_dir = os.path.join(save_dir, str(node))
            self.net[i].visualize(save_dir=node_save_dir, **kwargs)
