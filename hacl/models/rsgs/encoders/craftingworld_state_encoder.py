#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn

from hacl.envs.gridworld.crafting_world.broadcast_engine import INVENTORY_DIM, OBJECT_DIM, STATUS_DIM, TYPE_DIM
from hacl.nn.utils import Negative

GLOBAL_DIM = 4
RELATION_DIM = 1

class CraftingWorldStateEncoder(nn.Module):
    def __init__(self, h_dim=128, out_dim=1, agg_method='max', activation='relu', add_negative=False):
        super().__init__()
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.agg_method = agg_method
        self.inventory_relational = nn.Sequential(
            nn.Linear(GLOBAL_DIM + OBJECT_DIM, self.h_dim),
            nn.ReLU(),
        )
        self.blocks_relational = nn.Sequential(
            nn.Linear(GLOBAL_DIM + OBJECT_DIM + TYPE_DIM + 2 + STATUS_DIM + RELATION_DIM, self.h_dim),
            nn.ReLU(),
        )
        if activation == 'relu':
            activation_cls = nn.ReLU
        elif activation == 'logsigmoid':
            activation_cls = nn.LogSigmoid
        else:
            raise ValueError()
        self.propagator = nn.Linear(GLOBAL_DIM + self.h_dim * 2, self.out_dim)
        self.activator = activation_cls()
        if add_negative:
            assert activation == 'logsigmoid'
            self.neg_activator = nn.Sequential(
                Negative(),
                activation_cls()
            )
        else:
            self.neg_activator = None

    @classmethod
    def make_onehot(cls, s, length):
        t = torch.zeros(s.size(0), length, device=s.device)
        t.scatter_(1, s.view(-1, 1), 1)
        return t

    @classmethod
    def make_input(cls, s):
        n_objects = (s.size(1) - 6 - INVENTORY_DIM) // 5
        assert 6 + INVENTORY_DIM + n_objects * 5 == s.size(1)
        n = s.size(0)
        s_global = s[:, :6]
        s_inventory = s[:, 6:6 + INVENTORY_DIM]
        s_blocks = s[:, 6 + INVENTORY_DIM:].view(n, n_objects, 5)
        mask_inventory = ~s_inventory.eq(0).type(torch.long)
        mask_blocks = ~s_blocks[:, :, 0].eq(0).type(torch.long)
        object_onehot_inventory = cls.make_onehot(s_inventory.reshape(-1), OBJECT_DIM).view(
            *s_inventory.size(), OBJECT_DIM) * mask_inventory.unsqueeze(-1)

        object_onehot_blocks = cls.make_onehot(s_blocks[:, :, 0].reshape(-1), OBJECT_DIM).view(
            -1, n_objects, OBJECT_DIM) * mask_blocks.unsqueeze(-1)
        type_onehot_blocks = cls.make_onehot(s_blocks[:, :, 1].reshape(-1), TYPE_DIM).view(
            -1, n_objects, TYPE_DIM) * mask_blocks.unsqueeze(-1)
        xy_blocks = s_blocks[:, :, 2:4]
        status_onehot_blocks = cls.make_onehot(s_blocks[:, :, 4].reshape(-1), STATUS_DIM).view(
            -1, n_objects, STATUS_DIM) * mask_blocks.unsqueeze(-1)

        overlap_blocks = xy_blocks.eq(s_global[:, :2].unsqueeze(1)).min(2)[0].type(torch.float) * mask_blocks

        relation_blocks = torch.stack([
            overlap_blocks
        ], dim=2)

        global_input = s_global[:, :4].type(torch.float)
        inventory_input = object_onehot_inventory
        blocks_input = torch.cat([
            object_onehot_blocks,
            type_onehot_blocks,
            xy_blocks,
            status_onehot_blocks,
            relation_blocks
        ], dim=2)
        return global_input, inventory_input, mask_inventory, blocks_input, mask_blocks

    def _evaluate(self, inputs, use_neg=False):
        global_input, inventory_input, mask_inventory, blocks_input, mask_blocks = inputs
        n = global_input.size(0)
        c = inventory_input.size(1)
        k = blocks_input.size(1)

        ivt_input = torch.cat([
            global_input.unsqueeze(1).repeat(1, c, 1),
            inventory_input
        ], dim=2)

        blc_input = torch.cat([
            global_input.unsqueeze(1).repeat(1, k, 1),
            blocks_input
        ], dim=2)

        ivt_relation = self.inventory_relational(ivt_input.view(
            -1, ivt_input.size(-1))).view(*ivt_input.size()[:-1], -1) * mask_inventory.unsqueeze(-1)
        blc_relation = self.blocks_relational(blc_input.view(
            -1, blc_input.size(-1))).view(*blc_input.size()[:-1], -1) * mask_blocks.unsqueeze(-1)

        if self.agg_method == 'mean':
            ivt_eff = ivt_relation.mean(1)
            blc_eff = blc_relation.mean(1)
        elif self.agg_method == 'max':
            ivt_eff = ivt_relation.max(1)[0]
            blc_eff = blc_relation.max(1)[0]
        else:
            raise ValueError('Invalid agg_method %s' % self.agg_method)

        agg_input = torch.cat([
            global_input,
            ivt_eff,
            blc_eff,
        ], dim=1)

        output = self.propagator(agg_input)
        if not use_neg:
            output = self.activator(output)
        else:
            output = self.neg_activator(output)
        return output.view(n, self.out_dim)

    def forward(self, x, use_neg=False):
        siz = x.size()[:-1]
        y = self.make_input(x.view(-1, x.size(-1)))
        output = self._evaluate(y, use_neg=use_neg)
        if len(siz) == 0:
            output = output.reshape(output.size(-1))
        else:
            output = output.reshape(*siz, output.size(-1))
        return output
