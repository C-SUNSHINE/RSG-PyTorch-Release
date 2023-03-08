#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import torch


def transfer(states, f, state_machine, init_value, topo_seq=None, reverse=False, inplace=True, s_dim=2):
    if not inplace:
        f = f.clone()
    siz = states.size()[:-s_dim]
    states = states.view(-1, *states.size()[-s_dim:])
    f = f.view(-1, f.size(-1))
    ff = [f[:, i] for i in range(f.size(-1))]
    atomic_values = dict()
    if topo_seq is None:
        topo_seq = state_machine.get_topological_sequence()
    if not reverse:
        for u in topo_seq:
            for v, label in state_machine.adjs[u]:
                x, y = state_machine.node2index[u], state_machine.node2index[v]
                delta = state_machine.potential[y] - state_machine.potential[x]
                ff[y] = torch.max(ff[y], ff[x] + delta * init_value(label, states, atomic_values))
    else:
        for v in reversed(topo_seq):
            for u, label in state_machine.radjs[v]:
                x, y = state_machine.node2index[u], state_machine.node2index[v]
                delta = state_machine.potential[y] - state_machine.potential[x]
                ff[x] = torch.max(ff[x], ff[y] + delta * init_value(label, states, atomic_values))
    f = torch.stack(ff, dim=-1).view(*siz, f.size(-1))
    return f
