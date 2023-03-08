#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import torch

__all__ = ['get_all_paths', 'connection_probability', 'max_sum_multistep', 'max_sum_state_machine']


def get_all_paths(state_machine):
    q = []
    for node in state_machine.nodes:
        if node in state_machine.starts:
            q.append(((node,), tuple()))
    paths = []
    while len(q) > 0:
        (nodes, edges) = q[0]
        q = q[1:]
        if nodes[-1] in state_machine.ends:
            paths.append((nodes, edges))
        else:
            for v, l in state_machine.adjs[nodes[-1]]:
                q.append((nodes + (v,), edges + (l,)))
    return paths


def connection_probability(state_machine, length, prob):
    """
    :param state_machine: A state machine.
    :param length: number of frames.
    :param prob: a dict mapping each edge label to a probability 2d-Tensor p: it exists in interval [i,j] is p[i,j].
    :return: the probability starts is connected with ends in interval [0, length-1].
    """
    device = prob[state_machine.edges[0][2]].device
    f = {x: torch.zeros(length + 1, device=device) for x in state_machine.nodes}
    for x in state_machine.starts:
        f[x][0] = 1
    topo = state_machine.get_topological_sequence()
    for x in topo:
        for y, el in state_machine.adjs[x]:
            p = prob[el]
            fx = f[x]
            tfy = torch.zeros(length + 1, device=device)
            tfy[1:] = (fx[:length].unsqueeze(1).repeat(1, length) * p[:length, :length]).max(dim=0)[0]
            f[y] = f[y] + (1 - f[y]) * tfy
    g = torch.zeros(length + 1, device=device)
    for x in state_machine.ends:
        g = g + (1 - g) * f[x]
    return g[length]


def get_transition(a):
    assert a.dim() == 1
    n = a.size(0)
    b = torch.cat((torch.zeros(1).to(a.device), a), dim=0)
    b = b.view(1, n + 1).repeat(n + 1, 1)
    xx, yy = torch.meshgrid(torch.arange(n + 1).to(a.device), torch.arange(n + 1).to(a.device))
    b = b * (xx < yy).type(torch.float)
    b = b.cumsum(dim=1)
    b -= (xx > yy).type(torch.float) * 1e9
    return b


def transit(f, x):
    t = get_transition(x)
    return (f.unsqueeze(1) + t).max(dim=0)[0]


def max_sum_multistep(x):
    """
    Args:
        x: a 2-dimension K*T tensor.

    Returns: Cut interval [0, T-1] into N segments [l_i, r_i) and compute s = sum_i: x[i, l_i:r_i],
            and return maximized s over all cuttings.
    """
    f = torch.cumsum(torch.cat((torch.zeros(1).to(x.device), x[0]), dim=0), dim=0)
    for i in range(1, x.size(0)):
        f = transit(f, x[i])
    return f[-1]


def max_sum_state_machine(values, step_values, machine):
    """
    Args:
        values: values[e, i]: reward (log prob) of taking action a_i at edge e.
        step_values: step_values[e, i] reward (log prob) to transit to next step at s_i
        machine: the state machine

    Returns: the maximum reward from start node to end node.

    """
    assert set(machine.label2edge.keys()) == set(values.keys())
    sample_value = list(values.values())[0]
    n = sample_value.size(0)
    device = sample_value.device
    invalid = torch.zeros(n + 1).to(device) - 1e9
    f = {x: invalid.clone() for x in machine.nodes}
    g = {x: invalid.clone() for x in machine.label2edge}
    for x in machine.starts:
        f[x][0] = 0
    for x in machine.get_topological_sequence():
        for (w, l) in machine.radjs[x]:
            new = g[(w, l)] + step_values[(w, l)]
            f[x] = torch.max(f[x], new)
        for (y, l) in machine.adjs[x]:
            new = transit(f[x], values[(x, l)])
            g[(x, l)] = torch.max(g[(x, l)], new)
    res = invalid.clone()
    for x in machine.nodes:
        res = torch.max(res, f[x])
    for e in machine.label2edge:
        res = torch.max(res, g[e])
    return res[-1]
