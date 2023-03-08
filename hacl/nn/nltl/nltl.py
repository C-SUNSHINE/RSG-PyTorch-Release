#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools

import jactorch
import torch
from torch import nn


def merge(x, y, dim=-1):
    if x is None:
        return y
    if y is None:
        return x
    return torch.cat([x, y], dim=dim)


def exclude_mask(inputs, cnt=2, dim=1):
    """Produce an exclusive mask.
    Specifically, for cnt=2, given an array a[i, j] of n * n, it produces
    a mask with size n * n where only a[i, j] = 1 if and only if (i != j).
    Args:
      inputs: The tensor to be masked.
      cnt: The operation is performed over [dim, dim + cnt) axes.
      dim: The starting dimension for the exclusive mask.
    Returns:
      A mask that make sure the coordinates are mutually exclusive.
    """
    assert cnt > 0
    if dim < 0:
        dim += inputs.dim()
    n = inputs.size(dim)
    for i in range(1, cnt):
        assert n == inputs.size(dim + i)

    rng = torch.arange(0, n, dtype=torch.long, device=inputs.device)
    q = []
    for i in range(cnt):
        p = rng
        for j in range(cnt):
            if i != j:
                p = p.unsqueeze(j)
        p = p.expand((n,) * cnt)
        q.append(p)
    mask = q[0] == q[0]
    # Mutually Exclusive
    for i in range(cnt):
        for j in range(cnt):
            if i != j:
                mask *= q[i] != q[j]
    for i in range(dim):
        mask.unsqueeze_(0)
    for j in range(inputs.dim() - dim - cnt):
        mask.unsqueeze_(-1)

    return mask.expand(inputs.size()).float()


def mask_value(inputs, mask, value):
    assert inputs.size() == mask.size()
    return inputs * mask + value * (1 - mask)


class Compose(nn.ModuleList):
    def get_output_dim(self, input_dim):
        for module in self.children():
            input_dim = module.get_output_dim(input_dim)
        return input_dim


class Expander(nn.Module):
    """Capture a free variable into predicates, implemented by broadcast."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, inputs, n=None):
        # print(inputs.size(), self.dim)
        if self.dim == 0:
            assert n is not None
        elif n is None:
            n = inputs.size(-2)
        return jactorch.add_dim(inputs, -2, n)
        # return inputs.unsqueeze(self.dim + 1).repeat(*([1] * (self.dim + 1) + [n, 1]))

    def get_output_dim(self, input_dim):
        return input_dim


class Reducer(nn.Module):
    """Reduce out a variable via quantifiers (exists/forall), implemented by max/min-pooling."""

    def __init__(self, dim, exclude_self=True, exists=True):
        super().__init__()
        self.dim = dim
        self.exclude_self = exclude_self
        self.exists = exists

    def forward(self, inputs):
        shape = inputs.size()
        inp0, inp1 = inputs, inputs
        if self.exclude_self:
            mask = exclude_mask(inputs, cnt=self.dim, dim=-1 - self.dim)
            inp0 = mask_value(inputs, mask, 0.0)
            inp1 = mask_value(inputs, mask, 1.0)

        if self.exists:
            shape = shape[:-2] + (shape[-1] * 2,)
            exists = torch.max(inp0, dim=-2)[0]
            forall = torch.min(inp1, dim=-2)[0]
            return torch.stack((exists, forall), dim=-1).view(shape)

        shape = shape[:-2] + (shape[-1],)
        return torch.max(inp0, dim=-2)[0].view(shape)

    def get_output_dim(self, input_dim):
        if self.exists:
            return input_dim * 2
        return input_dim


class Permutation(nn.Module):
    """Create r! new predicates by permuting the axies for r-arity predicates."""

    def __init__(self, dim, permute=True):
        super().__init__()
        self.dim = dim
        self.permute = permute

    def forward(self, inputs):
        if self.dim <= 1 or not self.permute:
            return inputs
        nr_dims = len(inputs.size())
        # Assume the last dim is channel.
        index = tuple(range(nr_dims - 1))
        start_dim = nr_dims - 1 - self.dim
        assert start_dim > 0
        res = []
        for i in itertools.permutations(index[start_dim:]):
            p = index[:start_dim] + i + (nr_dims - 1,)
            res.append(inputs.permute(p))
        return torch.cat(res, dim=-1)

    def get_output_dim(self, input_dim):
        if not self.permute:
            return input_dim
        mul = 1
        for i in range(self.dim):
            mul *= i + 1
        return input_dim * mul


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, h_dims):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.h_dims = tuple(h_dims)
        layers = []
        dim = in_dim
        for new_dim in self.h_dims + (out_dim,):
            layers.append(nn.Linear(dim, new_dim))
            dim = new_dim
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs):
        input_size = inputs.size()[:-1]
        input_channel = inputs.size(-1)

        return self.layers(inputs.view(-1, input_channel)).view(*input_size, -1)


class MLPLogic(MLP):
    def __init__(self, in_dim, out_dim, h_dim):
        super().__init__(in_dim, out_dim, h_dim)
        self.layers.add_module(str(len(self.layers)), nn.Sigmoid())


def _get_tuple_n(x, n, tp):
    """Get a length-n list of type tp."""
    assert tp is not list
    if isinstance(x, tp):
        x = tuple(
            [
                x,
            ]
            * n
        )
    assert len(x) == n, 'Parameters should be {} or list of N elements.'.format(tp)
    for i in x:
        assert isinstance(i, tp), 'Elements of list should be {}.'.format(tp)
    return tuple(x)
