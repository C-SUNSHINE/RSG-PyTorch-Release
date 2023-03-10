#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List

import torch
from jacinle.utils.enum import JacEnum

__all__ = [
    'TemporalPoolingImplementation',
    'TemporalPoolingReduction',
    'transition_pooling_2d1d',
    'interval_pooling',
    'matrix_from_diags',
    'matrix_remove_diag',
]


class TemporalPoolingImplementation(JacEnum):
    BROADCAST = 'broadcast'
    FORLOOP = 'forloop'


class TemporalPoolingReduction(JacEnum):
    MAX = 'max'
    MIN = 'min'
    SOFTMAX = 'softmax'
    SOFTMIN = 'softmin'


def masked_min(input, mask, dim, inf=1e9):
    mask = mask.type(input.dtype)
    input = input * mask + inf * (1 - mask)
    return input.min(dim)[0]


def masked_max(input, mask, dim, inf=1e9):
    mask = mask.type(input.dtype)
    input = input * mask + inf * (mask - 1)
    return input.max(dim)[0]


def transition_pooling_2d1d(input, implementation=None, reduction='max'):
    t = input.size(1)
    indices = torch.arange(t, device=input.device)
    indices_i = indices.view(1, t, 1, 1, 1).repeat(input.size(0), 1, t, t, input.size(-1))
    indices_j = indices.view(1, 1, t, 1, 1).repeat(input.size(0), t, 1, t, input.size(-1))
    indices_k = indices.view(1, 1, 1, t, 1).repeat(input.size(0), t, t, 1, input.size(-1))
    mask = (indices_i < indices_j) & (indices_j < indices_k)
    return masked_max(input, mask, dim=2)


def interval_pooling(input, implementation='forloop', reduction='max', beta=None):
    """
    Args:
        input (torch.Tensor): 3D tensor of [batch_size, nr_frames, hidden_dim]
        implementation (Union[TemporalPoolingImplementation, str]): the implementation. Currently only support FORLOOP.
        reduction (Union[TemporalPoolingReduction, str]): reduction method. Either MAX or MIN.
    Return:
        output (torch.Tensor): 4D tensor of [batch_size, nr_frames, nr_frames, hidden_dim], where
        ```
            output[:, i, j, :] = min output[:, k, :] where i <= k <= j
        ```
        the k is cyclic-indexed.
    """
    implementation = TemporalPoolingImplementation.from_string(implementation)
    reduction = TemporalPoolingReduction.from_string(reduction)
    batch_size, nr_frames = input.size()[:2]

    if implementation is TemporalPoolingImplementation.FORLOOP:
        if reduction is TemporalPoolingReduction.MAX or reduction is TemporalPoolingReduction.MIN:
            input_doubled = torch.cat((input, input), dim=1)  # repeat the input at dim=1.

            output_tensors = list()
            output_tensors.append(input)
            for length in range(2, nr_frames + 1):
                last_tensor = output_tensors[-1]
                last_elems = input_doubled[:, length - 1 : length - 1 + nr_frames]
                if reduction is TemporalPoolingReduction.MAX:
                    this_tensor = torch.max(last_tensor, last_elems)
                elif reduction is TemporalPoolingReduction.MIN:
                    this_tensor = torch.min(last_tensor, last_elems)
                else:
                    raise ValueError('Wrong value {}.'.format(reduction))
                output_tensors.append(this_tensor)
            return matrix_from_diags(output_tensors, dim=1, triu=True)
        else:
            from math import exp

            scale = exp(beta)

            length = input.size(1)

            # output = torch.zeros(input.size()[:2] + input.size()[1:]).to(input.device)
            # assert beta is not None
            # for i in range(length):
            #     for j in range(i, length):
            #         if reduction is TemporalPoolingReduction.SOFTMAX:
            #             elem = (input[:, i:j + 1] * torch.nn.functional.softmax(input[:, i:j + 1] / scale, 1)).sum(1)
            #         elif reduction is TemporalPoolingReduction.SOFTMIN:
            #             elem = (input[:, i:j + 1] * torch.nn.functional.softmin(input[:, i:j + 1] / scale, 1)).sum(1)
            #         else:
            #             raise ValueError('Wrong value {}.'.format(reduction))
            #         output[:, i, j] = elem
            # # return output

            input_doubled = torch.cat((input, input), dim=1)  # repeat the input at dim=1.

            output_tensors = list()
            if reduction is TemporalPoolingReduction.SOFTMIN:
                scale = -scale
            else:
                assert reduction is TemporalPoolingReduction.SOFTMAX
            input_arg = torch.exp(input / scale)
            output_tensors.append((input * input_arg, input_arg))
            for length in range(2, nr_frames + 1):
                last_tensor, last_argsum = output_tensors[-1]
                last_elems = input_doubled[:, length - 1 : length - 1 + nr_frames]
                last_elems_arg = torch.exp(last_elems / scale)
                output_tensors.append((last_tensor + last_elems * last_elems_arg, last_argsum + last_elems_arg))
            output2 = matrix_from_diags([x[0] / x[1] for x in output_tensors], dim=1, triu=True)

            # Test:
            # X, Y = torch.meshgrid(torch.arange(length), torch.arange(length))
            # upper = (X < Y).float().view(1, length, length, 1).to(output.device)
            # print((((output - output2) ** 2) * upper).sum())
            # exit()

            return output2
    else:
        raise NotImplementedError('Unknown interval pooling implementation: {}.'.format(implementation))


def matrix_from_diags(diags: List[torch.Tensor], dim: int = 1, triu: bool = False):
    """
    Construct an N by N matrix from N diags of the matrix.
    Args:
        diags (List[torch.Tensor]): N length-N vectors regarding the 1st, 2nd, ... diags of the output matrix.
            They can also be same-dimensional tensors, where the matrix will be created at the dim and dim+1 axes.
        dim (int): the matrix will be created at dim and dim+1.
        triu (bool): use only the upper triangle of the matrix.
    Return:
        output: torch.Tensor
    """

    if dim < 0:
        dim += diags[0].dim()

    size = diags[0].size()
    diags.append(torch.zeros_like(diags[0]))
    output = torch.cat(diags, dim=dim)  # [..., (f+1)*f, ...]
    output = output.reshape(size[:dim] + (size[dim] + 1, size[dim]) + size[dim + 1 :])
    output = output.transpose(dim, dim + 1)
    output = output.reshape(
        size[:dim] + (size[dim] + 1, size[dim]) + size[dim + 1 :]
    )  # use to reshape for auto-contiguous.

    if triu:
        return output.narrow(dim, 0, size[dim])

    output = torch.cat(
        (output.narrow(dim, 0, 1), matrix_remove_diag(output.narrow(dim, 1, size[dim]), dim=dim, move_up=True)), dim=dim
    )
    return output


def matrix_remove_diag(matrix: torch.Tensor, dim: int = 1, move_up: bool = False):
    """
    Remove the first diag of the input matrix. The result is an N x (N-1) matrix.
    Args:
        matrix (torch.Tensor): the input matrix. It can be a tensor where the dim and dim+1 axes form a matrix.
        dim (int): the matrix is at dim and dim+1.
        move_up (bool): if True, the output matrix will be of shape (N-1) x N.
            In the move_left (default, move_up=False) mode, the left triangle will stay in its position and the upper triangle will move 1 element left.
            While in the move_up mode, the upper triangle will stay in its position, and the left triangle will move 1 element up.
    """

    if dim < 0:
        dim += matrix.size()

    if move_up:
        matrix = matrix.transpose(dim, dim + 1)

    size = matrix.size()
    n = size[dim]
    matrix = matrix.reshape(size[:dim] + (n * n,) + size[dim + 2 :])
    matrix = matrix.narrow(dim, 1, n * n - 1)
    matrix = matrix.reshape(size[:dim] + (n - 1, n + 1) + size[dim + 2 :])
    matrix = matrix.narrow(dim + 1, 0, n)
    matrix = matrix.reshape(size[:dim] + (n, n - 1) + size[dim + 2 :])

    if move_up:
        matrix = matrix.transpose(dim, dim + 1)

    return matrix
