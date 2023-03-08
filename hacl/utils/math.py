#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import torch


def gauss_log_prob(mu, var, x):
    p1 = -((mu - x) ** 2) / (2 * var.clamp(min=1e-5))
    p2 = -torch.log(torch.sqrt(2 * math.pi * var))
    return p1 + p2


def gauss_sample(mu, var):
    return torch.normal(0, 1, mu.size(), device=mu.device) * torch.sqrt(var.clamp(min=1e-5)) + mu
