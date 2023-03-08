#! /usr/bin/env python3
# -*- coding: utf-8 -*-


def growing_sampler(sampler, min_n, max_n, **kwargs):
    n = max_n
    while True:
        temp = sampler(n, **kwargs)
        if temp is not None:
            return temp
        if n == min_n:
            return None
        else:
            n = max(min_n, n // 2)
