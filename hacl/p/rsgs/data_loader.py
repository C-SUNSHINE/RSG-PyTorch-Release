#! /usr/bin/env python3
# -*- coding: utf-8 -*-

__all__ = ['DataLoader']


class DataLoader(object):
    def __init__(self):
        pass

    def batches(self, split, batch_size=10, **kwargs):
        """
        Returns: (split, batch) pairs as a generator.
        """
        raise NotImplementedError()

    def collect_data(self, batch, index, *args, **kwargs):
        """
        Args:
            batch: batch to collect data from
            index: the index of data to collect
        Returns: a object for the collectable data that can be printed by print_data
        """
        raise NotImplementedError()

    def print_data(self, data, log=print, *args, **kwargs):
        """
        Args:
            data: data to print
            log: the print method to print string, None if only print nothing but only return string
        Returns: the string we printed (may include '\n')
        """
        raise NotImplementedError()
