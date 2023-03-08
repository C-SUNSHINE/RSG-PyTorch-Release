#! /usr/bin/env python3
# -*- coding: utf-8 -*-

__all__ = ['sprint', 'BufferPrinter']


def sprint(*args, **kwargs):
    end = '\n' if 'end' not in kwargs else kwargs['end']
    sep = ' ' if 'sep' not in kwargs else kwargs['sep']
    current = ''
    first = False
    for arg in args:
        if not first:
            current += sep
            first = True
        current += str(arg)
    current += end
    return current


class FileLogger:
    def __init__(self, fout, display=True):
        self.fout = fout
        self.display = display

    def __call__(self, *args, **kwargs):
        s = sprint(*args, **kwargs)
        self.fout.write(s)
        self.fout.flush()
        if self.display:
            print(s, end='')


class BufferPrinter:
    """
    A wrapper to any printer, which also concatenate all contents as string.
    """

    def __init__(self, log=None):
        self.log = log
        assert log is None or callable(log)
        self.buffer = ''

    def __call__(self, *args, **kwargs):
        current = sprint(*args, **kwargs)
        self.buffer += current
        if self.log is not None:
            self.log(current, end='')
            if hasattr(self.log, 'flush') and callable(self.log.flush):
                self.log.flush()

    def clear(self):
        temp = self.buffer
        self.buffer = ''
        return temp

    def get(self):
        return self.buffer
