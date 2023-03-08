#! /usr/bin/env python3
# -*- coding: utf-8 -*-


class TaskPipeline(object):
    def __init__(self, *args, **kwargs):
        self.evaluators = dict()
        self.data_loader = None
        self.labels = None

    def init(self, *args, **kwargs):
        pass

    def register_evaluators(self, *args, **kwargs):
        if len(args) > 0:
            assert len(args) == 1 and len(kwargs) == 0
            for k, v in args[0].items():
                self.evaluators[k] = v
        else:
            for k, v in kwargs.items():
                self.evaluators[k] = v

    def epoch_init(self, *args, **kwargs):
        raise NotImplementedError()

    def batch_run(self, *args, **kwargs):
        raise NotImplementedError()

    def epoch_end(self, *args, **kwargs):
        raise NotImplementedError()

    def summary(self, *args, **kwargs):
        raise NotImplementedError()
