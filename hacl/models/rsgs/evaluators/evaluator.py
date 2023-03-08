#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from torch import nn
from jactorch.nn import TorchApplyRecorderMixin


class Evaluator(TorchApplyRecorderMixin):
    def __init__(self):
        super(Evaluator, self).__init__()

    def get_training_parameters(self):
        raise NotImplementedError()

    @property
    def qvalue_based(self):
        raise NotImplementedError()

    @property
    def online_optimizer(self):
        return None

    def prepare_epoch(self, *args, **kwargs):
        pass

    def extra_state_dict(self):
        return {key: getattr(self, key) for key in self._EXTRA_DICT_KEY}

    def load_extra_state_dict(self, extra_dict):
        for key in self._EXTRA_DICT_KEY:
            setattr(self, key, extra_dict[key])
