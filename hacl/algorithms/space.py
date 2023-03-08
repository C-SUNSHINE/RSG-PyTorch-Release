#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from math import ceil

__all__ = [
    'ConfigurationSpace',
    'ProxyConfigurationSpace',
    'BoxConfigurationSpace',
    'ConstrainedConfigurationSpace',
    'BoxConstrainedConfigurationSpace',
    'ProblemSpace',
]


class ConfigurationSpace(object):
    def sample(self):
        raise NotImplementedError()

    def difference(self, config1, config2):
        raise NotImplementedError()

    def distance(self, config1, config2):
        raise NotImplementedError()

    def validate_config(self, config):
        raise NotImplementedError()

    def gen_path(self, config1, config2):
        raise NotImplementedError()


class ProxyConfigurationSpace(ConfigurationSpace):
    def __init__(self, proxy):
        self.proxy = proxy

    def __getattr__(self, item):
        return getattr(self.proxy, item)

    def sample(self):
        return self.proxy.sample()

    def difference(self, config1, config2):
        return self.proxy.difference(config1, config2)

    def distance(self, config1, config2):
        return self.proxy.distance(config1, config2)

    def validate_config(self, config):
        return self.proxy.validate_config(config)

    def gen_path(self, config1, config2):
        raise NotImplementedError()


class BoxConfigurationSpace(ConfigurationSpace):
    def __init__(self, cspace_ranges, cspace_max_stepdiff):
        super().__init__()
        self.cspace_ranges = cspace_ranges

        if isinstance(cspace_max_stepdiff, (int, float)):
            cspace_max_stepdiff = (cspace_max_stepdiff,) * len(self.cspace_ranges)
        self.cspace_max_stepdiff = np.array(cspace_max_stepdiff, dtype='float32')

    def sample(self):
        return tuple(r.sample() for r in self.cspace_ranges)

    def difference(self, one: tuple, two: tuple):
        a = [self.cspace_ranges[i].difference(one[i], two[i]) for i in range(len(self.cspace_ranges))]
        return a

    def distance(self, one: tuple, two: tuple):
        a = [abs(self.cspace_ranges[i].difference(one[i], two[i])) for i in range(len(self.cspace_ranges))]
        return sum(a)

    def validate_config(self, config):
        return len(config) == len(self.cspace_ranges) and all(
            [self.cspace_ranges[i].contains(config[i]) for i in range(len(self.cspace_ranges))]
        )

    def gen_path(self, config1: tuple, config2: tuple):
        assert self.validate_config(config1) and self.validate_config(
            config2
        ), 'Invalid config pair: {} and {}.'.format(config1, config2)
        diffs = [self.cspace_ranges[i].difference(config1[i], config2[i]) for i in range(len(self.cspace_ranges))]
        samples = max([int(ceil(abs(diff) / max_diff)) + 1 for diff, max_diff in zip(diffs, self.cspace_max_stepdiff)])
        samples = max(2, samples)
        linear_interpolation = [diff / (samples - 1.0) for diff in diffs]
        path = [config1]
        for s in range(1, samples - 1):
            sample = tuple(
                [
                    self.cspace_ranges[i].in_range(config1[i] + s * linear_interpolation[i])
                    for i in range(len(self.cspace_ranges))
                ]
            )
            path.append(sample)
        return True, path + [config2]


class ConstrainedConfigurationSpace(ProxyConfigurationSpace):
    def __init__(self, cspace, constrain_config_func=None, extra_validate_config_func=None):
        super().__init__(cspace)
        self.constrain_config_func = constrain_config_func
        self.extra_validate_config_func = extra_validate_config_func

    def sample(self):
        return self.constrain_config(self.proxy.sample())

    def validate_config(self, config):
        if self.extra_validate_config_func is not None:
            return self.proxy.validate_config(config) and self.extra_validate_config_func(self, config)
        else:
            return self.proxy.validate_config(config)

    def constrain_config(self, config):
        if self.constrain_config_func is not None:
            return self.constrain_config_func(self, config)

        raise NotImplementedError()

    def gen_path(self, config1: tuple, config2: tuple):
        success, path = self.proxy.gen_path(config1, config2)
        for i, config in enumerate(path):
            path[i] = self.constrain_config(config)
        return success, path


class BoxConstrainedConfigurationSpace(ConstrainedConfigurationSpace):
    def __init__(
        self,
        cspace,
        constrain_config_func=None,
        extra_validate_config_func=None,
        cspace_max_stepdiff_proj=3,
        gen_path_atol=1e-5,
    ):

        assert isinstance(cspace, BoxConfigurationSpace)

        super().__init__(cspace, constrain_config_func, extra_validate_config_func)
        self._cspace_max_stepdiff_proj = cspace_max_stepdiff_proj
        self._gen_path_atol = gen_path_atol

    @property
    def cspace_ranges(self):
        return self.proxy.cspace_ranges

    @property
    def cspace_max_stepdiff(self):
        return self.proxy.cspace_max_stepdiff

    @property
    def cspace_max_stepdiff_proj(self):
        if isinstance(self._cspace_max_stepdiff_proj, (int, float)):
            return tuple(x * self._cspace_max_stepdiff_proj for x in self.cspace_max_stepdiff)
        return self._cspace_max_stepdiff_proj

    def gen_path(self, config1, config2):
        assert self.validate_config(config1) and self.validate_config(config2)

        config = config1
        path = [config]
        success = False

        for i in range(1000):
            config = config + np.fmin(self.cspace_max_stepdiff, np.array(config2) - np.array(config))
            config = self.constrain_config(config)

            diffs = [self.cspace_ranges[i].difference(path[-1][i], config[i]) for i in range(len(self.cspace_ranges))]
            valid = all(abs(diff) < constrain for diff, constrain in zip(diffs, self.cspace_max_stepdiff_proj))
            if valid:
                path.append(config)
                if np.allclose(config, config2):
                    success = True
                    break
                if self.distance(config, config2) >= self.distance(path[-1], config2) - self._gen_path_atol:
                    break
            else:
                break

        return success, path


class ProblemSpace(object):
    def __init__(self, cspace):
        self.cspace = cspace

    def sample(self, nr_trials=None):
        if nr_trials is None:
            nr_trials = self.sample.default_nr_trials
        for i in range(nr_trials):
            sample = self.cspace.sample()
            if self.validate_config(sample):
                return sample
        raise ValueError('Unable to produce a sample.')

    sample.default_nr_trials = 100

    def difference(self, config1, config2):
        return self.cspace.difference(config1, config2)

    def distance(self, config1, config2):
        return self.cspace.distance(config1, config2)

    def validate_config(self, config):
        return self.cspace.validate_config(config)

    def distance_path(self, path):
        distance = 0
        for i in range(len(path) - 1):
            distance += self.cspace.distance(path[i], path[i + 1])
        return distance

    def validate_path(self, config1, config2):
        success, _, _ = self.try_extend_path(config1, config2)
        return success

    def try_extend_path(self, start, end):
        assert self.validate_config(start) and self.validate_config(end)
        success, path = self.cspace.gen_path(start, end)
        if path is None:
            return False, start, path

        safe_path = []
        for configuration in path:
            if not self.validate_config(configuration):
                return False, safe_path[-1], safe_path
            safe_path.append(configuration)
        return success, safe_path[-1], safe_path


class CollisionFreeProblemSpace(ProblemSpace):
    def __init__(self, cspace):
        super().__init__(cspace)

    def validate_config(self, configuration):
        return self.cspace.validate_config(configuration) and not self.collide(configuration)

    def collide(self, configuration):
        raise NotImplementedError()
