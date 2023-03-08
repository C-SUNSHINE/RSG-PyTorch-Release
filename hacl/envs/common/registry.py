#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from jacinle.utils.registry import CallbackRegistry

__all__ = ['register_env', 'get_env', 'get_env_builder']


class EnvRegistry(CallbackRegistry):
    pass


env_registry = EnvRegistry()


def register_env(name, cls):
    env_registry.register(name, cls)


def get_env(name, *args, **kwargs):
    return env_registry.dispatch(name, *args, **kwargs)


def get_env_builder(name):
    return env_registry.lookup(name)
