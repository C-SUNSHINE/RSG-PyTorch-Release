#! /usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
from torch import nn
from jactorch.nn import TorchApplyRecorderMixin
from hacl.envs.simple_continuous.playroom_gdk.toyrobot_v20210423 import ToyRobotV20210423
from hacl.envs.simple_continuous.playroom_gdk.broadcast_engine import ToyRobotBroadcastEngine
from hacl.envs.gridworld.crafting_world.v20210515 import CraftingWorldV20210515
from hacl.envs.gridworld.crafting_world.broadcast_engine import CraftingWorldBroadcastEngine


class UnifiedBroadcastEngine(TorchApplyRecorderMixin):
    def __init__(self, env_name=None, env_args=None):
        super().__init__()
        self.env_name = env_name
        self.env_args = env_args
        if env_name == 'toyrobot':
            self.broadcast_env = ToyRobotBroadcastEngine(env_args)
            self.env = ToyRobotV20210423(env_args)
        elif env_name == 'craftingworld':
            self.broadcast_env = CraftingWorldBroadcastEngine(env_args)
            self.env = CraftingWorldV20210515(env_args)
        else:
            raise ValueError('Invalid env_name %s' % env_name)

    def states2tensor(self, states, **kwargs):
        if self.env_name == 'toyrobot':
            return self.broadcast_env.states2tensor(states).to(self.device)
        elif self.env_name == 'craftingworld':
            return self.broadcast_env.states2tensor(states).to(self.device)

    def tensor2states(self, x, map_id=None, **kwargs):
        if self.env_name == 'toyrobot':
            return self.broadcast_env.tensor2states(x)
        elif self.env_name == 'craftingworld':
            return self.broadcast_env.tensor2states(x)

    @property
    def is_continuous(self):
        if self.env_name == 'toyrobot':
            return True
        elif self.env_name == 'craftingworld':
            return False

    def get_action_dim(self):
        if self.env_name == 'toyrobot':
            return 3
        elif self.env_name == 'craftingworld':
            return 5

    def sample_action(self, rng=None):
        if rng is None:
            import random

            rng = random.Random()
        if self.env_name == 'toyrobot':
            max_diff = self.env.pspace.cspace.cspace_max_stepdiff
            return torch.Tensor([(rng.random() * 2.0 - 1.0) * max_diff[i] for i in range(3)]).to(
                self.device
            )
        elif self.env_name == 'craftingworld':
            return rng.random(0, 4)

    def action(self, x, action, inplace=True):
        if not inplace:
            x = x.clone()
        if self.env_name == 'toyrobot':
            return self.broadcast_env.action(x, action, inplace=True)
        elif self.env_name == 'craftingworld':
            return self.broadcast_env.action(x, action, inplace=True)
