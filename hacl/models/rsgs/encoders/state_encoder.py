#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from jactorch.nn import TorchApplyRecorderMixin
from torch import nn

from hacl.envs.gridworld.crafting_world.broadcast_engine import CraftingWorldBroadcastEngine
from hacl.envs.gridworld.crafting_world.v20210515 import CraftingWorldV20210515
from hacl.envs.simple_continuous.playroom_gdk.toyrobot_v20210423 import ToyRobotV20210423
from hacl.models.rsgs.encoders.craftingworld_state_encoder import CraftingWorldStateEncoder


class StateEncoder(TorchApplyRecorderMixin):
    def __init__(self, env_name=None, env_args=None, s_dim=None, h_dim=128, playroom_distance=False):
        super().__init__()
        self.env_name = env_name
        self.env_args = env_args
        self.s_dim = s_dim
        self.out_dim = s_dim
        if env_name == 'toyrobot':
            init_symbolic_state = ToyRobotV20210423(env_args).get_symbolic_state()
            self.playroom_distance = playroom_distance
            self.symbolic_state_dim = 3 + 2 + init_symbolic_state[1] * 3
            self.distance_dim = init_symbolic_state[1] if playroom_distance else 0
            if self.s_dim is not None:
                self.state_encoder = nn.Sequential(nn.Linear(self.symbolic_state_dim + self.distance_dim, self.s_dim), nn.ReLU())
            else:
                self.out_dim = self.symbolic_state_dim
        elif env_name == 'craftingworld':
            env_args = CraftingWorldV20210515.complete_env_args(env_args)
            self.broadcast_env = CraftingWorldBroadcastEngine(env_args=env_args)
            assert self.s_dim is not None
            self.state_encoder = CraftingWorldStateEncoder(h_dim=h_dim, out_dim=s_dim, agg_method='max')
        else:
            raise ValueError('Invalid env_name %s' % env_name)

    def forward(self, x):
        if self.s_dim is not None:
            if self.env_name == 'toyrobot':
                siz = x.size()[:-1]
                x = x.view(-1, x.size(-1))
                x = x[:, :self.symbolic_state_dim]

                if self.playroom_distance:
                    n_objects = (self.symbolic_state_dim - 5) // 3
                    assert self.symbolic_state_dim == n_objects * 3 + 5
                    distances = []
                    for i in range(n_objects):
                        distances.append(torch.sqrt(torch.square(x[:, :2] - x[:, 5 + 3 * i + 1:5 + 3 * i + 3]).sum(dim=1)))
                    distances = torch.stack(distances, dim=1)
                    y = torch.cat((x, distances), dim=1)
                else:
                    y = x
                y = self.state_encoder(y)
                y = y.view(*siz, y.size(-1))
                return y
            elif self.env_name == 'craftingworld':
                return self.state_encoder(x)
            else:
                raise ValueError()
        else:
            return x.type(torch.float)
