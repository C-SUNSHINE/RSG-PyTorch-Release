#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import torch

from jactorch.nn import TorchApplyRecorderMixin
from hacl.p.rsgs.unified_broadcast_engine import UnifiedBroadcastEngine
from hacl.models.rsgs.encoders import StateEncoder


class TrajEncoder(TorchApplyRecorderMixin):
    def __init__(self, env_name=None, env_args=None, s_dim=None, h_dim=128, require_encoder=True):
        super().__init__()
        self.env_name = env_name
        self.env_args = env_args
        self.s_dim = s_dim
        self.h_dim = h_dim
        self.broadcast_env = UnifiedBroadcastEngine(env_name, env_args)
        self.require_encoder = require_encoder
        if require_encoder:
            self.state_encoder = StateEncoder(env_name, env_args, s_dim=s_dim, h_dim=h_dim)

    def traj_to_tensor(self, traj, map_id=None, **kwargs):
        if self.env_name == 'toyrobot':
            x = self.broadcast_env.states2tensor(traj[0]).to(self.device)
            assert x.size(0) == len(traj[0])
        elif self.env_name == 'craftingworld':
            x = self.broadcast_env.states2tensor(traj[0]).to(self.device)
        else:
            raise ValueError()
        return x

    def traj_to_action_tensor(self, traj, map_id=None, **kwargs):
        if self.env_name == 'toyrobot':
            # actions = []
            # for config1, config2 in zip(traj[:-1], traj[1:]):
            #     actions.append(
            #         [
            #             self.broadcast_env.env.pspace.cspace.cspace_ranges[j].difference(config1[j], config2[j])
            #             for j in range(3)
            #         ]
            #     )
            x = torch.Tensor(traj[1]).to(self.device)
            assert x.size(0) == len(traj[0]) - 1 and x.size(1) == 3
        elif self.env_name == 'craftingworld':
            x = torch.LongTensor(traj[1]).to(self.device)
        else:
            raise ValueError()
        return x

    def forward(self, trajs, **kwargs):
        assert self.require_encoder
        res = []
        lengths = []
        for traj in trajs:
            x = self.traj_to_tensor(traj, **kwargs)
            res.append(x)
            lengths.append(x.size(0))
        if self.s_dim is not None:
            encodings = self.state_encoder(torch.cat(res, 0))
            res = []
            ptr = 0
            for i, length in enumerate(lengths):
                res.append(encodings[ptr : ptr + length])
                ptr += length
        return res
