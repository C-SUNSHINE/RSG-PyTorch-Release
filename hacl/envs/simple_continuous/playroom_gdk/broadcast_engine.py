#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from jactorch.nn import TorchApplyRecorderMixin

from hacl.envs.simple_continuous.playroom_gdk.configs import DEFAULT_ENV_ARGS_V1
from hacl.envs.simple_continuous.playroom_gdk.toyrobot_v20210423 import ToyRobotV20210423

REGION_DIM = 12
POSITION_DIM = 2
VARIABLE_DIM = 6
VALUE_DIM = 2


class ToyRobotBroadcastEngine(TorchApplyRecorderMixin):
    def __init__(self, env_args=None):
        super().__init__()
        self.dataset = env_args['dataset'] if 'dataset' in env_args else None
        self.env = ToyRobotV20210423(env_args=env_args)
        self.requires_grad_(False)

    @classmethod
    def _state2tensor(cls, state):
        flatten_state = list(state[0])
        flatten_state.extend(state[1:3])
        n_regions, n_env_variables = state[1], state[2]
        for (region_id, region_x, region_y) in state[3: 3 + n_regions]:
            flatten_state.extend((region_id, region_x, region_y))

        for (variable_id, value_id) in state[3 + n_regions: 3 + n_regions + n_env_variables]:
            flatten_state.extend((variable_id, value_id))

        return torch.tensor(flatten_state).view(-1)

    @classmethod
    def states2tensor(cls, states):
        res = torch.stack([cls._state2tensor(state) for state in states], dim=0)
        return res

    @classmethod
    def _tensor2state(cls, x):
        n_regions, n_env_variables = round(x[3].item()), round(x[4].item())
        state = [tuple(float(x[i].item()) for i in range(3)), n_regions, n_env_variables]
        for i in range(n_regions):
            region_id, region_x, region_y = x[5 + i * 3: 8 + i * 3]
            state.append((round(float(region_id)), float(region_x), float(region_y)))
        for i in range(n_env_variables):
            variable_id, value_id = x[5 + n_regions * 3 + i * 2: 7 + n_regions * 3 + i * 2]
            state.append((round(float(variable_id)), round(float(value_id))))
        return tuple(state)

    @classmethod
    def tensor2states(cls, x):
        states = [cls._tensor2state(x[i]) for i in range(x.size(0))]
        return states

    def _comperss_size(self, x):
        if len(x.size()) == 1:
            siz = None
        else:
            siz = x.size()[:-1]
        return x.view(-1, x.size(-1)), siz

    def _restore_size(self, x, siz):
        if siz is None:
            return x.view(*x.size()[1:])
        return x.view(*siz, *x.size()[1:])

    def check_symbols_aligning(self, x):
        x, siz = self._comperss_size(x)
        n_regions, n_env_variables = round(x[0, 3].item()), round(x[0, 4].item())
        assert x.size(1) == 5 + n_regions * 3 + n_env_variables * 2
        assert x[:, 3].max() - x[:, 3].min() < 1e-9
        assert x[:, 4].max() - x[:, 4].min() < 1e-9
        for i in range(n_regions):
            assert x[:, 5 + i * 3].max() - x[:, 5 + i * 3].min() < 1e-9
        for i in range(n_env_variables):
            assert x[:, 5 + n_regions * 3 + i * 2].max() - x[:, 5 + n_regions * 3 + i * 2].min() < 1e-9

    def action(self, x, action, inplace=True):
        if not inplace:
            x = x.clone()

        if isinstance(action, torch.Tensor) and len(action.size()) > 1:
            assert action.size()[:-1] == x.size()[:-1]
            actions = action.view(-1, action.size(-1))
        else:
            actions = None

        x, siz = self._comperss_size(x)
        states = self.tensor2states(x)
        new_states = []
        valid = []
        for si, state in enumerate(states):
            self.env.load_from_symbolic_state(state)
            _, done = self.env.action(action if actions is None else actions[si])
            valid.append(not done)
            new_states.append(self.env.get_symbolic_state())
        y = self.states2tensor(new_states).to(x.device)
        valid = torch.BoolTensor(valid).to(x.device)
        return self._restore_size(y, siz).type(torch.float), self._restore_size(valid, siz)


if __name__ == '__main__':
    benv = ToyRobotBroadcastEngine(env_args=DEFAULT_ENV_ARGS_V1['regions_empty'])
    states = [benv.env.get_symbolic_state() for i in range(3)]
    print(states)
    x = benv.states2tensor(states)
    print(x)
    new_states = benv.tensor2states(x)
    print(new_states)
