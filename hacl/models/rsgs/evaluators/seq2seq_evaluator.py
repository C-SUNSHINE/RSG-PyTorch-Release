#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
from torch import nn

from hacl.algorithms.planning.behavior_planner import BehaviorPlannerWithStateMachine
from hacl.p.rsgs.unified_broadcast_engine import UnifiedBroadcastEngine
from hacl.models.rsgs.behavior_cloning import BehaviorNet
from hacl.models.rsgs.encoders import StateEncoder, TrajEncoder
from hacl.models.rsgs.evaluators.evaluator import Evaluator
from hacl.models.rsgs.state_machine import StateMachine
from hacl.utils.math import gauss_log_prob


class Seq2Seq_Evaluator(Evaluator):
    _EXTRA_DICT_KEY = []

    def __init__(
        self,
        env_name='gridworld',
        env_args=None,
        add_labels=None,
        use_lstm=False,
        h_dim=256,
    ):
        super().__init__()
        if env_args is None:
            env_args = dict()
        self.env_args = env_args
        self.env_name = env_name

        if env_name in ['craftingworld']:
            self.continuous = False
        elif env_name in ['toyrobot']:
            self.continuous = True
        else:
            raise ValueError('Invalid env_name %s' % env_name)

        assert not use_lstm

        self.broadcast_env = UnifiedBroadcastEngine(env_name, env_args)
        self.traj_encoder = TrajEncoder(env_name, env_args, require_encoder=False)
        self.action_dim = self.broadcast_env.get_action_dim()
        self.labels = []
        self.state_machines = dict()
        self.use_lstm = use_lstm
        self.h_dim = h_dim

        self.behaviors = nn.ModuleDict()

        if add_labels is not None:
            for label in add_labels:
                self.add_label(label, exist_ok=True)

    def get_training_parameters(self):
        if self.training:
            return self.parameters()
        return None

    @property
    def qvalue_based(self):
        return False

    def extra_state_dict(self):
        return {key: getattr(self, key) for key in self._EXTRA_DICT_KEY}

    def load_extra_state_dict(self, extra_dict):
        for key in self._EXTRA_DICT_KEY:
            setattr(self, key, extra_dict[key])

    def train(self, mode=True):
        super(Seq2Seq_Evaluator, self).train(mode=mode)

    def add_label(self, label, state_machine=None, exist_ok=False, **kwargs):
        if label in self.labels:
            if not exist_ok:
                raise ValueError("label already added")
            return
        if state_machine is None:
            from hacl.models.rsgs.state_machine.builders import default_primitive_constructor
            state_machine = StateMachine.from_expression(label, primitive_constructor=default_primitive_constructor)
        self.labels.append(label)
        self.state_machines[label] = state_machine
        self.state_machines[label].update_potential()
        for el in state_machine.get_edge_label_set():
            if el not in self.behaviors:
                self.behaviors[el] = BehaviorNet(
                    None,
                    out_dim=self.action_dim,
                    h_dim=self.h_dim,
                    input_encoder_gen=lambda: StateEncoder(self.env_name, self.env_args, s_dim=self.h_dim),
                    use_lstm=self.use_lstm,
                    continuous=self.continuous
                )

    def get_planner(self):
        planner = BehaviorPlannerWithStateMachine(env_name=self.env_name, env_args=self.env_args, behaviors=self.behaviors, state_machines=self.state_machines)
        return planner.to(self.device)

    def plan(self, start_states, labels, action_set=None, **kwargs):
        planner = self.get_planner()
        trajs = planner.plan(
            start_states, labels,
            action_set=action_set,
            **kwargs
        )
        return trajs

    def forward(self, trajs, labels=None, training=False, progress=None, **kwargs):
        n = len(trajs)
        states_list = [self.traj_encoder.traj_to_tensor(traj, **kwargs) for traj in trajs]
        actions_list = [self.traj_encoder.traj_to_action_tensor(traj, **kwargs) for traj in trajs]

        lens = [int(actions.size(0)) for actions in actions_list]
        max_len = max(lens) + 1
        packed_states = torch.stack([F.pad(states, (0, 0, 0, max_len - states.size(0))) for states in states_list], dim=0)
        if not self.continuous:
            packed_actions = torch.stack([F.pad(actions, (0, max_len - actions.size(0))) for actions in actions_list], dim=0)
        else:
            packed_actions = torch.stack([F.pad(actions, (0, 0, 0, max_len - actions.size(0))) for actions in actions_list], dim=0)
        mask = torch.stack([torch.arange(max_len, device=packed_states.device).lt(length) for length in lens], dim=0)

        hidden = None

        label_scores = []

        for label in labels:
            state_machine = self.state_machines[label]
            action_logprobs = dict()
            stop_logprobs = dict()
            for el in state_machine.get_edge_label_set():
                if not self.continuous:
                    outputs = self.behaviors[el](packed_states, hidden=hidden)
                    if self.use_lstm:
                        action_logprob, stop_logprob, hidden = outputs
                    else:
                        action_logprob, stop_logprob = outputs
                    action_logprobs[el] = action_logprob.gather(2, packed_actions.unsqueeze(2)).squeeze(2)
                    stop_logprobs[el] = stop_logprob
                else:
                    outputs = self.behaviors[el](packed_states, hidden=hidden)
                    if self.use_lstm:
                        (action_mu, action_var), stop_logprob, hidden = outputs
                    else:
                        (action_mu, action_var), stop_logprob = outputs
                    action_logprobs[el] = gauss_log_prob(action_mu, action_var, packed_actions).sum(-1)
                    stop_logprobs[el] = stop_logprob
                    # print(action_mu.size(), action_var.size(), packed_actions.size())
                    # print(action_logprobs[el].size(), stop_logprobs[el].size())
            merged_action_log_probs = self.merge_sequences(action_logprobs, stop_logprobs, state_machine)
            label_scores.append(merged_action_log_probs.gather(1, torch.LongTensor(lens).to(self.device).unsqueeze(1)).squeeze(1))

        return torch.stack(label_scores, dim=1)

    def merge_sequences(self, action_logprobs, stop_logprobs, state_machine):
        f0 = torch.zeros_like(list(action_logprobs.values())[0]) - 1e9
        f0 = F.pad(f0, pad=(1, 0))
        f = {x: f0.clone() for x in state_machine.nodes}
        topo_seq = state_machine.get_topological_sequence()
        for u in topo_seq:
            for v, el in state_machine.adjs[u]:
                f[v] = torch.max(f[v], self.sequence_transfer(f[u], action_logprobs[el] + stop_logprobs[el][:, :, 0], stop_logprobs[el][:, :, 1]))
        res = None
        for end in state_machine.ends:
            if res is None:
                res = f[end]
            else:
                res = torch.max(res, f[end])
        return res

    def sequence_transfer(self, f, step_probs, transit_probs):
        s = F.pad(torch.cumsum(step_probs, dim=1), pad=(1, 0))
        r = s.unsqueeze(1) - s.unsqueeze(2)
        mask = (torch.arange(s.size(1), device=f.device).unsqueeze(0) - torch.arange(s.size(1), device=f.device).unsqueeze(1)).gt(0)
        r = r * mask.type(torch.long).unsqueeze(0) - 1e9 * (~mask).type(torch.long).unsqueeze(0)
        t = F.pad(transit_probs, pad=(1, 0))
        nf = (f.unsqueeze(2) + r + t.unsqueeze(1)).max(1)[0]
        return nf
