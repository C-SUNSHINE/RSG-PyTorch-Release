#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F

from hacl.p.rsgs.unified_broadcast_engine import UnifiedBroadcastEngine
from hacl.models.rsgs.behavior_cloning import BehaviorNet
from hacl.models.rsgs.encoders import StateEncoder, TrajEncoder
from hacl.models.rsgs.encoders.label_encoder import LabelEncoder
from hacl.models.rsgs.evaluators.evaluator import Evaluator
from hacl.utils.math import gauss_log_prob
from hacl.algorithms.planning.behavior_planner import BehaviorPlanner


class BC_Evaluator(Evaluator):
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

        self.broadcast_env = UnifiedBroadcastEngine(env_name, env_args)
        self.traj_encoder = TrajEncoder(env_name, env_args, require_encoder=False)
        self.action_dim = self.broadcast_env.get_action_dim()
        self.label_encoder = LabelEncoder(add_labels, h_dim=h_dim // 2)
        self.use_lstm = use_lstm
        self.h_dim = h_dim

        self.behavior = BehaviorNet(
            None,
            out_dim=self.action_dim,
            h_dim=self.h_dim,
            input_encoder_gen=lambda: StateEncoder(self.env_name, self.env_args, s_dim=self.h_dim),
            emb_dim=h_dim,
            use_lstm=self.use_lstm,
            continuous=self.continuous
        )

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
        super(BC_Evaluator, self).train(mode=mode)

    def get_planner(self):
        planner = BehaviorPlanner(env_name=self.env_name, env_args=self.env_args, behavior=self.behavior, label_encoder=self.label_encoder)
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
        if self.continuous:
            packed_actions = torch.stack([F.pad(actions, (0, 0, 0, max_len - actions.size(0))) for actions in actions_list], dim=0)
        else:
            packed_actions = torch.stack([F.pad(actions, (0, max_len - actions.size(0))) for actions in actions_list], dim=0)
        mask = torch.stack([torch.arange(max_len, device=packed_states.device).lt(length) for length in lens], dim=0)

        hidden = self.behavior.get_zero_lstm_state(packed_states) if self.use_lstm else None

        label_scores = []

        for label in labels:
            embeddings = self.label_encoder([label]).repeat(n, 1)
            if not self.continuous:
                outputs = self.behavior(packed_states, hidden=hidden, embeddings=embeddings)
                if self.use_lstm:
                    action_logprob, stop_logprob, hidden = outputs
                else:
                    action_logprob, stop_logprob = outputs
                real_action_logprob = action_logprob + stop_logprob[:, :, 0:1]
                action_scores = real_action_logprob.gather(2, packed_actions.unsqueeze(2)).squeeze(2)
                stop_scores = stop_logprob[:, :, 1].gather(1, torch.LongTensor(lens).to(self.device).unsqueeze(1)).squeeze(1)
                label_scores.append((action_scores * mask.to(torch.long)).sum(1) + stop_scores)
            else:
                outputs = self.behavior(packed_states, hidden=hidden, embeddings=embeddings)
                if self.use_lstm:
                    (action_mu, action_var), stop_logprob, hidden = outputs
                else:
                    (action_mu, action_var), stop_logprob = outputs
                action_scores = gauss_log_prob(action_mu, action_var, packed_actions).sum(-1) + stop_logprob[:, :, 0]
                stop_scores = stop_logprob[:, :, 1].gather(1, torch.LongTensor(lens).to(self.device).unsqueeze(1)).squeeze(1)
                label_scores.append((action_scores * mask.to(torch.long)).sum(1) + stop_scores)

        return torch.stack(label_scores, dim=1)
