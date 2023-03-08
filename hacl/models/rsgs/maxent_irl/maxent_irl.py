#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn
from hacl.models.rsgs.dqn import DQN
from hacl.models.rsgs.a2c import A2C


class MaxEntIRL(nn.Module):
    def __init__(
        self,
        in_dim,
        action_dim,
        h_dim=256,
        input_encoder_gen=lambda: None,
        label_encoder_gen=lambda: None,
        use_lstm=False,
        continuous=False,
        rl_model='dqn',
        reward_activation='linear',
        reward_logprob=False,
    ):
        super().__init__()
        self.rl_model = rl_model
        self.continuous = continuous
        self.use_lstm = use_lstm
        assert self.rl_model in ['dqn', 'a2c']
        assert self.rl_model == 'a2c' or not continuous
        self.h_dim = h_dim
        if self.rl_model == 'dqn':
            self.dqn = DQN(
                in_dim,
                action_dim,
                h_dim=h_dim,
                input_encoder_gen=input_encoder_gen,
                label_encoder_gen=label_encoder_gen,
                use_lstm=use_lstm,
            )
        elif self.rl_model == 'a2c':
            self.a2c = A2C(
                in_dim,
                action_dim,
                h_dim=h_dim,
                input_encoder_gen=input_encoder_gen,
                label_encoder_gen=label_encoder_gen,
                use_lstm=use_lstm,
                continuous=continuous,
            )

        if not self.continuous:
            self.reward_net = DQN(
                in_dim,
                action_dim,
                h_dim=h_dim,
                input_encoder_gen=input_encoder_gen,
                label_encoder_gen=label_encoder_gen,
                use_lstm=use_lstm,
                activation=reward_activation,
                log_normalize=reward_logprob,
            )
        else:
            self.reward_net = DQN(
                in_dim,
                1,
                h_dim=h_dim,
                input_encoder_gen=input_encoder_gen,
                label_encoder_gen=label_encoder_gen,
                use_lstm=use_lstm,
                activation=reward_activation,
                add_dim=action_dim
            )

    def get_zero_lstm_state(self, x):
        h = torch.zeros(1, x.size(0), self.h_dim, device=x.device)
        c = torch.zeros(1, x.size(0), self.h_dim, device=x.device)
        return h, c

    def forward(self, x, *args, **kwargs):
        if self.rl_model == 'dqn':
            if self.use_lstm:
                res, hidden = self.reward_net(x, *args, **kwargs)
                return res.squeeze(-1), hidden
            else:
                res = self.reward_net(x, *args, **kwargs)
                return res.squeeze(-1)
        elif self.rl_model == 'a2c':
            y = args[0]
            args = args[1:]
            if self.use_lstm:
                res, hidden = self.reward_net(x, *args, **kwargs, add=y)
                return res.squeeze(-1), hidden
            else:
                res = self.reward_net(x, *args, **kwargs, add=y)
                return res.squeeze(-1)


    @property
    def rl_net(self):
        if self.rl_model == 'dqn':
            return self.dqn
        elif self.rl_model == 'a2c':
            return self.a2c

    @property
    def device(self):
        return self.rl_net.device