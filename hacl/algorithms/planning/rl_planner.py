#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import random

import torch
from jactorch.nn import TorchApplyRecorderMixin

from hacl.p.rsgs.unified_broadcast_engine import UnifiedBroadcastEngine


class RLPlanner(TorchApplyRecorderMixin):
    def __init__(self, env_name, env_args, net, is_irl=False, is_irlcont=False, gamma=1):
        super().__init__()
        self.env_name = env_name
        self.broadcast_engine = UnifiedBroadcastEngine(env_name, env_args)
        self.net = net
        self.is_irl = is_irl
        self.is_irlcont = is_irlcont
        self.continuous = env_name == 'toyrobot'
        self.gamma = gamma

    def get_discrete_valid_actions(self, s_tensor, action_set):
        valid_actions = []
        valid_next_states = []
        for a in action_set:
            new_s_tensor, valid = self.broadcast_engine.action(s_tensor, a, inplace=False)
            if valid.item():
                valid_actions.append(a)
                valid_next_states.append(new_s_tensor)
        return valid_actions, valid_next_states

    def evaluate_actions(self, s_tensor, hidden=None, label=None):
        if not self.continuous:
            if self.is_irl:
                if self.net.use_lstm and hidden is None:
                    hidden_irl = self.net.get_zero_lstm_state(s_tensor.unsqueeze(0))
                    hidden_rl = self.net.rl_net.get_zero_lstm_state(s_tensor.unsqueeze(0))
                else:
                    hidden_irl, hidden_rl = hidden
                reward = self.net(s_tensor.unsqueeze(0), labels=[label], hidden=hidden_irl)
                if self.net.use_lstm:
                    reward, hidden_irl = reward
                    _, hidden_rl = self.net.rl_net(s_tensor.unsqueeze(0), labels=[label], hidden=hidden_rl)
                qvalues = []
                for a in range(reward.size(-1)):
                    new_s_tensor, valid = self.broadcast_engine.action(s_tensor.unsqueeze(0), a, inplace=False)
                    qvalue = self.net.rl_net(new_s_tensor, labels=[label], hidden=hidden_rl)
                    if self.net.use_lstm:
                        qvalue, _ = qvalue
                    qvalues.append(qvalue.max(-1)[0].unsqueeze(-1))
                a_tensor = reward + torch.stack(qvalues, dim=-1) * self.gamma
                if self.net.use_lstm:
                    return a_tensor, (hidden_irl, hidden_rl)
                return a_tensor, None
            elif self.is_irlcont:
                raise NotImplementedError()
            else:
                if self.net.use_lstm and hidden is None:
                    hidden = self.net.get_zero_lstm_state(s_tensor.unsqueeze(0))
                a_tensor = self.net(s_tensor.unsqueeze(0), labels=[label], hidden=hidden)
                if self.net.use_lstm:
                    a_tensor, hidden = a_tensor
                a_tensor = a_tensor.view(-1)
                return a_tensor, hidden

        else:
            raise NotImplementedError()

    def plan_single(self, state, label, action_set=None, action_cost=None, policy=None, **kwargs):
        states, actions = [state], []
        s_tensor = self.broadcast_engine.states2tensor([state]).to(self.net.device)
        hidden = None
        from tqdm import tqdm
        n_iters = 100
        for it in range(n_iters) if not self.is_irlcont else tqdm(range(n_iters), total=n_iters):
            if not self.is_irlcont:
                if policy != 'random':
                    a_tensor, hidden = self.evaluate_actions(s_tensor, hidden, label)
                    a_tensor = a_tensor.view(-1)
                    assert a_tensor.size(0) == len(action_set)
                else:
                    a_tensor = None
                valid_actions, valid_next_states = self.get_discrete_valid_actions(s_tensor, action_set)
                if len(valid_actions) > 0:
                    if policy == 'random':
                        aid = random.choice([i for i in range(len(valid_actions))])
                    elif policy == 'optimal':
                        aid = max([i for i in range(len(valid_actions))], key=lambda x: a_tensor[valid_actions[x]])
                    elif policy == 'rational':
                        q = torch.tensor([a_tensor[valid_actions[i]] for i in range(len(valid_actions))])
                        q = torch.softmax(q, dim=0)
                        aid = torch.multinomial(q, 1).item()
                    else:
                        raise ValueError()
                    action = valid_actions[aid]
                    s_tensor = valid_next_states[aid]
                    states.append(self.broadcast_engine.tensor2states(s_tensor, **kwargs)[0])
                    actions.append(action)
                else:
                    break
            else:
                if self.net.use_lstm and hidden is None:
                    hidden_reward = self.net.reward.get_zero_lstm_state(s_tensor.unsqueeze(0))
                    hidden_value = self.net.value.get_zero_lstm_state(s_tensor.unsqueeze(0))
                else:
                    hidden_reward, hidden_value = hidden
                _, hidden_value = self.net.value(s_tensor.unsqueeze(0), labels=[label], hidden=hidden_value)
                n = 10
                action_candidates = torch.normal(torch.zeros(n, 3, device=s_tensor.device), torch.ones(n, 3, device=s_tensor.device)).clip(-0.95, 0.95)
                action_values = []
                for ai in range(n):
                    action = action_candidates[ai]
                    new_s_tensor, valid = self.broadcast_engine.action(s_tensor, action, inplace=False)
                    if valid.item():
                        value, _ = self.net.value(new_s_tensor.unsqueeze(0), labels=[label], hidden=hidden_value)
                        reward, _ = self.net.reward(s_tensor.unsqueeze(0), labels=[label], hidden=hidden_reward, add=action.view(1, 1, -1))
                        action_values.append((action, reward + value * self.gamma))
                if len(action_values) ==0:
                    break
                aid = max(range(len(action_values)), key=lambda i: action_values[i][1])
                action = action_values[aid][0]
                new_s_tensor, valid = self.broadcast_engine.action(s_tensor, action, inplace=False)
                _, hidden_reward = self.net.reward(s_tensor.unsqueeze(0), labels=[label], hidden=hidden_reward, add=action.view(1, 1, -1))
                hidden = (hidden_reward, hidden_value)
                s_tensor = new_s_tensor

                states.append(self.broadcast_engine.tensor2states(s_tensor, **kwargs)[0])
                actions.append(action)
        return {'traj': (states, actions)}

    def plan(self, start_states, labels, **kwargs):
        results = []
        for plan_id, (start_state, label) in enumerate(zip(start_states, labels)):
            print('Plan %d/%d' % (plan_id + 1, len(start_states)))
            results.append(self.plan_single(start_state, label, **kwargs))
        return results
