#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import random

import torch
from jactorch.nn import TorchApplyRecorderMixin

from hacl.p.rsgs.unified_broadcast_engine import UnifiedBroadcastEngine
from hacl.utils.math import gauss_log_prob, gauss_sample


class BehaviorPlanner(TorchApplyRecorderMixin):
    def __init__(self, env_name, env_args, behavior, label_encoder):
        super(BehaviorPlanner, self).__init__()
        self.env_name = env_name
        self.broadcast_engine = UnifiedBroadcastEngine(env_name, env_args)
        self.env = UnifiedBroadcastEngine(env_name, env_args).env
        self.behavior = behavior
        self.label_encoder = label_encoder
        self.continuous = env_name == 'toyrobot'

    def get_discrete_valid_actions(self, s_tensor, action_set):
        valid_actions = []
        valid_next_states = []
        for a in action_set:
            new_s_tensor, valid = self.broadcast_engine.action(s_tensor, a, inplace=False)
            if valid.item():
                valid_actions.append(a)
                valid_next_states.append(new_s_tensor)
        return valid_actions, valid_next_states

    def plan_path(self, state, behavior, action_set=None, embedding=None, policy='optimal', max_iters=100, **kwargs):
        s_tensor = self.broadcast_engine.states2tensor([state]).to(self.device)
        states, actions = [state], []
        prob = 0
        visited = set()

        hidden = behavior.get_zero_lstm_state(s_tensor.unsqueeze(1)) if behavior.use_lstm else None
        for it in range(max_iters):
            if not self.continuous:
                valid_actions, valid_next_states = self.get_discrete_valid_actions(s_tensor, action_set)

                outputs = behavior(s_tensor.unsqueeze(1), hidden=hidden, embeddings=embedding.view(1, -1) if embedding is not None else None)
                if behavior.use_lstm:
                    score, stop, hidden = outputs
                else:
                    score, stop = outputs
                score, stop = score.view(-1), stop.view(-1)
                score += stop[0]
                logprobs = torch.tensor([score[va] for va in valid_actions] + [stop[1]])
                if policy == 'optimal':
                    aid = logprobs.max(0)[1]
                elif policy == 'rational':
                    aid = torch.multinomial(torch.softmax(logprobs, dim=0), 1).item()
                elif policy.startswith('eps'):
                    eps = float(policy[3:])
                    aid = logprobs.max(0)[1]
                    if random.random() > eps and aid != logprobs.size(0) - 1:
                        aid = random.randint(0, logprobs.size(0) - 2)
                elif policy == 'distinct_optimal':
                    aid = logprobs.max(0)[1]
                    if aid != logprobs.size(0) - 1:
                        if (states[-1], int(valid_actions[aid])) in visited:
                            aid = random.randint(0, logprobs.size(0) - 2)
                        visited.add((states[-1], int(valid_actions[aid])))
                elif policy == 'random':
                    aid = random.randint(0, logprobs.size(0) - 2)
                else:
                    raise ValueError('Invalid policy %s' % policy)
                if it + 1 == max_iters:
                    aid = logprobs.size(0) - 1
                if aid == logprobs.size(0) - 1:
                    prob += logprobs[-1]
                    return states, actions, prob
                else:
                    prob += logprobs[aid]
                    s_tensor = valid_next_states[aid]
                    states.append(self.broadcast_engine.tensor2states(s_tensor, **kwargs)[0])
                    actions.append(valid_actions[aid])
            else:
                outputs = behavior(s_tensor.unsqueeze(1), hidden=hidden, embeddings=embedding.view(1, -1) if embedding is not None else None)
                if behavior.use_lstm:
                    (mu, var), stop, hidden = outputs
                else:
                    (mu, var), stop = outputs
                mu, var, stop = mu.view(-1), var.view(-1), stop.view(-1)
                action = None
                if policy == 'random':
                    if it + 1 == max_iters:
                        prob += stop[1]
                        action = None
                    else:
                        action = torch.rand(self.broadcast_engine.get_action_dim(), device=mu.device) * 1.98 - 0.99
                        prob += stop[0] + gauss_log_prob(mu, var, action).sum(-1)
                elif policy == 'rational':
                    if random.random() < torch.exp(stop[1]).item() or it + 1 == max_iters:
                        action = None
                        prob += stop[1]
                    else:
                        action = gauss_sample(mu, var)
                        action /= torch.abs(action).max() * 1.01
                        prob += stop[0] + gauss_log_prob(mu, var, gauss_log_prob).sum(-1)
                elif policy == 'optimal':
                    if torch.exp(stop[1]).item() > .5 or it + 1 == max_iters:
                        action = None
                        prob += stop[1]
                    else:
                        action = mu.clamp(-0.99, 0.99)
                        action = gauss_sample(mu, var)
                        action /= torch.abs(action).max() * 1.01
                        prob += stop[0] + gauss_log_prob(mu, var, action).sum(-1)
                else:
                    raise ValueError('Invalid policy %s' % policy)
                if action is None:
                    return states, actions, prob
                else:
                    s_tensor = self.broadcast_engine.action(s_tensor, action=action, inplace=False)[0]
                    states.append(self.broadcast_engine.tensor2states(s_tensor)[0])
                    actions.append(action)

    def plan_single(self, state, label, action_set=None, policy='optimal', **kwargs):
        states, actions, prob = self.plan_path(
            state,
            self.behavior,
            action_set=action_set,
            embedding=self.label_encoder([label]),
            policy=policy,
            **kwargs
        )
        return {'traj': (states, actions)}

    def plan(self, start_states, labels, **kwargs):
        trajs = []
        for start_state, label in zip(start_states, labels):
            trajs.append(self.plan_single(start_state, label, **kwargs))
        return trajs


class BehaviorPlannerWithStateMachine(BehaviorPlanner):
    def __init__(self, env_name, env_args, behaviors, state_machines):
        super(BehaviorPlanner, self).__init__()
        self.env_name = env_name
        self.broadcast_engine = UnifiedBroadcastEngine(env_name, env_args)
        self.env = UnifiedBroadcastEngine(env_name, env_args).env
        self.behaviors = behaviors
        self.state_machines = state_machines
        self.continuous = env_name == 'toyrobot'

    def plan_single(self, state, label, action_set=None, policy='optimal', max_iters=200, **kwargs):
        state_machine = self.state_machines[label]
        f = {x: None for x in state_machine.nodes}
        for x in state_machine.starts:
            f[x] = ([state], [], 0)
        topo_seq = state_machine.get_topological_sequence()
        d = {x: 0 for x in state_machine.nodes}
        critical_length = 1
        for u in topo_seq:
            for v, el in state_machine.adjs[u]:
                d[v] = max(d[v], d[u]) + 1
                critical_length = max(critical_length, d[v])
        for u in topo_seq:
            if f[u] is not None:
                cs, ca, cp = f[u]
                for v, el in state_machine.adjs[u]:
                    ps, pa, pp = self.plan_path(
                        cs[-1],
                        self.behaviors[el],
                        action_set=action_set,
                        policy=policy,
                        max_iters=round(max_iters / critical_length),
                        **kwargs
                    )
                    if pp is not None:
                        if f[v] is None or cp + pp > f[v][2]:
                            f[v] = (cs + ps[1:], ca + pa, cp + pp)
        bestp = None
        for x in state_machine.ends:
            if f[x] is not None and (bestp is None or f[x][2] > bestp):
                states, actions, bestp = f[x]
        if bestp is not None:
            return {'traj': (states, actions)}
        else:
            return {'traj': ([state], [])}
