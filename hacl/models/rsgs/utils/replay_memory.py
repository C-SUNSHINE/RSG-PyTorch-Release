#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'label'))


class TransitionReplayMemory(object):
    def __init__(self, capacity=100, has_expert=False, expert_capacity=None):
        self.transitions = []
        self.records = []
        self.position = 0
        self.capacity = capacity
        self.has_expert = has_expert
        if self.has_expert:
            self.expert_memory = TransitionReplayMemory(capacity=expert_capacity, has_expert=False)

    def push_expert(self, *args, **kwargs):
        assert self.has_expert
        self.expert_memory.push(*args, **kwargs)

    def push(self, *args, record=None, **kwargs):
        if len(self.transitions) < self.capacity:
            self.transitions.append(Transition(*args))
            self.records.append(record)
        else:
            self.transitions[self.position] = Transition(*args)
            self.records[self.position] = record
        self.position = (self.position + 1) % self.capacity

    def sample(self, n, rho=1, record=False):
        """
        Args:
            n: number of samples
            rho: the probability to sample from self exploration.
        Returns: (states, actions, next_states, rewards)
        """
        assert n > 0
        if self.has_expert:
            m = round(n * rho)
        else:
            m = n
        if m > 0:
            if len(self.transitions) < m:
                return None
            index = torch.randint(0, len(self.transitions), (m,))
            trans = [self.transitions[index[i]] for i in range(m)]
            records = [self.records[index[i]] for i in range(m)]
            if n - m > 0:
                exp_trans = self.expert_memory.sample(n - m, record=record)
                if exp_trans is None:
                    return None
                if record:
                    exp_trans, exp_records = exp_trans
                    records.extend(exp_records)
                trans.extend(exp_trans)
            if record:
                return trans, records
            else:
                return trans
        else:
            return self.expert_memory.sample(n, record=record)


Episode = namedtuple('Episode', ('states', 'actions', 'rewards', 'label'))


class EpisodeReplayMemory(object):
    def __init__(self, capacity=100, has_expert=False, expert_capacity=None):
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.episodes = []
        self.records = []
        self.position = 0
        self.capacity = capacity
        self.weights = torch.zeros(capacity, dtype=torch.float, device=device)
        self.has_expert = has_expert
        if self.has_expert:
            self.expert_memory = EpisodeReplayMemory(capacity=expert_capacity, has_expert=False)

    def push(self, ep, record=None, **kwargs):
        length = ep.actions.size(0)
        if len(self.episodes) < self.capacity:
            self.episodes.append(ep)
            self.records.append(record)
        else:
            self.episodes[self.position] = ep
            self.records[self.position] = record
        self.weights[self.position] = length
        self.position = (self.position + 1) % self.capacity

    def push_expert(self, ep, **kwargs):
        assert self.has_expert
        self.expert_memory.push(ep, **kwargs)

    def sample_episodes(self, n, rho=1, record=False):
        assert n > 0
        if self.has_expert:
            m = round(n * rho)
        else:
            assert rho == 1
            m = n
        if m > 0:
            if len(self.episodes) < m:
                return None
            indices = torch.multinomial(self.weights[: len(self.episodes)], m)
            episodes = [self.episodes[i] for i in indices]
            records = [self.records[indices[i]] for i in range(m)]
            if n - m > 0:
                exp_episodes = self.expert_memory.sample_episodes(n - m, record=record)
                if exp_episodes is None:
                    return None
                if record:
                    exp_episodes, exp_records = exp_episodes
                    records.extend(exp_records)
                episodes.extend(exp_episodes)
            if record:
                return episodes, records
            else:
                return episodes
        else:
            return self.expert_memory.sample_episodes(n, record=record)

    def sample_transitions(self, n, rho=1):
        assert n > 0
        if self.has_expert:
            m = round(n * rho)
        else:
            assert rho == 1
            m = n
        if m > 0:
            if len(self.episodes) == 0:
                return None
            episode_indices = torch.multinomial(self.weights[: len(self.episodes)], m, replacement=True)
            transitions = []
            for i in episode_indices:
                episode = self.episodes[i]
                length = episode.actions.size(0)
                j = torch.randint(length, size=(1,)).item()
                transitions.append(
                    Transition(
                        episode.states[j],
                        episode.actions[j],
                        episode.states[j + 1] if j + 1 < episode.states.size(0) else None,
                        episode.rewards[j],
                        episode.label,
                    )
                )
            if n - m > 0:
                exp_transitions = self.expert_memory.sample_transitions(n - m)
                if exp_transitions is None:
                    return None
                transitions.extend(exp_transitions)
            return transitions
        else:
            return self.expert_memory.sample_transitions(n)
