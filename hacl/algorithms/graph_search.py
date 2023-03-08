#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import defaultdict
from heapq import heappop, heappush
from typing import List, Dict, Any

import torch

from hacl.models.rsgs.state_machine import StateMachine

__all__ = ['GraphizedEnv', 'NoPathFoundException', 'build_graph_bfs', 'dijkstra', 'value_iteration']


class GraphizedEnv(object):
    def __init__(self, actions=None):
        self.state2index = dict()
        self.actions = actions
        self.edges = defaultdict(list)

    @property
    def states(self):
        return self.state2index.keys()

    @property
    def nr_states(self):
        return len(self.states)

    @property
    def nr_actions(self):
        return len(self.actions)

    def get_state_list_by_index(self):
        index2state = {v: k for k, v in self.state2index.items()}
        return [index2state[i] for i in range(self.nr_states)]

    def add_state(self, state):
        if state in self.state2index:
            return self.state2index[state]
        self.state2index[state] = len(self.state2index)
        return len(self.state2index) - 1

    def add_edge(self, x, y, action, c):
        x = self.add_state(x)
        y = self.add_state(y)
        self.edges[x].append((y, action, c))

    def get_by_ends(self, x, y):
        for e in self.edges[x]:
            if e[0] == y:
                return e
        return None

    def get_by_index(self, index):
        return self.edges.get(index, tuple())


def build_graph_bfs(env, max_steps, action_space):
    init_ctx = env.clone_ctx()
    queue = [(init_ctx, env.get_symbolic_state(), max_steps)]
    visited = set()
    graph = GraphizedEnv(action_space)

    while len(queue):
        ctx, sym_state, remain_steps = queue[0]
        queue = queue[1:]
        if sym_state in visited:
            continue
        if remain_steps <= 0:
            continue

        visited.add(sym_state)

        for action in action_space:
            env.overwrite_ctx(ctx)
            reward, _ = env.action(action)
            next_ctx = env.clone_ctx()
            next_sym_state = env.get_symbolic_state()

            graph.add_edge(sym_state, next_sym_state, action, reward)
            queue.append((next_ctx, next_sym_state, remain_steps - 1))

    env.overwrite_ctx(init_ctx)
    return graph


class NoPathFoundException(Exception):
    pass


def dijkstra(g: GraphizedEnv, source, target):
    # TODO: reward or cost?
    source = g.state2index[source]
    target = g.state2index[target]

    q, visited, distances = [(0, source, ())], set(), {source: 0}
    while q:
        (cost, u, path) = heappop(q)
        if u not in visited:
            visited.add(u)
            path = (u, path)
            if u == target:
                return (cost, path)

            for v, _, c in g.get_by_index(u, ()):
                if v in visited:
                    continue
                prev = distances.get(v, None)
                next = cost + c
                if prev is None or next < prev:
                    distances[v] = next
                    heappush(q, (next, v, path))

    raise NoPathFoundException()


def value_iteration(g, actions, init_value, nr_iters=10, gamma=1):
    value = init_value.clone()
    transition = torch.zeros(g.nr_states, len(actions), dtype=torch.int64, device=init_value.device)
    transition_cost = torch.zeros(g.nr_states, len(actions), dtype=torch.float32, device=init_value.device)
    transition_mask = torch.zeros(g.nr_states, len(actions), dtype=torch.float32, device=init_value.device)

    for u in range(len(g.states)):
        for v, action, cost in g.get_by_index(u):
            transition[u, action] = v
            transition_cost[u, action] = cost
            transition_mask[u, action] = 1

    for i in range(nr_iters):
        value = (
            (transition_cost + gamma * index_value(value, transition)) * transition_mask
            + (1 - transition_mask) * (-1e9)
        ).max(dim=-1)[0]

    return value


def value_iteration_multistep(
    g: GraphizedEnv, actions: List[int], init_values: List[torch.Tensor], nr_iters=30, gamma=1
):
    goal_init_value = init_values[-1]

    transition = torch.zeros(g.nr_states, len(actions), dtype=torch.int64, device=goal_init_value.device)
    transition_cost = torch.zeros(g.nr_states, len(actions), dtype=torch.float32, device=goal_init_value.device)
    transition_mask = torch.zeros(g.nr_states, len(actions), dtype=torch.float32, device=goal_init_value.device)

    for u in range(len(g.states)):
        for v, action, cost in g.get_by_index(u):
            transition[u, action] = v
            transition_cost[u, action] = cost
            transition_mask[u, action] = 1

    nr_steps = len(init_values)
    values = [None for _ in range(nr_steps)]
    values[-1] = init_values[-1]

    for i in range(nr_iters):
        next_values = [None for _ in range(nr_steps)]
        for j in range(nr_steps):
            if values[j] is not None:
                value = values[j]
                next_value = (
                    (transition_cost + gamma * index_value(value, transition)) * transition_mask
                    + (1 - transition_mask) * (-1e9)
                ).max(dim=-1)[0]
                next_values[j] = torch.max(next_value, next_values[j]) if next_values[j] is not None else next_value

                if j != 0:
                    next_value = (
                        (
                            index_value(init_values[j - 1], transition)
                            + transition_cost
                            + gamma * index_value(value, transition)
                        )
                        * transition_mask
                        + (1 - transition_mask) * (-1e9)
                    ).max(dim=-1)[0]
                    next_values[j - 1] = (
                        torch.max(next_value, next_values[j - 1]) if next_values[j - 1] is not None else next_value
                    )
        values = next_values

    return values


def value_iteration_state_machine_transition(
    transition,
    transition_mask,
    transition_cost,
    init_values: Dict[Any, torch.Tensor],
    state_machine: StateMachine,
    nr_iters=30,
    gamma=1,
):
    transition_mask = transition_mask.type(torch.float)
    edges = set(state_machine.label2edge.keys())
    values = {e: None for e in edges}
    step_values = {e: None for e in edges}
    for end in state_machine.ends:
        # print('end', end)
        for u, e in state_machine.radjs[end]:
            values[(u, e)] = step_values[(u, e)] = init_values[e]

    r_topo_seq = list(reversed(state_machine.get_topological_sequence()))

    for i in range(nr_iters):
        next_values = {e: values[e].clone() if values[e] is not None else None for e in edges}

        for x in r_topo_seq:
            # print(x)
            for v, el in state_machine.adjs[x]:
                if values[(x, el)] is not None:
                    value = values[(x, el)]
                    next_value = (
                        (transition_cost + gamma * index_value(value, transition)) * transition_mask
                        + (1 - transition_mask) * (-1e9)
                    ).max(dim=-1)[0]
                    next_values[(x, el)] = (
                        torch.max(next_value, next_values[(x, el)]) if next_values[(x, el)] is not None else next_value
                    )
                    # if x == 0:
                    #     print(value, ((transition_cost + gamma * index_value(value, transition)) * transition_mask + (
                    #             1 - transition_mask) * (-1e9)))
                    #     input()

                    for u, pel in state_machine.radjs[x]:
                        next_value = init_values[pel] + value
                        next_values[(u, pel)] = (
                            torch.max(next_value, next_values[(u, pel)])
                            if next_values[(u, pel)] is not None
                            else next_value
                        )
                        step_values[(u, pel)] = (
                            torch.max(step_values[(u, pel)], next_value)
                            if step_values[(u, pel)] is not None
                            else next_value
                        )
        # TODO make the complexity O(m) but not O(n^2+m)
        values = next_values
        # print(values)
        # input()
    return values, step_values


def value_iteration_state_machine(
    g: GraphizedEnv,
    actions: List[int],
    init_values: Dict[Any, torch.Tensor],
    state_machine: StateMachine,
    nr_iters=30,
    gamma=1,
):
    sample_init_value = list(init_values.values())[0]
    transition = torch.zeros(g.nr_states, len(actions), dtype=torch.long, device=sample_init_value.device)
    transition_cost = torch.zeros(g.nr_states, len(actions), dtype=torch.float32, device=sample_init_value.device)
    transition_mask = torch.zeros(g.nr_states, len(actions), dtype=torch.float32, device=sample_init_value.device)

    for u in range(g.nr_states):
        for v, action, cost in g.get_by_index(u):
            transition[u, action] = v
            transition_cost[u, action] = cost
            transition_mask[u, action] = 1
    return value_iteration_state_machine_transition(
        transition, transition_mask, transition_cost, init_values, state_machine, nr_iters, gamma
    )


def index_value(value, indices):
    return value[indices.view(-1)].view(indices.shape)
