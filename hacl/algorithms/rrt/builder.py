#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import random
from collections import defaultdict

import jacinle

from .rrt import RRT, RRTNode, traverse_rrt_bfs


class RRTizedEnv(object):
    def __init__(self):
        self.state2index = dict()
        self.edges = defaultdict(list)

    @property
    def states(self):
        return self.state2index.keys()

    @property
    def nr_states(self):
        return len(self.state2index)

    @jacinle.cached_property
    def index2state(self):
        return self.get_state_list_by_index()

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


def build_rrt(pspace, start_state=None, target_configs=None, nr_iterations=100):
    if start_state is None:
        start_state = pspace.sample()

    if target_configs is None:
        target_configs = []
    elif isinstance(target_configs, dict):
        target_configs = list(target_configs.values())

    rrt = RRT(pspace, RRTNode.from_states(start_state))
    it = 0
    while it < nr_iterations:
        if len(target_configs) > 0 and random.random() < .5:
            target_id = random.randint(0, len(target_configs) - 1)
            next_config = target_configs[target_id]
        else:
            target_id = None
            next_config = pspace.sample()
            it += 1

        node = rrt.nearest(next_config)
        success, next_config, _ = pspace.try_extend_path(node.config, next_config)
        if next_config is not None:
            new_node = rrt.extend(node, next_config)
        if target_id is not None and next_config == target_configs[target_id]:
            target_configs.pop(target_id)
    # print('remain=', len(target_configs))
    return rrt


def build_rrt_graph(pspace, rrt, traj_states, tqdm=False, full_graph=False):
    graph = RRTizedEnv()

    def add_edge(config1, config2):
        graph.add_edge(
            config1,
            config2,
            action=pspace.cspace.difference(config1, config2),
            c=pspace.cspace.distance(config1, config2),
        )

    all_rrt_nodes = tuple(traverse_rrt_bfs(rrt.roots))
    for node in all_rrt_nodes:
        graph.add_state(node.config)

    if not full_graph:
        for node in all_rrt_nodes:
            for child in node.children:
                add_edge(node.config, child.config)
                add_edge(child.config, node.config)

    if not full_graph:
        if tqdm:
            traj_states_iter = jacinle.tqdm(traj_states, desc='Building RRT Graph (step 2)')
        else:
            traj_states_iter = traj_states

        for node1_config in traj_states_iter:
            for node2 in all_rrt_nodes:
                success, next_config, _ = pspace.try_extend_path(node1_config, node2.config)
                if success:
                    add_edge(node1_config, node2.config)
                    add_edge(node2.config, node1_config)
        for node1_config, node2_config in zip(traj_states[:-1], traj_states[1:]):
            add_edge(node1_config, node2_config)
            add_edge(node2_config, node1_config)
    else:
        nodes = tuple(graph.states) + tuple(traj_states)
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if i != j:
                    success, _, _ = pspace.try_extend_path(nodes[i], nodes[j])
                    if success:
                        add_edge(nodes[i], nodes[j])

    return graph
