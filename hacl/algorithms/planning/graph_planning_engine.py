#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import random
from math import pi
from multiprocessing.pool import Pool
from queue import PriorityQueue

import torch
from tqdm import tqdm

from hacl.algorithms.rrt.builder import build_rrt, build_rrt_graph
from hacl.envs.simple_continuous.playroom_gdk.toyrobot_v20210423 import ToyRobotV20210423


class GraphPlanningEngine(object):
    class Node(object):
        def __init__(self, state, t, heuristic=0, parent=None, action=None):
            self.state = state
            self.t = t
            self.heuristic = heuristic
            self.parent = parent
            self.action = action

    class StepAction(object):
        def __init__(self, x, y):
            self.x = x
            self.y = y

    def __init__(
        self, env_name, env_args, broadcast_env, init_values, label, state_machine, action_cost=None, **kwargs
    ):
        self.env_name = env_name
        self.env_args = env_args
        self.broadcast_env = broadcast_env
        self.env = self.broadcast_env.env
        self.env_args = self.broadcast_env.env.complete_env_args(self.env_args)
        init_symbolic_state = self.env.get_symbolic_state()
        self.symbolic_state_dim = 3 + 2 + init_symbolic_state[1] * 3
        self.init_values = init_values
        self.label = label
        self.state_machine = state_machine
        self.action_cost = action_cost
        assert self.action_cost > 0
        self.kept_args = kwargs

        self.st2node = None
        self.heap = None
        self.openings = None
        self.closed = None

    def get_states_tensor(self, states, init_symbolic_state=None):
        if init_symbolic_state is None:
            init_symbolic_state = self.env.get_symbolic_state()
        states_tensor = self.broadcast_env.states2tensor([(state,) + init_symbolic_state[1:] for state in states]).to(self.broadcast_env.device)
        states_tensor = states_tensor[:, :self.symbolic_state_dim]
        return states_tensor

    def dijkstra(self, dist, parent, graph):
        q = PriorityQueue()
        for x in range(dist.size(0)):
            q.put((-dist[x], x))
        while not q.empty():
            t, x = q.get()
            t = -t
            if t - 1e-9 > dist[x]:
                continue
            for (y, a, c) in graph.edges[x]:
                if dist[x] - c * self.action_cost > dist[y]:
                    dist[y] = dist[x] - c * self.action_cost
                    parent[y] = x
                    q.put((-dist[y], y))
        return dist, parent

    def search(self, start_state, graph, rrt, use_tqdm=True, use_point_net=False, **kwargs):
        self.env.load_from_symbolic_state(start_state)
        start_node = start_state[0]
        path = []
        states = graph.get_state_list_by_index()
        states_tensor = self.get_states_tensor(states)
        dist = {t: torch.zeros(states_tensor.size(0)) - 1e9 for t in self.state_machine.nodes}
        parent = {t: torch.zeros(states_tensor.size(0), dtype=torch.int) - 1 for t in self.state_machine.nodes}
        prev = {t: torch.zeros(states_tensor.size(0), dtype=torch.int) - 1 for t in self.state_machine.nodes}

        for start in self.state_machine.starts:
            dist[start][graph.state2index[start_node]] = 0
        topo_seq = self.state_machine.get_topological_sequence()
        if use_tqdm:
            topo_seq = tqdm(topo_seq, total=len(topo_seq))
        for x in topo_seq:
            dist[x], parent[x] = self.dijkstra(dist[x], parent[x], graph)
            for y, el in self.state_machine.adjs[x]:
                step_value = self.init_values(el, states_tensor)
                for i in range(states_tensor.size(0)):
                    if dist[x][i] + step_value[i] > dist[y][i]:
                        dist[y][i] = dist[x][i] + step_value[i]
                        parent[y][i] = -1
                        prev[y][i] = x

        z, k, best = None, None, None
        for end in self.state_machine.ends:
            for i in range(states_tensor.size(0)):
                if best is None or dist[end][i] > best:
                    best = dist[end][i]
                    z, k = end, i

        while True:
            path.append(states[k])
            if parent[z][k] != -1:
                k = int(parent[z][k])
            elif prev[z][k] != -1:
                z = int(prev[z][k])
            else:
                break
        path.append(start_node)
        path = list(reversed(path))
        cpath = [path[0]]
        for config in path[1:]:
            cpath.extend(self.env.pspace.cspace.gen_path(cpath[-1], config)[1][1:])
        return [(tuple(node),) + start_state[1:] for node in cpath], None

    @classmethod
    def build_rrt_with_point_targets(cls, broadcast_engine, env_args, init_value, start_states, use_point_net=False):
        input_args_list = [
            (
                env_args, start_state, (get_target_configs(broadcast_engine, init_value, start_state) if use_point_net else None)
            ) for start_state in start_states
        ]
        target_configs_list = [input_args[2] for input_args in input_args_list]
        if len(start_states) < 5:  # Single processing
            graphs = []
            rrts = []
            for input_args in input_args_list:
                rrt, graph = build_rrt_graph_from_init_value(input_args)
                rrts.append(rrt)
                graphs.append(graph)
        else:  # Multi processing
            with Pool(min(len(start_states), 20)) as pool:
                rrt_graphs = list(pool.map(build_rrt_graph_from_init_value, input_args_list))
            rrts = [x[0] for x in rrt_graphs]
            graphs = [x[1] for x in rrt_graphs]
        return graphs, rrts, target_configs_list


def build_rrt_graph_from_init_value(args):
    env_args, start_state, target_configs = args

    start_config = start_state[0]
    env = ToyRobotV20210423(env_args)
    env.load_from_symbolic_state(start_state)
    rrt = build_rrt(env.pspace, start_state=start_config, target_configs=target_configs, nr_iterations=200)
    graph = build_rrt_graph(env.pspace, rrt, [], tqdm=False, full_graph=False)
    return rrt, graph


def get_target_configs(broadcast_engine, init_value, start_state):
    symbolic_state_dim = 3 + 2 + start_state[1] * 3
    target_configs = {}
    map_x = broadcast_engine.env.pspace.map_x
    map_y = broadcast_engine.env.pspace.map_y
    n = 10
    radius = 2
    for target in init_value.init_value.net:
        grid_states = []
        for i in range(n):
            for j in range(n):
                x, y = (i + .5) / n * map_x, (j + .5) / n * map_y
                z = random.random() * pi * 2 - pi
                grid_states.append(((x, y, z), ((x, y, z),) + start_state[1:]))
        grid_states_tensor = broadcast_engine.states2tensor([x[1] for x in grid_states]).to(broadcast_engine.device)
        grid_states_init_value = init_value(target, grid_states_tensor[:, :symbolic_state_dim]).view(-1)
        grid_argmax = grid_states_init_value.argmax(0).item()
        gx, gy, gz = grid_states[grid_argmax][0]

        grid_states = []
        for i in range(-n, n):
            for j in range(-n, n):
                for k in range(-n, n):
                    x, y = gx + (i + .5) * radius / n, gy + (j + .5) * radius / n
                    x, y = min(max(x, 0), map_x), min(max(y, 0), map_y)
                    z = gz + (k + .5) / n * pi
                    while z < -pi:
                        z += pi
                    while z >= pi:
                        z -= pi
                    grid_states.append(((x, y, z), ((x, y, z),) + start_state[1:]))
        grid_states_tensor = broadcast_engine.states2tensor([x[1] for x in grid_states])
        grid_states_init_value = init_value(target, grid_states_tensor[:, :symbolic_state_dim]).view(-1)
        grid_argmax = grid_states_init_value.argmax(0).item()
        target_config = grid_states[grid_argmax][0]

        target_configs[target] = target_config
    return target_configs
