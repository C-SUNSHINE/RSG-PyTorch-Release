#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import torch

from hacl.algorithms.planning.astar_planning_engine import AstarPlanningEngine
from hacl.algorithms.planning.graph_planning_engine import GraphPlanningEngine
from hacl.p.rsgs.unified_broadcast_engine import UnifiedBroadcastEngine


class InitValuePlanner(object):
    def __init__(self, env_name, env_args, init_value, device=torch.device('cpu')):
        self.env_name = env_name
        self.env_args = env_args
        self.broadcast_engine = UnifiedBroadcastEngine(env_name, env_args).to(device)
        self.continuous = env_name == 'toyrobot'

        self.init_value = init_value.to(device)

    def plan_single(self, constraint, label, state_machine, *args, **kwargs):
        if not self.continuous:
            search_engine = AstarPlanningEngine(
                self.env_name,
                self.env_args,
                self.broadcast_engine,
                self.init_value,
                label,
                state_machine,
                **kwargs
            )
            states, actions, search_count = search_engine.search(constraint, **kwargs)
            return {'traj': (states, actions), 'search_count': search_count}
        else:
            search_engine = GraphPlanningEngine(
                self.env_name,
                self.env_args,
                self.broadcast_engine,
                self.init_value,
                label,
                state_machine,
                **kwargs
            )
            traj = search_engine.search(constraint, *args, **kwargs)
            return {'traj': traj}

    def plan(self, constraints, labels, state_machines, use_point_net=False, *args, **kwargs):
        results = []
        if self.continuous:
            graphs, rrts, target_configs_list = GraphPlanningEngine.build_rrt_with_point_targets(
                self.broadcast_engine,
                self.env_args,
                self.init_value,
                constraints,
                use_point_net=use_point_net,
            )
            args = [graphs, rrts] + list(args)
        for start_state, label, state_machine, *requires in zip(constraints, labels, state_machines, *args):
            results.append(self.plan_single(start_state, label, state_machine, *requires, **kwargs))
        if self.continuous:
            for result, graph, rrt, target_configs in zip(results, graphs, rrts, target_configs_list):
                result.update(graph=graph, rrt=rrt, target_configs=target_configs)
        return results
