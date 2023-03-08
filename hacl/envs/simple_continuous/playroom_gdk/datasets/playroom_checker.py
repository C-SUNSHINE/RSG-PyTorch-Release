#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from hacl.envs.simple_continuous.playroom_gdk.toyrobot_v20210423 import ToyRobotV20210423
from hacl.models.rsgs.state_machine.algorithms import get_all_paths
from hacl.models.rsgs.state_machine.builders import StateMachine
from .playroom_v1 import ACTION_TO_REGION
import copy

class PlayroomChecker(object):
    class TerminateChecker(object):
        def __init__(self, env_args, label):
            self.env = ToyRobotV20210423(env_args)

        def __call__(self, state):
            self.env.load_from_symbolic_state(state)
            return bool(self.env.env_variables['monkey_cry'])

    def __init__(self, env_args, plan_search=False):
        self.env_args = copy.deepcopy(env_args)
        self.env = ToyRobotV20210423(env_args)
        self.plan_search = plan_search

    def check_traj_transition(self, states):
        for config1, config2 in zip(states[:-1], states[1:]):
            for k in range(3):
                if abs(self.env.pspace.cspace.cspace_ranges[k].difference(config1[k], config2[k])) > self.env.pspace.cspace.cspace_max_stepdiff[k]:
                    print(config1, '->', config2, ' Failed!')
                    return False
        return True

    def get_state_machine_from_label(self, label):
        def primitive_constructor(obj_name):
            a = StateMachine()
            s = a.add_node()
            t = a.add_node()
            a.add_edge(s, t, obj_name)
            return a, s, t

        return StateMachine.from_expression(label, primitive_constructor=primitive_constructor)

    def state_predicate(self, state, predicate):
        assert predicate.startswith('eff_')
        target = ACTION_TO_REGION[predicate[4:]]
        return self.env.pspace.in_region(state, target)

    def check_complete(self, states, path):
        nodes, edges = path
        n_steps = len(edges)
        f = [False for i in range(len(nodes))]
        f[0] = True
        for state in states:
            for i in range(n_steps):
                if f[i] and not f[i + 1]:
                    if self.state_predicate(state, 'eff_' + edges[i]):
                        f[i + 1] = True
        if f[n_steps]:
            return True, 1
        for i in range(n_steps - 1, -1, -1):
            if f[i]:
                return False, i / n_steps
        return False, 0

    def get_terminate_checker(self, label):
        return PlayroomChecker.TerminateChecker(self.env_args, label)

    def __call__(self, traj, label, start_state, *args, **kwargs):
        if start_state is not None:
            self.env.load_from_symbolic_state(start_state)
        path = [state[0] for state in traj[0]]
        if not self.check_traj_transition(path):
            print('transition fail')
            return False, 0
        state_machine = self.get_state_machine_from_label(label)
        target_paths = get_all_paths(state_machine)
        best_progress = 0
        for target_path in target_paths:
            done, progress = self.check_complete(path, target_path)
            if done:
                return True, 1
            else:
                best_progress = max(best_progress, progress)
        return False, best_progress
