#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from hacl.envs.gridworld.crafting_world.v20210515 import CraftingWorldV20210515
from hacl.envs.gridworld.crafting_world.engine.engine import CraftingWorldActions
from hacl.models.rsgs.state_machine.algorithms import get_all_paths
from hacl.models.rsgs.state_machine.builders import StateMachine


class CraftingChecker(object):
    class TerminateChecker(object):
        def __init__(self, env_args, label):
            self.env = CraftingWorldV20210515(env_args)
            holding_exp = env_args['checker_holdings'][label]
            self.machine = StateMachine.from_expression(holding_exp, StateMachine.default_primitive_constructor)
        def __call__(self, state):
            self.env.load_from_symbolic_state(state)
            f = {x: (x in self.machine.starts) for x in self.machine.nodes}
            for x in self.machine.get_topological_sequence():
                if f[x]:
                    for y, el in self.machine.adjs[x]:
                        if self.env.engine.agent.holding(el):
                            f[y] = True
            if any(f[y] for y in self.machine.ends):
                return True
            return False

    def __init__(self, env_args, plan_search=False):
        self.env_args = CraftingWorldV20210515.complete_env_args(env_args)
        self.env = CraftingWorldV20210515(env_args)
        self.action_set = (CraftingWorldActions.Up, CraftingWorldActions.Down, CraftingWorldActions.Left,
                           CraftingWorldActions.Right, CraftingWorldActions.Toggle)
        self.plan_search = plan_search

    def check_traj_transition(self, states, actions):
        self.env.load_from_symbolic_state(states[0])
        if self.env.get_symbolic_state() != states[0]:
            print('The first state is not valid.')
            return False
        for i in range(len(actions)):
            self.env.load_from_symbolic_state(states[i])
            if actions[i] not in self.action_set:
                return False
            self.env.action(actions[i])
            inferred_next_state = self.env.get_symbolic_state()
            self.env.load_from_symbolic_state(states[i + 1])
            if self.env.get_symbolic_state() != inferred_next_state:
                print(states[i], '->', states[i + 1], ' true:', inferred_next_state)
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

    def get_object_status(self, name, fail_ok=False):
        obj_list = list(self.env.engine.find_objects(name))
        if fail_ok and len(obj_list) != 1:
            return None
        assert len(obj_list) == 1
        return obj_list[0][1].get_status()

    def check_holding(self, name):
        return self.env.engine.agent.holding(name)

    def state_goal_predicate(self, state, goal):
        action, target = goal[:goal.find('_')], goal[goal.find('_') + 1:]
        assert action + '_' + target == goal
        self.env.load_from_symbolic_state(state)
        if action == 'toggle':
            assert target == 'switch'
            status = self.get_object_status('switch', fail_ok=True)
            return status == 1
        elif action == 'grab':
            return self.check_holding(target)
        elif action == 'mine':
            return self.check_holding(target)
        elif action == 'craft':
            return self.check_holding(target)
        else:
            raise ValueError('Invalid action %s' % action)

    def check_complete(self, states, path):
        nodes, edges = path
        n_steps = len(edges)
        f = [False for i in range(len(nodes))]
        f[0] = True
        for state in states:
            for i in range(n_steps):
                if f[i] and not f[i + 1]:
                    if self.state_goal_predicate(state, edges[i]):
                        f[i + 1] = True
        if f[n_steps]:
            return True, 1
        for i in range(n_steps - 1, -1, -1):
            if f[i]:
                return False, i / n_steps
        return False, 0

    def get_terminate_checker(self, label):
        return CraftingChecker.TerminateChecker(self.env_args, label)

    def __call__(self, traj, label, start_state, *args, **kwargs):
        states, actions = traj
        if len(states) != len(actions) + 1:
            print('states length and actions length mismatch')
            return False, 0
        if start_state is not None:
            self.env.load_from_symbolic_state(start_state)
            if start_state != states[0]:
                print('invalid start')
                return False, 0
        if not self.check_traj_transition(states, actions):
            print('transition fail')
            return False, 0
        if not self.plan_search:
            state_machine = self.get_state_machine_from_label(label)
            paths = get_all_paths(state_machine)
            best_progress = 0
            for path in paths:
                done, progress = self.check_complete(states, path)
                if done:
                    return True, 1
                else:
                    best_progress = max(best_progress, progress)
            return False, best_progress
        else:
            if self.get_terminate_checker(label)(states[-1]):
                return True, 1
            print('termination check fail')
            return False, 0
