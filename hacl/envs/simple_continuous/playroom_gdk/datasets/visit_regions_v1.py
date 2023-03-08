#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import random
from tqdm import tqdm
import numpy as np

from hacl.models.rsgs.state_machine import StateMachine
from hacl.envs.simple_continuous.playroom_gdk.toyrobot_v20210423 import ToyRobotV20210423
from hacl.envs.simple_continuous.playroom_gdk.visualize import visualize_problem_and_solution
from hacl.algorithms.rrt.rrt import birrt, optimize_path


class VisitRegionsV1:

    def __init__(self, seed=2333, labels=None, render_human=False):
        self.rng = random.Random(seed)
        self.nprng = np.random.default_rng(seed=seed)
        self.default_labels = ('B>A', 'C>B', 'A|B>C', 'A&B>C',)
        self.labels = self.default_labels if labels is None else labels
        self.render_human = render_human

    @classmethod
    def build_env_from_args(cls, env_args):
        env = ToyRobotV20210423(env_args=env_args)
        return env

    @classmethod
    def visualize_from_env_args(cls, env_args):
        env = cls.build_env_from_args(env_args)
        visualize_problem_and_solution(env.pspace)

    def sample_start_config(self, pspace):
        while True:
            start_state = pspace.sample()
            if pspace.collide(start_state):
                continue
            if any(pspace.in_region(start_state).values()):
                continue
            return start_state

    def _solve_pspace(self, pspace, start_state=None, goal_state=None):
        if start_state is None:
            start_state = pspace.start_state
        if goal_state is None:
            goal_state = pspace.goal_state
        if pspace.collide(start_state) or pspace.collide(goal_state):
            return None, None
        path, rrt = birrt(pspace, start_state, goal_state, nr_iterations=2000)
        spath = optimize_path(pspace, path)
        return rrt, spath

    def sample_path_from_target_path(self, pspace, start_config, target_path, nprng=None):
        current_state = start_config
        path_all = [current_state]
        for target_action in target_path:
            target_region = target_action
            for it in range(10):
                next_point = pspace.regions[target_region].sample(nprng=nprng)
                next_state = (next_point.x, next_point.y, pspace.sample()[2])
                rrt, path = self._solve_pspace(pspace, current_state, next_state)
                if path is not None:
                    break
            if path is None:
                return False, None, None
            path_all.extend(path[1:])
            current_state = next_state
        return True, path_all, pspace.distance_path(path_all)

    def generate(
        self,
        n_data=None,
        env_args=None,
        selected_labels=None,
        rationality=1.00,
        split=None,
        data_pack_name=None,
        force_regen=False,
    ):
        if data_pack_name is not None and not force_regen:
            data_pack_dir = os.path.join('data', 'ToyRobotVisitRegionsV1')
            os.makedirs(data_pack_dir, exist_ok=True)
            data_pack_path = os.path.join(data_pack_dir, data_pack_name + '.pkl')
            if os.path.exists(data_pack_path):
                try:
                    data_pack = pickle.load(open(data_pack_path, 'rb'))
                    print("Successfully loaded from %s." % data_pack_path)
                    return data_pack
                finally:
                    pass

        data = []

        label2state_machine = {label: self.get_state_machine(label) for label in self.labels}

        label_count = {label: 0 for label in self.labels}

        for label in self.labels:
            if selected_labels is not None and label not in selected_labels:
                continue
            pbar = tqdm(range(n_data))
            pbar.set_description('Generating graphs for label ' + label)
            state_machine = label2state_machine[label]
            for data_id in pbar:
                while True:
                    env = self.build_env_from_args(env_args)

                    start_config = self.sample_start_config(env.pspace)

                    init_symbolic_state = env.get_symbolic_state()

                    target_paths = self.get_target_paths(state_machine)

                    sampled_paths = []

                    for target_path in target_paths:
                        success, path, length = self.sample_path_from_target_path(env.pspace, start_config, target_path, nprng=self.nprng)
                        if success:
                            sampled_paths.append((path, length))
                    if len(sampled_paths) == 0:
                        continue
                    sampled_paths.sort(key=lambda v: v[1])
                    sampled_path = sampled_paths[0][0]

                    if self.render_human:
                        visualize_problem_and_solution(env.pspace, path=sampled_path, window=label)

                    traj_actions = env.get_step_actions(sampled_path)
                    assert None not in traj_actions
                    sample_traj = [(pose,) + init_symbolic_state[1:] for pose in sampled_path]

                    data.append(dict(
                        split=split,
                        start_config=start_config,
                        label=label,
                        traj=(tuple(sample_traj), traj_actions),
                    ))
                    label_count[label] += 1
                    break
        label_prior = {}
        total_prob = 0
        for label in label_count:
            total_prob += label_count[label]
        for label in self.labels:
            label_prior[label] = label_count[label] / total_prob

        data_pack = dict(
            data=data,
            labels=tuple(self.labels),
            label_prior=label_prior,
            env_args=env_args
        )
        if data_pack_name is not None:
            data_pack_dir = os.path.join('data', 'ToyRobotVisitRegionsV1')
            os.makedirs(data_pack_dir, exist_ok=True)
            data_pack_path = os.path.join(data_pack_dir, data_pack_name + '.pkl')
            pickle.dump(data_pack, open(data_pack_path, 'wb'))
        return data_pack

    def get_state_machine(self, label):
        def primitive_constructor(obj_name):
            a = StateMachine()
            s = a.add_node()
            t = a.add_node()
            a.add_edge(s, t, str(s) + '>' + obj_name)
            return a, s, t

        res = StateMachine.from_expression(label, primitive_constructor)
        return res

    def get_target_paths(self, state_machine):
        q = []
        for node in state_machine.nodes:
            if node in state_machine.starts:
                q.append(((node,), tuple()))
        paths = []
        while len(q) > 0:
            (nodes, edges) = q[0]
            q = q[1:]
            if nodes[-1] in state_machine.ends:
                paths.append((nodes, edges))
            else:
                for v, l in state_machine.adjs[nodes[-1]]:
                    q.append((nodes + (v,), edges + (l,)))
        target_paths = []
        for (nodes, edges) in paths:
            target_path = []
            for i in range(len(edges)):
                action = edges[i].split('>')[1]
                target_path.append(action)
            target_paths.append(target_path)
        return target_paths


if __name__ == '__main__':
    VisitRegionsV1.visualize_from_env_args('regions_empty')
    G = VisitRegionsV1(seed=2333, labels=('A>B', 'A&B'), render_human=True)
    G.generate(n_data=1, env_args='regions_maze1')
