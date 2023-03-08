#! /usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import pickle
import random

import numpy as np
from tqdm import tqdm

from hacl.envs.gridworld.crafting_world.configs import OBJECT2IDX
from hacl.envs.gridworld.crafting_world.crafting_ai import ALL_RULES, CraftingAI
from hacl.envs.gridworld.crafting_world.v20210515 import CraftingWorldV20210515
from hacl.envs.gridworld.crafting_world.engine.objects import WorldObject
from hacl.models.rsgs.state_machine import StateMachine


class CraftingV1:

    def __init__(self, seed=2333, labels=None):
        self.rng = random.Random(seed)
        self.nprng = np.random.default_rng(seed=self.rng.randint(0, 10 ** 9))
        self.default_labels = (
            'goto_weapon_station',
            'grab_axe',
            'mine_wood',
            'craft_wood_plank',
            'grab_pickaxe>(mine_iron_ore&mine_coal)>craft_iron_ingot',
            '(grab_pickaxe>(mine_iron_ore&mine_coal)>craft_iron_ingot>craft_shears>mine_wool)&(grab_axe>mine_wood>craft_wood_plank)>craft_bed',
            'grab_pickaxe>grab_axe>((mine_iron_ore&mine_coal)>craft_iron_ingot>mine_wood>craft_wood_plank>craft_stick>craft_sword>mine_feather)&(mine_wood>craft_wood_plank>craft_stick)>craft_arrow'
        )
        self.labels = self.default_labels if labels is None else labels

    @classmethod
    def build_env_from_args(cls, env_args):
        return CraftingWorldV20210515(
            env_args
        )

    def make_initial_inventory(self, env, state_machine, prerequisite):
        start_actions = []
        for start in state_machine.starts:
            for v, el in state_machine.adjs[start]:
                start_actions.append(el.split('>')[1])
        rule_name = random.choice(list(set(start_actions)))
        rules = list(filter(lambda r: r['rule_name'] == rule_name, ALL_RULES))
        if len(rules) > 0:
            rule = random.choice(rules)
        else:
            rule = None
        items_add = set(prerequisite)
        if rule is not None:
            for item in rule['holding'] + rule['recipe']:
                items_add.add(item)
        for item in items_add:
            if not env.engine.agent.holding(item):
                env.engine.agent.push(WorldObject.from_string(item))
        while not env.engine.agent.full() and random.random() < 0.66:
            while True:
                obj = WorldObject.from_string(random.choice(list(OBJECT2IDX.keys())))
                if obj.type == 'item' and not env.engine.agent.holding(obj.name):
                    env.engine.agent.push(obj)
                    break

    def get_object_requires(self, label):
        s = label.replace('>', ' ').replace('&', ' ').replace('|', ' ').replace('(', ' ').replace(')', ' ')
        res = set()
        for a in s.split(' '):
            if a != '':
                if a.startswith('grab_'):
                    res.add(a[5:])
                elif a.startswith('toggle'):
                    res.add(a[7:])
                else:
                    for rule in filter(lambda x: x['rule_name'] == a, ALL_RULES):
                        res.add(rule['location'])
        return res

    def generate(
        self,
        n_data=None,
        env_args=None,
        map_ids=None,
        split=None,
        data_pack_name=None,
        force_regen=False,
        max_steps=None,
        prerequisites=None,
    ):
        if data_pack_name is not None and not force_regen:
            data_pack_dir = os.path.join('data', 'CraftingV1')
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

        if prerequisites is None:
            prerequisites = [tuple() for label in self.labels]

        for label, prerequisite in zip(self.labels, prerequisites):
            pbar = tqdm(range(n_data))
            pbar.set_description('Generating data for label ' + label)
            state_machine = label2state_machine[label]
            env = CraftingWorldV20210515(env_args)
            for data_id in pbar:
                while True:
                    if map_ids is not None:
                        map_candidates = map_ids[:]
                    else:
                        map_candidates = list(range(len(env.env_args['maps'])))
                    recommended = []
                    for map_id in map_candidates:
                        map_args = env.get_map_args_by_map_id(map_id)
                        if 'recommended' in map_args and label in map_args['recommended']:
                            recommended.append(map_id)
                    if len(recommended) > 0:
                        map_candidates = recommended
                    # print(label, map_candidates)
                    map_id = self.rng.choice(map_candidates)
                    env.override_map_id(map_id)
                    object_requires = self.get_object_requires(label)
                    env.override_object_requires(object_requires)
                    env.restart()

                    self.make_initial_inventory(env, state_machine, prerequisite)

                    start_state = env.get_symbolic_state()

                    target_paths = self.get_target_paths(state_machine)

                    sampled_trajs = []
                    for target_path in target_paths:
                        success, path, length = self.sample_path_from_target_path(
                            env, start_state, target_path, max_steps=max_steps,
                        )
                        if success:
                            sampled_trajs.append((path, length))
                    if len(sampled_trajs) == 0:
                        continue
                    sampled_trajs.sort(key=lambda v: v[1])
                    sampled_traj = sampled_trajs[0][0]

                    data.append(
                        {
                            'split': split,
                            'start_state': start_state,
                            'label': label,
                            'traj': sampled_traj,
                        }
                    )
                    label_count[label] += 1
                    break

        label_prior = {}
        total_prob = 0
        for label in label_count:
            total_prob += label_count[label]
        for label in self.labels:
            label_prior[label] = label_count[label] / total_prob

        data_pack = {
            'data': data,
            'labels': self.labels[:],
            'label_prior': label_prior,
            'env_args': env_args,
        }
        if data_pack_name is not None:
            data_pack_dir = os.path.join('data', 'CraftingV1')
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

    def sample_path_from_target_path(self, env, start_state, target_path, max_steps=None):
        env.load_from_symbolic_state(start_state)
        ai = CraftingAI()
        success, path = ai.complex_task(env, target_path, max_steps=max_steps, verbose=False)
        if success:
            return True, path, len(path[1])
        else:
            return False, None, None

    @classmethod
    def render_data_human(self, samples, env_args=None, env=None):
        if env is None:
            env = self.build_env_from_args(env_args=env_args)
        for sample in samples:
            env.load_from_symbolic_state(sample['start_state'])
            label = sample['label']
            for state in sample['traj'][0]:
                env.load_from_symbolic_state(state)
                env.engine.render_cli(mission=label)
                input('press enter for next step')
            while True:
                cmd = input('Next data? y/n')
                if cmd == 'y':
                    break
                elif cmd == 'n':
                    return


if __name__ == '__main__':
    generator = CraftingV1()
    data_pack = generator.generate(n_data=5, env_args='plains', split='train')
    samples = data_pack['data']
    random.shuffle(samples)
    CraftingV1.render_data_human(samples, data_pack['env_args'])

    while input('Type \'end\' to terminate.') != 'end':
        pass
