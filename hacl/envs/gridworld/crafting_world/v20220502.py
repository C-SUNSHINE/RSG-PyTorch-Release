#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import os.path as osp
import torch
import numpy as np
from typing import Optional, Union, Tuple, Iterable
from tabulate import tabulate
from jacinle.utils.enum import JacEnum

from hacl.pdsketch.interface.v2.state import State, ValueDict
from hacl.pdsketch.interface.v2.domain import Domain, OperatorApplier
from hacl.pdsketch.interface.v2.parser import load_domain_file
from hacl.pdsketch.interface.v2.rl import RLEnvAction

from .engine.objects import WorldObject, WorldObjectItem
from .engine.rules import MINING_RULES, CRAFTING_RULES

__all__ = [
    'set_domain_mode', 'get_domain',
    'SUPPORTED_ACTION_MODES', 'SUPPORTED_STRUCTURE_MODES', 'SUPPORTED_TASKS',
    'CraftingWorldGrid', 'CraftingWorldEnvV20220502', 'CraftingWorldStateV20220502', 'make'
]

g_domain_action_mode = 'envaction'
g_domain_structure_mode = 'abskin'
g_domain: Optional[Domain] = None

SUPPORTED_ACTION_MODES = ['envaction', 'absaction']
SUPPORTED_STRUCTURE_MODES = ['abskin', 'abskin2']
SUPPORTED_TASKS = ['mining', 'crafting']


def set_domain_mode(action_mode, structure_mode):
    global g_domain
    assert g_domain is None, 'Domain has been loaded.'
    assert action_mode in SUPPORTED_ACTION_MODES, f'Unsupported action mode: {action_mode}.'
    assert structure_mode in SUPPORTED_STRUCTURE_MODES, f'Unsupported structure mode: {structure_mode}.'

    global g_domain_action_mode
    global g_domain_structure_mode
    g_domain_action_mode = action_mode
    g_domain_structure_mode = structure_mode


def get_domain(force_reload=False):
    global g_domain
    if g_domain is None or force_reload:
        g_domain = load_domain_file(osp.join(
            osp.dirname(__file__),
            'pds_files',
            f'cw-v20220502-{g_domain_action_mode}-{g_domain_structure_mode}.pdsketch'
        ))
    return g_domain


class CraftingWorldGrid(object):
    def __init__(self, map_h, map_w):
        self.map_h = map_h
        self.map_w = map_w
        self.grid = [[None for j in range(map_w)] for i in range(map_h)]
        self.used_objects = list()

    def get(self, x, y):
        return self.grid[y][x]

    def set(self, x, y, v):
        self.grid[y][x] = v

    def remove(self, x, y):
        grid = self.get(x, y)
        assert grid is not None
        self.set(x, y, None)
        self.used_objects.append(grid)

    def horizontal_wall(self, x1, x2, y):
        for x in range(x1, x2):
            self.set(x, y, WorldObject.from_string('wall', f'wall:{x},{y}'))

    def vertical_wall(self, x, y1, y2):
        for y in range(y1, y2):
            self.set(x, y, WorldObject.from_string('wall', f'wall:{x},{y}'))

    def render_text(self, agent_pos):
        rows = list()
        for y in range(self.map_h):
            row = list()
            for x in range(self.map_w):
                obj = self.get(x, y)
                prefix = (x, y) == agent_pos and '@' or ''
                if obj is None:
                    row.append(prefix)
                else:
                    row.append(prefix + obj.type)
            rows.append(row)
        return tabulate(rows)

    def iter_objects(self, sorted=True) -> Iterable[Tuple[int, int, WorldObject]]:
        def gen():
            for y in range(self.map_h):
                for x in range(self.map_w):
                    obj = self.get(x, y)
                    if obj is not None:
                        yield x, y, obj
            for item in self.used_objects:
                yield -1, -1, item
        output = list(gen())

        if sorted:
            output.sort(key=lambda x: id(x[2]))

        return output


class CraftingWorldEnvV20220502(object):
    class Actions(JacEnum):
        MOVE_UP = 'move-up'
        MOVE_RIGHT = 'move-right'
        MOVE_DOWN = 'move-down'
        MOVE_LEFT = 'move-left'
        MINE = 'mine'
        MINE_TOOL = 'mine-tool'

    DIR_TO_VEC = {
        'up': (0, -1),
        'right': (1, 0),
        'down': (0, 1),
        'left': (-1, 0)
    }

    MINING_RULES = MINING_RULES
    CRAFTING_RULES = CRAFTING_RULES

    ALL_MINING_LOCATIONS = {rule['location'] for rule in MINING_RULES}
    ALL_MINING_TOOLS = {holding for rule in MINING_RULES for holding in rule['holding']}
    ALL_MINING_OUTCOMES = {rule['create'] for rule in MINING_RULES}

    def __init__(self, task, map_h=7, map_w=7, inventory_size=5):
        assert task in SUPPORTED_TASKS, f'Unsupported task: {task}.'

        self.task = task
        self.map_h = map_h
        self.map_w = map_w
        self.inventory_size = inventory_size

        self.grid: Optional[CraftingWorldGrid] = None
        self.agent_pos = None
        self.inventory = None
        self.goal_type = None
        self.mission = None

    def random_location(self):
        x, y = np.random.randint(self.map_w), np.random.randint(self.map_h)
        for i in range(100):
            if self.grid.get(x, y) is not None:
                x, y = np.random.randint(self.map_w), np.random.randint(self.map_h)
        if i == 100:
            raise RuntimeError('Cannot find an empty location.')
        return x, y

    def _gen_grid_mining(self, holding_objects=None):
        """
        This function should set the grid and the agent position. The inventory has been set to empty.
        """

        if holding_objects is None:
            holding_objects = list()

        self.grid.horizontal_wall(0, self.map_w, 0)
        self.grid.horizontal_wall(0, self.map_w, self.map_h - 1)
        self.grid.vertical_wall(0, 0, self.map_h)
        self.grid.vertical_wall(self.map_w - 1, 0, self.map_h)

        for loc_type in type(self).ALL_MINING_LOCATIONS:
            pose = self.random_location()
            self.grid.set(*pose, WorldObject.from_string(loc_type, f'{loc_type}:0'))

        for tool_type in type(self).ALL_MINING_TOOLS:
            pose = self.random_location()
            self.grid.set(*pose, WorldObject.from_string(tool_type, f'{tool_type}:0'))

        self.agent_pos = (3, 3)
        self.goal_type = np.random.choice(list(type(self).ALL_MINING_OUTCOMES))

        for i, object_type in enumerate(holding_objects):
            self.inventory[i] = WorldObject.from_string(object_type, f'inventory:{i}')

    def reset(self, holding_objects=None):
        self.grid = CraftingWorldGrid(self.map_h, self.map_w)
        self.inventory = [None for _ in range(self.inventory_size)]

        if self.task == 'mining':
            self._gen_grid_mining(holding_objects)
        elif self.task == 'crafting':
            self._gen_grid_crafting(holding_objects)
        else:
            raise ValueError(f'Unsupported task: {self.task}.')

        self.mission = get_domain().parse(
            f'(exists (?o - item) (and (is-inventory-object ?o) (is-{self.goal_type} ?o)))'
        )
        return self.compute_obs()

    def step(self, action: Union[RLEnvAction, OperatorApplier]):
        assert self.grid is not None, 'The grid is not initialized.'

        def get_inv_id(string):
            assert string.startswith('inventory:')
            return int(string[10:])

        if isinstance(action, OperatorApplier):
            action = RLEnvAction(action.name, *action.arguments[1:])

        if action.name == 'move-up':
            self.step_move_generic('up')
        elif action.name == 'move-right':
            self.step_move_generic('right')
        elif action.name == 'move-down':
            self.step_move_generic('down')
        elif action.name == 'move-left':
            self.step_move_generic('left')
        elif action.name == 'pickup':
            self.step_pickup(get_inv_id(action.args[0]))
        elif action.name == 'mine':
            self.step_mine(get_inv_id(action.args[0]))
        elif action.name == 'mine-tool':
            self.step_mine(get_inv_id(action.args[0]), get_inv_id(action.args[1]))
        else:
            raise RuntimeError(f'Unknown action: {action}')

        obs = self.compute_obs()
        done = self.compute_done()
        return obs, -1, done, {}

    def compute_obs(self):
        return {'state': self.render_state(), 'mission': self.mission}

    def compute_done(self):
        if self.task in ('mining', 'crafting'):
            for x in self.inventory:
                if x is not None and x.type == self.goal_type:
                    return True
        else:
            raise ValueError(f'Unknown task: {self.task}.')
        return False

    def step_move_generic(self, action):
        dx, dy = type(self).DIR_TO_VEC[action]
        x, y = self.agent_pos[0] + dx, self.agent_pos[1] + dy
        grid = self.grid.get(x, y)
        if grid is None or grid.can_walk():
            self.agent_pos = (x, y)

    def step_pickup(self, target_id):
        if self.inventory[target_id] is not None:
            return

        grid = self.grid.get(*self.agent_pos)
        if grid is not None and grid.can_pickup() and self.inventory[target_id] is None:
            self.inventory[target_id] = WorldObject.from_string(grid.type, f'inventory:{target_id}')
            self.grid.remove(*self.agent_pos)

    def step_mine(self, target_id, tool_id=None):
        if self.inventory[target_id] is not None:
            return

        grid = self.grid.get(*self.agent_pos)
        if grid is None:
            return

        tool_type = None
        if tool_id is not None:
            tool = self.inventory[tool_id]
            if tool is not None:
                tool_type = tool.type

        for mine_rule in type(self).MINING_RULES:
            if grid.type == mine_rule['location']:
                holding = mine_rule['holding']
                if len(holding) == 0 and tool_type is None or len(holding) == 1 and holding[0] == tool_type:
                    self.inventory[target_id] = WorldObject.from_string(mine_rule['create'], f'inventory:{target_id}')

    def render_text(self):
        fmt = self.grid.render_text(self.agent_pos)
        fmt += '\n'
        fmt += 'Inventory: ' + ' '.join(f'{self.inventory[i].type}' if self.inventory[i] is not None else 'EMPTY' for i in range(self.inventory_size)) + '\n'
        return fmt

    def render_state(self):
        return CraftingWorldStateV20220502.from_env(self)

    def render(self):
        print(self.render_text())


class CraftingWorldStateV20220502(State):
    @classmethod
    def from_env(cls, env: CraftingWorldEnvV20220502):
        if g_domain_structure_mode == 'abskin':
            return cls.from_env_abskin(env)
        elif g_domain_structure_mode == 'abskin2':
                return cls.from_env_abskin2(env)
        else:
            raise ValueError(f'Unknown domain structure mode: {g_domain_structure_mode}')

    @classmethod
    def from_env_abskin(cls, env: CraftingWorldEnvV20220502):
        object_names = ['r']
        object_types = ['robot']
        object_poses = list()
        object_images = list()

        empty_obj = WorldObject.from_string('empty', 'empty:0')

        for x, y, obj in env.grid.iter_objects():
            object_names.append(obj.name)
            object_types.append('item')
            object_images.append(obj.encode())
            object_poses.append((x, y))
        for i, obj in enumerate(env.inventory):
            object_poses.append((-1, -1))
            object_types.append('item')
            if obj is not None:
                object_names.append(obj.name)
                object_images.append(obj.encode())
            else:
                object_names.append(f'inventory:{i}')
                object_images.append(empty_obj.encode())

        domain = get_domain()
        state = cls([domain.types[t] for t in object_types], ValueDict(), object_names)

        ctx = state.define_context(domain)
        ctx.define_predicates([
            ctx.is_inventory_object(f'inventory:{i}') for i in range(env.inventory_size)
        ])
        ctx.define_feature('robot-pose', torch.tensor([env.agent_pos], dtype=torch.float32))
        ctx.define_feature('item-pose', torch.tensor(object_poses, dtype=torch.float32))
        ctx.define_feature('item-image', torch.tensor(object_images, dtype=torch.float32))
        return state

    @classmethod
    def from_env_abskin2(cls, env: CraftingWorldEnvV20220502):
        object_names = ['r']
        object_types = ['robot']
        object_poses = list()
        object_images = list()
        location_poses = list()
        location_images = list()

        empty_obj = WorldObject.from_string('empty', 'empty:0')

        for x, y, obj in env.grid.iter_objects():
            object_names.append(obj.name)
            if isinstance(obj, WorldObjectItem):
                object_types.append('item')
                object_images.append(obj.encode())
                object_poses.append((x, y))
            else:
                object_types.append('location')
                location_images.append(obj.encode())
                location_poses.append((x, y))
        for i, obj in enumerate(env.inventory):
            object_poses.append((-1, -1))
            object_types.append('item')
            if obj is not None:
                object_names.append(obj.name)
                object_images.append(obj.encode())
            else:
                object_names.append(f'inventory:{i}')
                object_images.append(empty_obj.encode())

        domain = get_domain()
        state = cls([domain.types[t] for t in object_types], ValueDict(), object_names)

        ctx = state.define_context(domain)
        ctx.define_predicates([
            ctx.is_inventory_object(f'inventory:{i}') for i in range(env.inventory_size)
        ] + [
            ctx.inventory_used(f'inventory:{i}') for i in range(env.inventory_size) if env.inventory[i] is not None
        ])
        ctx.define_feature('robot-pose', torch.tensor([env.agent_pos], dtype=torch.float32))
        ctx.define_feature('item-pose', torch.tensor(object_poses, dtype=torch.float32))
        ctx.define_feature('item-image', torch.tensor(object_images, dtype=torch.float32))
        ctx.define_feature('location-pose', torch.tensor(location_poses, dtype=torch.float32))
        ctx.define_feature('location-image', torch.tensor(location_images, dtype=torch.float32))
        return state


def make(*args, **kwargs):
    return CraftingWorldEnvV20220502(*args, **kwargs)


def visualize_planner(env: CraftingWorldEnvV20220502, planner):
    torch.set_grad_enabled(False)
    while True:
        init_obs = env.reset()
        state, mission = init_obs['state'], init_obs['mission']
        assert planner is not None
        plan = planner(state, mission)

        cmd = visualize_plan(env, plan)
        if cmd == 'q':
            break


def visualize_plan(env: CraftingWorldEnvV20220502, plan):
    env.render()
    print('Plan: ' + ', '.join([str(x) for x in plan]))
    print('Press <Enter> to visualize.')
    _ = input('> ').strip()

    for action in plan:
        if action.name.startswith('move'):
            env.step(RLEnvAction(action.name))
        elif action.name == 'pickup':
            env.step(RLEnvAction(action.name, action.arguments[1]))
        elif action.name == 'mine':
            env.step(RLEnvAction(action.name, action.arguments[1]))
        elif action.name == 'mine-tool':
            env.step(RLEnvAction(action.name, action.arguments[1], action.arguments[2]))
        else:
            raise NotImplementedError(action)
        env.render()
        time.sleep(0.5)

    print('Visualization finished.')
    print('Press <Enter> to continue. Type q to quit.')
    cmd = input('> ').strip()
    return cmd