#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
from math import pi

from hacl.envs.simple_continuous.playroom_gdk.configs import DEFAULT_ENV_ARGS_V1
from hacl.envs.simple_continuous.playroom_gdk.robot import Robot
from hacl.envs.simple_continuous.playroom_gdk.space import ToyRobotConfigurationSpace, ToyRobotProblemSpace
from hacl.utils.geometry_2d import Object, Point, Polygon
from hacl.utils.range import Range

INDEX_TO_REGION = {
    0: '_invalid',
    1: 'A',
    2: 'B',
    3: 'C',
    4: 'D',
    5: 'ball',
    6: 'bell',
    7: 'light_switch',
    8: 'monkey',
    9: 'music_button_off',
    10: 'music_button_on',
}
REGION_TO_INDEX = {v: k for k, v in INDEX_TO_REGION.items()}

INDEX_TO_VARIABLE = {
    0: '_invalid',
    1: 'light',
    2: 'ring',
    3: 'music',
    4: 'monkey_cry',
}
VARIABLE_TO_INDEX = {v: k for k, v in INDEX_TO_VARIABLE.items()}

VALUE_TO_INDEX = {
    0: False,
    1: True,
}
INDEX_TO_VALUE = {v: k for k, v in VALUE_TO_INDEX.items()}


class ToyRobotV20210423(object):
    __default_env_args__ = dict(
        robot_size=2,
        max_step_diff=0.25,
        max_x=20,
        max_y=20,
        obstacles=[],
        regions=[],
        regions_desc=dict(),
        robot_pose=(0, 0, 0),
        env_variables=dict(),
    )

    def __init__(self, env_args=None, seed=2333):
        env_args = copy.deepcopy(env_args)
        env_args = self.complete_env_args(env_args)
        self.pspace = self.make_pspace_from_env_args(env_args)
        self.pose = copy.deepcopy(env_args['robot_pose'])
        self.env_variables = copy.deepcopy(env_args['env_variables'])

    @classmethod
    def complete_env_args(cls, env_args, toplevel=True):
        if isinstance(env_args, str):
            env_args = copy.deepcopy(DEFAULT_ENV_ARGS_V1[env_args])
        else:
            env_args = copy.deepcopy(env_args)
        if 'parent' in env_args:
            parent_env_args = cls.complete_env_args(env_args['parent'], toplevel=False)
            for k, v in parent_env_args.items():
                if k not in env_args:
                    env_args[k] = copy.deepcopy(v)
        if toplevel:
            for k in cls.__default_env_args__:
                if k not in env_args:
                    env_args[k] = copy.deepcopy(cls.__default_env_args__[k])
        return env_args

    @classmethod
    def _make_cspace(cls, max_x=20, max_y=20, robot_size=2, max_step_diff=0.25, **kwargs):
        robot = Robot(
            [
                Polygon(
                    [
                        Point(-1 * robot_size, -0.5 * robot_size),
                        Point(1 * robot_size, -0.5 * robot_size),
                        Point(1 * robot_size, 0.5 * robot_size),
                        Point(-1 * robot_size, 0.5 * robot_size),
                    ]
                ),
                Polygon(
                    [
                        Point(-0.5 * robot_size, 0.5 * robot_size),
                        Point(0.5 * robot_size, 0.5 * robot_size),
                        Point(0 * robot_size, 1 * robot_size),
                    ]
                ),
            ]
        )
        cspace = ToyRobotConfigurationSpace(
            robot,
            [Range(0, max_x), Range(0, max_y), Range(-pi, pi, wrap_around=True)],
            3 * [max_step_diff],  # maximum diff at a single step
        )
        return cspace

    @classmethod
    def _make_pspace(cls, cspace, max_x, max_y, obstacles, regions, regions_desc, **kwargs):
        pspace = ToyRobotProblemSpace(
            cspace,
            obstacles,
            max_x,
            max_y,
            start_state=None,
            goal_state=None,
            regions=regions,
            regions_desc=regions_desc,
        )
        return pspace

    def make_pspace_from_env_args(self, env_args):
        cspace = self._make_cspace(**env_args)
        return self._make_pspace(cspace, **env_args)

    def get_symbolic_state(self):
        state = []
        state.append(tuple(self.pose))

        region_states = []
        for name, region in self.pspace.regions.items():
            region_states.append((REGION_TO_INDEX[name], region.reference.x, region.reference.y))

        env_values = []
        for name, value in self.env_variables.items():
            env_values.append((VARIABLE_TO_INDEX[name], VALUE_TO_INDEX[value]))
        state.append(len(region_states))
        state.append(len(env_values))

        state.extend(sorted(region_states))
        state.extend(sorted(env_values))
        return tuple(state)

    def load_from_symbolic_state(self, state):
        self.pose = copy.copy(state[0])
        n_regions = state[1]
        n_env_variables = state[2]
        for (region_id, region_x, region_y) in state[3: 3 + n_regions]:
            name = INDEX_TO_REGION[region_id]
            assert name in self.pspace.regions
            new_region = Object(Point(region_x, region_y), self.pspace.regions[name].local_polys)
            self.pspace.regions[name] = new_region
        for (variable_id, value_id) in state[3 + n_regions: 3 + n_regions + n_env_variables]:
            name = INDEX_TO_VARIABLE[variable_id]
            assert name in self.env_variables
            value = INDEX_TO_VALUE[value_id]
            self.env_variables[name] = value

    def _update_env_status(self):
        for name in self.pspace.regions:
            if self.pspace.in_region(self.pose, name=name):
                if name == 'music_button_on':
                    self.env_variables['music'] = True
                elif name == 'music_button_off':
                    self.env_variables['music'] = False
                elif name == 'bell':
                    self.env_variables['ring'] = True
                elif name == 'ball':
                    self.env_variables['ring'] = True
                    if not self.env_variables['light'] and self.env_variables['music']:
                        self.env_variables['monkey_cry'] = True
                elif name == 'light_switch':
                    self.env_variables['light'] = True
                elif name == 'monkey':
                    pass

    def get_step_action(self, config1, config2):
        assert len(config1) == len(config2) == 3
        action = [self.pspace.cspace.cspace_ranges[i].difference(config1[i], config2[i]) for i in range(3)]
        if any(abs(action[i]) > self.pspace.cspace.cspace_max_stepdiff[i] for i in range(3)):
            return None
        return tuple(action)

    def get_step_actions(self, configs):
        actions = tuple(self.get_step_action(config1, config2) for config1, config2 in zip(configs[:-1], configs[1:]))
        return actions

    def action(self, action):
        assert len(action) == 3
        action = [
            max(
                min(float(action[i]), self.pspace.cspace.cspace_max_stepdiff[i]),
                -self.pspace.cspace.cspace_max_stepdiff[i],
            )
            for i in range(3)
        ]
        virtual_new_pose = [self.pose[i] + action[i] for i in range(3)]
        new_pose = [self.pspace.cspace.cspace_ranges[i].in_range(virtual_new_pose[i]) for i in range(3)]
        if None in new_pose:
            return tuple((new_pose[i] or virtual_new_pose[i]) for i in range(3)), True
        if self.pspace.collide(new_pose):
            return tuple(new_pose), True
        self.pose = new_pose
        self._update_env_status()
        return tuple(new_pose), False
