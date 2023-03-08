#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Tuple, NamedTuple
from hacl.algorithms.poc.heuristic_search import run_heuristic_search

__all__ = ['find_path']


class _PathFindingState(NamedTuple):
    pose: Tuple[int, int]


NAVIGATION_ACTIONS = ['move-up', 'move-right', 'move-down', 'move-left']
DIR_TO_VEC = {
    'up': (0, -1),
    'right': (1, 0),
    'down': (0, 1),
    'left': (-1, 0)
}


def gen_get_navigation_successors(env):
    def get_sucessor(state: _PathFindingState):
        for action in NAVIGATION_ACTIONS:
            dx, dy = DIR_TO_VEC[action[5:]]
            xx, yy = state.pose[0] + dx, state.pose[1] + dy
            cell = env.grid.get(xx, yy)
            if cell is None or cell.can_walk():
                yield action, _PathFindingState((xx, yy)), 1
    return get_sucessor


def find_path(env, target_pose: Tuple[int, int]):
    target_pose = tuple(target_pose)

    def check_goal(state: _PathFindingState):
        return state.pose == target_pose

    def get_priority(state: _PathFindingState, g: int):
        return abs(state.pose[0] - target_pose[0]) + abs(state.pose[1] - target_pose[1]) + g

    state = _PathFindingState(tuple(env.agent_pos))
    try:
        return run_heuristic_search(
            state,
            check_goal,
            get_priority,
            gen_get_navigation_successors(env),
        )[1]
    except RuntimeError:
        return None

