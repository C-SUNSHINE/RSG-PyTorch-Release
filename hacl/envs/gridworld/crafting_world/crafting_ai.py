#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import random

from hacl.envs.gridworld.crafting_world.configs import PRIMITIVES
from hacl.envs.gridworld.crafting_world.engine.rules import ALL_RULES
from hacl.envs.gridworld.crafting_world.engine.engine import CraftingWorldActions


class CraftingAI(object):
    def __init__(self):
        pass

    def _check_bfs_path(self, env, start_state, visited, ex, ey):
        x, y = ex, ey
        actions = []
        while visited[(x, y)] is not None:
            actions.append(visited[(x, y)][0])
            x, y = visited[(x, y)][1]
        states = [start_state]
        actions = list(reversed(actions))
        env.load_from_symbolic_state(start_state)
        for action in actions:
            move_success = env.action(action)
            states.append(env.get_symbolic_state())
            if not move_success:
                env.load_from_symbolic_state(start_state)
                return False, None
        if env.engine.pos == (ex, ey):
            return True, (states, actions)
        env.load_from_symbolic_state(start_state)
        return False, None

    def _goto(self, env, target, verbose=False):
        start_state = env.get_symbolic_state()
        x, y = start_state[:2]
        visited = dict()
        q, qh = [(x, y)], 0
        visited[(x, y)] = None
        while qh < len(q):
            (x, y) = q[qh]
            qh += 1
            directions = [
                CraftingWorldActions.Up,
                CraftingWorldActions.Left,
                CraftingWorldActions.Down,
                CraftingWorldActions.Right,
            ]
            random.shuffle(directions)
            for action in directions:
                env.load_from_symbolic_state((x, y) + start_state[2:])
                move_success = env.action(action)
                if move_success:
                    nx, ny = env.engine.pos
                    if (nx, ny) not in visited:
                        visited[(nx, ny)] = (action, (x, y))
                        q.append((nx, ny))
                    if env.engine.board[nx][ny].name == target:
                        return self._check_bfs_path(env, start_state, visited, nx, ny)
        env.load_from_symbolic_state(start_state)
        return False, None

    def _goto_location_and_create(self, env, target, create_action, verbose=False):
        start_state = env.get_symbolic_state()
        if env.engine.agent.holding(target):
            return False, None
        if create_action in ['mine', 'craft']:
            rules = list(filter(lambda r: r['rule_name'] == create_action + '_' + target, ALL_RULES))
            assert len(rules) > 0
            rule = random.choice(rules)
            success, path = self._goto(env, rule['location'], verbose=verbose)
            if verbose:
                print("Target: %s, Goto %s, success: %s" % (target, rule['location'], str(success)))
        else:
            success, path = self._goto(env, target, verbose=verbose)
            if verbose:
                print("Target: %s, Goto %s, success: %s" % (target, target, str(success)))
        if success:
            states, actions = path
            toggle_success = env.action(CraftingWorldActions.Toggle)
            states.append(env.get_symbolic_state())
            actions.append(CraftingWorldActions.Toggle)
            if toggle_success and (create_action == 'toggle' or env.engine.agent.holding(target)):
                return success, (states, actions)
        env.load_from_symbolic_state(start_state)
        return False, None

    def _grab(self, env, target, verbose=False):
        return self._goto_location_and_create(env, target, 'grab', verbose=verbose)

    def _craft(self, env, target, verbose=False):
        return self._goto_location_and_create(env, target, 'craft', verbose=verbose)

    def _mine(self, env, target, verbose=False):
        return self._goto_location_and_create(env, target, 'mine', verbose=verbose)

    def single_task(self, env, goal, verbose=False):
        if goal not in PRIMITIVES:
            raise ValueError('Goal %s is not a primitive' % goal)
        if verbose:
            print('Primitive goal:', goal)
        if goal.startswith('grab_'):
            if env.engine.agent.full():
                return False, None
            return self._grab(env, goal[5:], verbose=verbose)
        elif goal.startswith('goto_'):
            return self._goto(env, goal[5:], verbose=verbose)
        elif goal.startswith('craft_'):
            return self._craft(env, goal[6:], verbose=verbose)
        elif goal.startswith('mine_'):
            return self._mine(env, goal[5:], verbose=verbose)
        elif goal == 'toggle_switch':
            return self._goto_location_and_create(env, 'switch', 'toggle', verbose=verbose)
        else:
            raise ValueError('Invalid primitive goal %s' % goal)

    def complex_task(self, env, goals, max_steps=None, verbose=False):
        if verbose:
            print('Goal sequence:', goals)
        states, actions = [env.get_symbolic_state()], []
        for goal in goals:
            success, path = self.single_task(env, goal, verbose=verbose)
            if not success:
                return False, None
            states += path[0][1:]
            actions += path[1]
            assert env.get_symbolic_state() == states[-1]
            if max_steps is not None and len(actions) > max_steps:
                return False, None
        return True, (states, actions)


def _main_test():
    from hacl.envs.gridworld.crafting_world.v20210515 import CraftingWorldV20210515

    env = CraftingWorldV20210515('test')
    start_state = env.get_symbolic_state()
    ai = CraftingAI()
    # success, path = ai.complex_task(
    #     env, [
    #         'grab_pickaxe',
    #         'mine_coal',
    #         'mine_iron_ore',
    #         'craft_iron_ingot'
    #     ], verbose=True)

    success, path = ai.complex_task(
        env, [
            'grab_axe',
            'mine_wood',
            'craft_wood_plank',
            'craft_boat',
            'grab_pickaxe',
            'mine_coal',
            'mine_iron_ore',
            'craft_iron_ingot'
        ], verbose=True)

    assert success
    print(path)
    states, actions = path
    assert states[0] == start_state
    env.load_from_symbolic_state(start_state)
    env.engine.render_cli()
    for action, state in zip(actions, states[1:]):
        input('next?')
        success = env.action(action)
        assert success
        assert env.get_symbolic_state() == state
        env.engine.render_cli()



if __name__ == '__main__':
    _main_test()