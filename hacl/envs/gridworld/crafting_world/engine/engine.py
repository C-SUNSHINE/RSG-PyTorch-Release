#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import random

from jacinle.utils.enum import JacEnum

import hacl.envs.gridworld.crafting_world.engine.objects as O
from hacl.envs.gridworld.crafting_world.engine.objects import WorldObject, Agent, Empty
from hacl.envs.gridworld.crafting_world.engine.rules import ALL_RULES


class CraftingWorldActions(JacEnum):
    Idle = -1
    Up = 0
    Down = 1
    Left = 2
    Right = 3
    Toggle = 4

    def __int__(self):
        return self.value


DIRECTION_VEC = {
    CraftingWorldActions.Up: (0, 1),
    CraftingWorldActions.Down: (0, -1),
    CraftingWorldActions.Left: (-1, 0),
    CraftingWorldActions.Right: (1, 0),
}


class CraftingWorldEngine(object):
    def __init__(self, width, height, capacity=5, blocks=None, init_pos=None, rules=None):
        """
        Args:
            width (int): the width of the board.
            height (int): the height of the board.
            capacity (int): the capacity of the agent's inventory.
            blocks (optional, Map[(int, int), WorldObject]).
            init_pos (optional, Tuple[int, int]): initial pos of the agent. If None, use a randomly sampled location.
            rules (optional, List[Rule]): all rules.
        """
        self.width = width
        self.height = height
        self.agent = Agent(capacity)

        self.rules = ALL_RULES if rules is None else rules

        self.board = [[Empty() for j in range(self.height)] for i in range(self.width)]
        self.init_board_from_blocks(blocks)

        if init_pos is None:
            empty_positions = []
            for i in range(self.width):
                for j in range(self.height):
                    if isinstance(self.board[i][j], Empty):
                        empty_positions.append((i, j))
            self.pos = random.choice(empty_positions)
        else:
            self.pos = init_pos

    def init_board_from_blocks(self, blocks):
        self.board = [[Empty() for j in range(self.height)] for i in range(self.width)]
        collision_set = []
        if blocks is not None:
            for (i, j), obj in blocks.items():
                self.board[i][j] = obj
                collision_set.append((i, j))
        if len(set(collision_set)) < len(collision_set):
            raise ValueError('Collision Blocks.')

    @property
    def x(self):
        return self.pos[0]

    @property
    def y(self):
        return self.pos[1]

    def find_objects(self, name):
        for i in range(self.width):
            for j in range(self.height):
                if self.board[i][j] is not None and self.board[i][j].name == name:
                    yield (i, j), self.board[i][j]

    def _grab(self):
        obj = self.board[self.x][self.y]
        if obj is not None and obj.can_pickup() and not self.agent.full():
            self.agent.push(obj)
            self.board[self.x][self.y] = Empty()
            return True
        return False

    def _toggle(self):
        stand = self.board[self.x][self.y]
        if isinstance(stand, O.Switch):
            stand.toggle()

            door: O.Door
            for (x, y), door in self.find_objects('door'):
                door.set(stand.is_on())
            return True

        return False

    def _mine(self):
        stand = self.board[self.x][self.y]
        for rule in self.rules:
            if rule['rule_name'].startswith('mine_'):
                if stand.name == rule['location'] and all(self.agent.holding(tool) for tool in rule['holding']) and not self.agent.full():
                    self.agent.push(WorldObject.from_string(rule['create']))
                    return True
        return False

    def _craft(self):
        stand = self.board[self.x][self.y]
        for rule in self.rules:
            if rule['rule_name'].startswith('craft_'):
                if stand.name == rule['location'] and all(self.agent.holding(ingredient) for ingredient in rule['recipe']):
                    for ingredient in rule['recipe']:
                        self.agent.pop(ingredient)
                    self.agent.push(WorldObject.from_string(rule['create']))
                    return True
        return False

    def _is_walkable(self, x, y):
        return self.board[x][y] is None or self.board[x][y].can_walk(self.agent)

    def action(self, a: CraftingWorldActions):
        if a == CraftingWorldActions.Idle:
            return True
        elif a in [CraftingWorldActions.Up, CraftingWorldActions.Down, CraftingWorldActions.Left, CraftingWorldActions.Right]:
            x = self.x + DIRECTION_VEC[a][0]
            y = self.y + DIRECTION_VEC[a][1]
            if 0 <= x < self.width and 0 <= y < self.height and self._is_walkable(x, y):
                self.pos = (x, y)
                return True
            return False
        elif a == CraftingWorldActions.Toggle:
            success = False
            obj = self.board[self.x][self.y]
            if obj.type == 'item':
                success = self._grab()
            elif obj.type == 'structure':
                success = self._toggle()
            elif obj.type == 'resource':
                success = self._mine()
            elif obj.type == 'station':
                success = self._craft()
            else:
                raise ValueError('Invalid type %s.' % obj.type)
            return success
        else:
            raise ValueError('Invalid Action %d' % a)

    def render_cli(self, mission=None):
        cell_width = 10
        cell_height = 5
        line_width = 8
        n_lines = 2
        cell_desc = [['' for j in range(self.height)] for i in range(self.width)]
        for i in range(self.width):
            for j in range(self.height):
                cell_str = self.board[i][j].name
                if cell_str == 'empty':
                    cell_str = ''
                if self.board[i][j].name in ['door', 'switch']:
                    cell_str = cell_str + str(self.board[i][j].get_status())
                cell_str = cell_str[:line_width * n_lines]
                while len(cell_str) < line_width * n_lines:
                    cell_str = cell_str + ' '
                cell_desc[i][j] = cell_str

        for j in range(self.height, -1, -1):
            if j == self.height:
                print('#' * ((cell_width + 1) * self.width + 1))
                continue
            for k in range(cell_height):
                for i in range(-1, self.width):
                    if i == -1:
                        print('#', end='')
                        continue
                    elif 1 <= k <= n_lines:
                        left_space = (cell_width - line_width) // 2
                        right_space = cell_width - line_width - left_space
                        print(' ' * left_space + cell_desc[i][j][(k - 1) * line_width: k * line_width] + ' ' * right_space, end='')
                    elif k == cell_height * 2 // 3 and self.x == i and self.y == j:
                        print(' ' * ((cell_width - 1) // 2) + 'A' + ' ' * (cell_width // 2), end='')
                    else:
                        print(' ' * cell_width, end='')
                    print('#', end='')
                print('')
            print('#' * ((cell_width + 1) * self.width + 1))
        print('Inventory: ', ', '.join(obj.name for obj in self.agent.inventory))
        print('Mission: ', str(mission))


def _main_test():
    import keyboard
    import time

    engine = CraftingWorldEngine(5, 5, blocks={
        (3, 3): O.Tree(),
        (4, 4): O.Axe(),
        (1, 3): O.WorkStation(),
        (4, 1): O.IronOreVein(),
        (4, 2): O.Furnace(),
        (2, 4): O.CoalOreVein(),
        (1, 1): O.Pickaxe(),
        (3, 4): O.BedStation(),
        (2, 2): O.Sheep(),
        (2, 3): O.ToolStation()
    })
    while True:
        engine.render_cli()

        action = CraftingWorldActions.Idle
        while True:
            # cmd = input('input cmd')
            if keyboard.is_pressed('w'):
                action = CraftingWorldActions.Up
            elif keyboard.is_pressed('a'):
                action = CraftingWorldActions.Left
            elif keyboard.is_pressed('d'):
                action = CraftingWorldActions.Right
            elif keyboard.is_pressed('s'):
                action = CraftingWorldActions.Down
            elif keyboard.is_pressed('f'):
                action = CraftingWorldActions.Toggle
            elif keyboard.is_pressed('k'):
                exit()
            else:
                continue
            break

        success = engine.action(action)
        print(f"Action {action}", "succeed" if success else "failed.")
        time.sleep(0.1)


if __name__ == '__main__':
    _main_test()
