#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import random

from hacl.envs.gridworld.crafting_world.configs import DEFAULT_ENV_ARGS_V1, IDX2OBJECT, IDX2TYPE, MAP_ARGS, OBJECT2IDX, TYPE2IDX, generate_map
from hacl.envs.gridworld.crafting_world.engine.engine import CraftingWorldEngine
from hacl.envs.gridworld.crafting_world.engine.objects import WorldObject, Agent


class CraftingWorldV20210515(object):
    def __init__(self, env_args):
        env_args = self.complete_env_args(env_args)
        self.env_args = env_args
        self.engine = None
        self.overriding_map_id = None
        self.overriding_object_requires = None
        self.restart()

    @classmethod
    def complete_env_args(cls, env_args):
        if isinstance(env_args, str):
            env_args = copy.deepcopy(DEFAULT_ENV_ARGS_V1[env_args])
        return env_args

    def override_map_id(self, map_id):
        self.overriding_map_id = map_id

    def override_object_requires(self, object_requires):
        self.overriding_object_requires = object_requires

    def get_map_args_by_map_id(self, map_id):
        map_args = MAP_ARGS[self.env_args['maps'][map_id]]
        return map_args

    def restart(self):
        if self.overriding_map_id is not None:
            map_id = self.overriding_map_id
            self.overriding_map_id = None
        else:
            map_id = random.randint(0, len(self.env_args['maps']) - 1)
        map_args = self.get_map_args_by_map_id(map_id)
        object_limit = self.env_args['object_limit'] if 'object_limit' in self.env_args else None
        blocks, init_pos, init_objs = generate_map(map_args, object_limit=object_limit, object_requires=self.overriding_object_requires)
        self.overriding_object_requires = None
        width, height, capacity = map_args['width'], map_args['height'], map_args['capacity']
        self.engine = CraftingWorldEngine(width, height, capacity, blocks=blocks, init_pos=init_pos, rules=self.env_args['rules'])
        for obj in init_objs:
            self.engine.agent.push(WorldObject.from_string(obj))

    def action(self, engine_action):
        success = self.engine.action(engine_action)
        return success

    def get_symbolic_state(self):
        state = [
            self.engine.pos[0],
            self.engine.pos[1],
            self.engine.width,
            self.engine.height,
            self.engine.agent.capacity,
            len(self.engine.agent.inventory),
        ]

        inventory = []
        for item in self.engine.agent.inventory:
            inventory.append(OBJECT2IDX[item.name])

        block_states = []
        for i in range(self.engine.width):
            for j in range(self.engine.height):
                if self.engine.board[i][j].name != 'empty':
                    block_states.append((
                        OBJECT2IDX[self.engine.board[i][j].name],
                        TYPE2IDX[self.engine.board[i][j].type],
                        i,
                        j,
                        self.engine.board[i][j].get_status()
                    ))

        state.append(tuple(sorted(inventory)))
        state.append(tuple(sorted(block_states)))

        return tuple(state)

    def load_from_symbolic_state(self, state):
        x, y, width, height, capacity, used, inventory, block_states = state
        assert used == len(inventory) <= capacity
        self.engine.width, self.engine.height = width, height
        self.engine.agent = Agent(capacity=capacity)
        self.engine.pos = (x, y)

        for item_id in inventory:
            if self.engine.agent.full():
                print(state)
            self.engine.agent.push(WorldObject.from_string(IDX2OBJECT[item_id]))

        blocks = {}
        for (obj_id, type_id, i, j, status) in block_states:
            assert 0 <= i < width and 0 <= j < height
            obj = WorldObject.from_string(IDX2OBJECT[obj_id])
            assert obj.type == IDX2TYPE[type_id]
            obj.set_status(status)
            assert (i, j) not in blocks
            blocks[(i, j)] = obj
        self.engine.init_board_from_blocks(blocks)


def _main_test():
    import keyboard, time
    from hacl.envs.gridworld.crafting_world.engine.engine import CraftingWorldActions

    env = CraftingWorldV20210515('test')
    while True:
        env.engine.render_cli()
        symbolic_state = env.get_symbolic_state()
        env.load_from_symbolic_state(symbolic_state)
        print(symbolic_state)

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
        env.current_state
        success = env.action(action)
        print("Succeed" if success else "Fail")
        time.sleep(0.1)



if __name__ == '__main__':
    _main_test()
