#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import hacl.envs.gridworld.crafting_world.engine.objects as O
from .engine.objects import WorldObject
from .engine.rules import ALL_RULES
from .config_instructions import *

IDX2OBJECT = {idx: obj for idx, obj in enumerate(['_invalid'] + sorted(WorldObject.STR_TO_OBJECT_CLS.keys()))}
OBJECT2IDX = {obj: idx for idx, obj in IDX2OBJECT.items()}

IDX2TYPE = {
    0: '_invalid',
    1: 'item',
    2: 'structure',
    3: 'resource',
    4: 'station',
}
TYPE2IDX = {typ: idx for idx, typ in IDX2TYPE.items()}

SYMBOL_TO_OBJECTS = {
    '#': 'wall',
    'D': 'door',
    'S': 'switch',
    'K': 'key',
    'A': 'agent',
    '*': 'water',
}

MAP_ARGS = dict(
    plain1=dict(
        width=6, height=6, capacity=6,
        generate=dict(
            structure=[
                '......',
                '......',
                '......',
                '......',
                '......',
                '......',
            ],
            spawn={
                '.': ['pickaxe', 'coal_ore_vein', 'iron_ore_vein', 'furnace']
            },
            spawn_prob=0.6,
        )
    ),
    plain2=dict(
        width=6, height=6, capacity=6,
        generate=dict(
            structure=[
                '......',
                '......',
                '......',
                '......',
                '......',
                '......',
            ],
            spawn={
                '.': ['key', 'door', 'pickaxe', 'axe', 'tree', 'potato_plant', 'beetroot_crop']
            },
            spawn_prob=0.6,
        )
    ),
    plain3=dict(
        width=6, height=6, capacity=6,
        generate=dict(
            structure=[
                '......',
                '......',
                '......',
                '......',
                '......',
                '......',
            ],
            spawn={
                '.': ['key', 'door', 'pickaxe', 'axe', 'tree', 'potato_plant', 'beetroot_crop', 'chicken', 'sheep',
                      'coal_ore_vein', 'iron_ore_vein', 'cobblestone_stash', 'sugar_cane_plant', 'gold_ore_vein',
                      'work_station', 'weapon_station', 'tool_station', 'bed_station', 'boat_station', 'furnace', 'food_station']
            },
            spawn_prob=(0.5, 1.0),
        )
    ),
    plain4=dict(
        width=6, height=6, capacity=6,
        generate=dict(
            structure=[
                '......',
                '..*...',
                '..*...',
                '....#.',
                '....#.',
                '......',
            ],
            spawn={
                '.': ['switch', 'door', 'pickaxe', 'axe', 'tree', 'potato_plant', 'beetroot_crop', 'chicken', 'sheep',
                      'coal_ore_vein', 'iron_ore_vein', 'cobblestone_stash', 'sugar_cane_plant', 'gold_ore_vein',
                      'work_station', 'weapon_station', 'tool_station', 'bed_station', 'boat_station', 'furnace', 'food_station']
            },
            spawn_prob=(0.5, 1.0),
        )
    ),
    open_door_with_key=dict(
        width=6, height=6, capacity=6,
        generate=dict(
            structure=[
                '111111',
                '111111',
                '111...',
                '####D#',
                'AA2222',
                'AA2222',
            ],
            spawn={
                '1': ['pickaxe', 'axe', 'tree', 'potato_plant', 'beetroot_crop', 'chicken', 'sheep',
                      'work_station', 'weapon_station', 'tool_station', 'boat_station', 'food_station'],
                '2': ['key'],
            },
            spawn_prob=(0.5, 1.0),
        ),
        recommended=[
            'grab_key>grab_axe'
        ]
    ),
    open_door_with_switch=dict(
        width=6, height=6, capacity=6,
        generate=dict(
            structure=[
                '111111',
                '111111',
                '111...',
                '####D#',
                'AA2222',
                'AA2222',
            ],
            spawn={
                '1': ['pickaxe', 'axe', 'tree', 'potato_plant', 'beetroot_crop', 'chicken', 'sheep', 'sugar_cane_plant',
                      'work_station', 'weapon_station', 'tool_station', 'boat_station', 'food_station'],
                '2': ['switch'],
            },
            spawn_prob=(0.5, 1.0),
        ),
        recommended=[
            'toggle_switch>mine_beetroot'
        ]
    ),
    cross_river_with_boat=dict(
        width=6, height=6, capacity=6,
        generate=dict(
            structure=[
                '111111',
                '111111',
                '111111',
                '******',
                '2A2222',
                '2222AA',
            ],
            spawn={
                '1': ['pickaxe', 'tree', 'potato_plant', 'beetroot_crop', 'chicken', 'sheep', 'sugar_cane_plant',
                      'work_station', 'weapon_station', 'tool_station', 'bed_station', 'furnace', 'food_station'],
                '2': ['axe', 'boat_station', 'tree', 'work_station', 'furnace', 'tool_station', 'weapon_station'],
            },
            spawn_prob=(0.5, 1.0),
        ),
        recommended=[
            'grab_axe>mine_wood>craft_wood_plank>craft_boat>mine_sugar_cane',
            'grab_axe>mine_wood>craft_wood_plank>craft_boat>grab_pickaxe'
        ]
    ),
    cross_river_with_boat_axe_in_room=dict(
        width=6, height=6, capacity=6,
        generate=dict(
            structure=[
                '111111',
                '111111',
                '******',
                'A22222',
                '22A2##',
                'A22AD3',
            ],
            spawn={
                '1': ['pickaxe', 'tree', 'potato_plant', 'beetroot_crop', 'chicken', 'sheep', 'sugar_cane_plant',
                      'bed_station', 'furnace'],
                '2': ['key', 'boat_station', 'tree', 'work_station', 'furnace', 'tool_station', 'weapon_station'],
                '3': ['axe']
            },
            spawn_prob=(0.5, 1.0),
        ),
        recommended=[
            'grab_key>grab_axe>mine_wood>craft_wood_plank>craft_boat>mine_potato',
        ]
    ),
    cross_river_to_access_iron=dict(
        width=6, height=6, capacity=6,
        generate=dict(
            structure=[
                '111111',
                '111111',
                '******',
                'A22222',
                '22A2##',
                'A22AD3',
            ],
            spawn={
                '1': ['pickaxe', 'switch', 'potato_plant', 'beetroot_crop', 'chicken', 'sheep', 'sugar_cane_plant',
                      'bed_station', 'furnace', 'food_station'],
                '2': ['axe', 'boat_station', 'tree', 'work_station', 'tool_station', 'weapon_station'],
                '3': ['iron_ore_vein']
            },
            spawn_prob=(0.5, 1.0),
        ),
        recommended=[
            'grab_axe>mine_wood>craft_wood_plank>craft_boat>(toggle_switch&grab_pickaxe)>mine_iron_ore'
        ]
    ),
    craft_iron_among_islands=dict(
        width=6, height=6, capacity=6,
        generate=dict(
            structure=[
                '44*222',
                '444***',
                '***555',
                '33*555',
                '333*##',
                'AA.D11',
            ],
            spawn={
                '1': ['pickaxe', 'furnace'],
                '2': ['iron_ore_vein'],
                '3': ['axe', 'boat_station', 'tree', 'work_station'],
                '4': ['key', 'switch'],
                '5': ['coal_ore_vein'],
            },
            spawn_prob=1.0,
        ),
        recommended=[
            'grab_axe>mine_wood>craft_wood_plank>craft_boat>grab_key|toggle_switch>grab_pickaxe>mine_iron_ore&mine_coal>craft_iron_ingot'
        ]
    ),
    boat_or_key=dict(
        width=6, height=6, capacity=6,
        generate=dict(
            structure=[
                '..222.',
                '555.11',
                '*****D',
                '4.3333',
                '4.3333',
                '4.AAAA',
            ],
            spawn={
                '1': ['pickaxe', 'furnace'],
                '2': ['gold_ore_vein'],
                '3': ['axe', 'boat_station', 'tree', 'work_station'],
                '4': ['key'],
                '5': ['coal_ore_vein'],
            },
            spawn_prob=[0.5, 1.0],
        ),
        recommended=[
            'grab_key|(grab_axe>mine_wood>craft_wood_plank>craft_boat)>grab_pickaxe>mine_gold_ore'
        ]
    ),
    boat_or_switch=dict(
        width=6, height=6, capacity=6,
        generate=dict(
            structure=[
                '111111',
                '111111',
                '****D*',
                '4..333',
                '4..333',
                '4.AAAA',
            ],
            spawn={
                '1': ['pickaxe', 'furnace', 'gold_ore_vein', 'coal_ore_vein'],
                '3': ['axe', 'boat_station', 'tree', 'work_station'],
                '4': ['switch'],
            },
            spawn_prob=[0.5, 1.0],
        ),
        recommended=[
            'toggle_switch|(grab_axe>mine_wood>craft_wood_plank>craft_boat)>grab_pickaxe>mine_coal'
        ]
    ),
    plain_big1=dict(
        width=7, height=7, capacity=6,
        generate=dict(
            structure=[
                '.......',
                '.......',
                '.......',
                '.......',
                '.......',
                '.......',
                '.......',
            ],
            spawn={
                '.': ['key', 'door', 'pickaxe', 'axe', 'tree', 'potato_plant', 'beetroot_crop', 'chicken', 'sheep',
                      'coal_ore_vein', 'iron_ore_vein', 'cobblestone_stash', 'sugar_cane_plant', 'gold_ore_vein',
                      'work_station', 'weapon_station', 'tool_station', 'bed_station', 'boat_station', 'furnace',
                      'food_station']
            },
            spawn_prob=(0.5, 1.0),
        )
    ),

    plain_big_shears=dict(
        width=7, height=7, capacity=6,
        generate=dict(
            structure=[
                'AAAA...',
                '..111..',
                '...222.',
                '.333...',
                '...444.',
                '.555...',
                '....666',
            ],
            spawn={
                '.': ['axe', 'tree', 'potato_plant', 'beetroot_crop', 'chicken', 'sheep',
                      'cobblestone_stash', 'sugar_cane_plant', 'work_station', 'boat_station'],
                '1': ['pickaxe'],
                '2': ['coal_ore_vein'],
                '3': ['iron_ore_vein'],
                '4': ['furnace'],
                '5': ['tool_station'],
            },
            spawn_prob=(0.9, 1.0),
        ),
        recommended=[
            'grab_pickaxe>mine_coal>mine_iron_ore>craft_iron_ingot>craft_shears'
        ]
    ),

    plain_big_feather=dict(
        width=7, height=7, capacity=6,
        generate=dict(
            structure=[
                'AAAA...',
                '..111..',
                '5..2225',
                '5333..5',
                '5..3335',
                '5444..5',
                '....444',
            ],
            spawn={
                '.': ['axe', 'tree', 'potato_plant', 'beetroot_crop', 'chicken', 'sheep',
                      'cobblestone_stash', 'sugar_cane_plant', 'work_station', 'boat_station'],
                '1': ['pickaxe'],
                '2': ['work_station'],
                '3': ['furnace'],
                '4': ['chicken'],
                '5': ['weapon_station'],
            },
            holding=['pickaxe', 'coal', 'iron_ore', 'wood'],
            spawn_prob=(0.9, 1.0),
        ),
        recommended=[
            'craft_wood_plank>craft_iron_ingot>craft_stick>craft_sword>mine_feather'
        ]
    ),
)

DEFAULT_ENV_ARGS_V1 = dict(
    test=dict(
        rules=ALL_RULES,
        object_limit=30,
        maps=['cross_river_with_boat_axe_in_room'],
        dataset='crafting'
    ),
    primitives=dict(
        rules=ALL_RULES,
        object_limit=8,
        maps=['plain1', 'plain2', 'plain3', 'plain4'],
        dataset='crafting'
    ),
    integrated=dict(
        rules=ALL_RULES,
        object_limit=8,
        maps=['plain1', 'plain2', 'plain3', 'plain4'],
        dataset='crafting'
    ),
    novels=dict(
        rules=ALL_RULES,
        object_limit=23,
        maps=[
            'plain3', 'plain4',
            'open_door_with_key',
            'open_door_with_switch',
            'cross_river_with_boat',
            'cross_river_with_boat_axe_in_room',
            'cross_river_to_access_iron',
            'craft_iron_among_islands',
            'boat_or_key',
            'boat_or_switch',
        ]
    ),
    plan_search=dict(
        rules=ALL_RULES,
        object_limit=30,
        maps=[
            # 'plain1',
            # 'plain2',
            # 'plain3',
            # 'plain4',
            # 'open_door_with_key',
            # 'open_door_with_switch',
            # 'cross_river_with_boat',
            # 'cross_river_with_boat_axe_in_room',
            # 'cross_river_to_access_iron',
            # 'craft_iron_among_islands',
            'plain_big1',
            'plain_big_shears',
            'plain_big_feather',
        ],
        search_primitives={
            'grab_axe>mine_wood': [
                'grab_axe',
                'grab_pickaxe',
                'grab_key',
                'toggle_switch',
                'mine_wood',
                'mine_coal',
                'mine_iron_ore',
                'mine_gold_ore',
                'craft_gold_ingot',
            ],
            '(grab_key|toggle_switch)>grab_pickaxe>(mine_coal|mine_iron_ore)': [
                'grab_axe',
                'grab_pickaxe',
                'grab_key',
                'toggle_switch',
                'mine_wood',
                'mine_coal',
                'mine_iron_ore',
                'mine_gold_ore',
                'craft_gold_ingot',
            ],
            'grab_pickaxe>mine_gold_ore&mine_coal>craft_gold_ingot': [
                'grab_axe',
                'grab_pickaxe',
                'grab_key',
                'toggle_switch',
                'mine_wood',
                'mine_coal',
                'mine_iron_ore',
                'mine_gold_ore',
                'craft_gold_ingot',
            ],
        },
        final_skills={
            'grab_axe>mine_wood': [
                'mine_wood',
            ],
            'grab_pickaxe>mine_gold_ore&mine_coal>craft_gold_ingot': [
                'craft_gold_ingot'
            ],
            'grab_axe>mine_wood>craft_wood_plank>craft_boat': [
                'craft_boat',
            ],
            'grab_pickaxe>mine_coal>mine_iron_ore>craft_iron_ingot>craft_shears': [
                'craft_shears'
            ],
            'mine_sugar_cane>craft_paper': [
                'craft_paper'
            ],
            'mine_beetroot&craft_bowl>craft_beetroot_soup': [
                'craft_beetroot_soup'
            ],
            'craft_wood_plank>mine_wool>craft_bed': [
                'craft_bed',
            ],
            'grab_pickaxe>mine_coal>mine_potato>craft_cooked_potato': [
                'craft_cooked_potato'
            ]
        },

        checker_holdings={
            'grab_axe>mine_wood': 'wood',
            'mine_sugar_cane>craft_paper': 'paper',
            'mine_beetroot&craft_bowl>craft_beetroot_soup': 'beetroot_soup',
            'craft_wood_plank>mine_wool>craft_bed': 'bed',
            'grab_pickaxe>mine_gold_ore&mine_coal>craft_gold_ingot': 'gold_ingot',
            'grab_axe>mine_wood>craft_wood_plank>craft_boat': 'boat',
            'grab_pickaxe>mine_coal>mine_potato>craft_cooked_potato': 'potato',
            'grab_pickaxe>mine_coal>mine_iron_ore>craft_iron_ingot>craft_shears': 'shears',
        },
        skills=PRIMITIVES,
    )
)

import random


def generate_map(map_args, object_limit=None, object_requires=None):
    width, height = map_args['width'], map_args['height']
    structure = map_args['generate']['structure']
    board = [[' ' for j in range(height)] for i in range(width)]
    for i in range(width):
        for j in range(height):
            board[i][j] = structure[height - 1 - j][i]
    spawn = map_args['generate']['spawn']
    spawn_prob = map_args['generate']['spawn_prob'] if 'spawn_prob' in map_args['generate'] else 1.0
    if isinstance(spawn_prob, (tuple, list)):
        if random.random() < .1:
            spawn_prob = random.choice(spawn_prob)
        else:
            spawn_prob = random.random() * (spawn_prob[1] - spawn_prob[0]) + spawn_prob[0]
    if object_requires is None:
        object_requires = set()
    else:
        object_requires = set(object_requires)
    if any(obj in object_requires for obj in ['door', 'key', 'switch']):
        object_requires.update(['door', 'key', 'switch'])

    blocks = {}
    init_pos_candidates = []
    symbol2positions = {}
    for i in range(width):
        for j in range(height):
            if board[i][j] == '#':
                blocks[(i, j)] = O.Wall()
            elif board[i][j] == '*':
                blocks[(i, j)] = O.Water()
            else:
                if board[i][j] not in symbol2positions:
                    symbol2positions[board[i][j]] = []
                symbol2positions[board[i][j]].append((i, j))
    for symbol in SYMBOL_TO_OBJECTS:
        if symbol in symbol2positions:
            pos = random.choice(symbol2positions[symbol])
            if SYMBOL_TO_OBJECTS[symbol] == 'agent':
                init_pos_candidates.append(pos)
            else:
                blocks[pos] = WorldObject.from_string(SYMBOL_TO_OBJECTS[symbol])
    init_pos = random.choice(init_pos_candidates) if len(init_pos_candidates) > 0 else None
    door_key_switch = None
    spawn_blocks = dict()
    for symbol in spawn:
        obj_list = []
        for obj_name in spawn[symbol]:
            if obj_name in ['door', 'key', 'switch']:
                if door_key_switch is None:
                    door_key_switch = random.random() <= spawn_prob
                if door_key_switch:
                    obj_list.append(obj_name)
            elif random.random() <= spawn_prob:
                obj_list.append(obj_name)
        pos_list = symbol2positions[symbol]
        if len(obj_list) > len(pos_list):
            raise ValueError('Too many objects on map with args %s' % str(map_args))
        random.shuffle(obj_list)
        random.shuffle(pos_list)
        for pos, obj in zip(pos_list[:len(obj_list)], obj_list):
            if obj in object_requires:
                blocks[pos] = WorldObject.from_string(obj)
            else:
                spawn_blocks[pos] = WorldObject.from_string(obj)

    spawned_blocks = list(spawn_blocks.items())
    random.shuffle(spawned_blocks)
    if object_limit is not None:
        spawned_blocks = spawned_blocks[:object_limit - len(blocks)]
    for (i, j), obj in spawned_blocks:
        blocks[(i, j)] = obj
    if object_limit is not None and len(blocks) > object_limit:
        # print('Warning! Too many objects!')
        # if 'recommended' in map_args:
        #     print('Recommend:', map_args['recommended'])
        # else:
        #     print('No recommend')
        new_object_keys = list(blocks.keys())
        new_blocks = {k: blocks[k] for k in new_object_keys[:object_limit]}
        blocks = new_blocks
    assert len(blocks) <= object_limit
    if not any(obj.name == 'door' for obj in blocks.values()):
        for pos in blocks:
            if blocks[pos].name == 'switch':
                blocks.pop(pos)
                break
    init_objs = map_args['generate']['holding'] if 'holding' in map_args['generate'] else []
    return blocks, init_pos, init_objs
