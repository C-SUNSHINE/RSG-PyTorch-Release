PRIMITIVES = [
    'grab_pickaxe',
    'grab_axe',
    'grab_key',

    'toggle_switch',

    'craft_wood_plank',
    'craft_stick',
    'craft_shears',
    'craft_bed',
    'craft_boat',
    'craft_sword',
    'craft_arrow',
    'craft_cooked_potato',
    'craft_iron_ingot',
    'craft_gold_ingot',
    'craft_bowl',
    'craft_beetroot_soup',
    'craft_paper',

    'mine_gold_ore',
    'mine_iron_ore',
    'mine_sugar_cane',
    'mine_coal',
    'mine_wood',
    'mine_feather',
    'mine_wool',
    'mine_potato',
    'mine_beetroot',
]

FULL_INSTRUCTIONS = [
    'grab_pickaxe',
    'grab_axe',
    'grab_key',

    'toggle_switch',

    'grab_axe>mine_wood>craft_wood_plank',
    'grab_axe>mine_wood>craft_wood_plank>craft_stick',
    'grab_pickaxe>mine_iron_ore&mine_coal>craft_iron_ingot>craft_shears',
    'grab_axe>mine_wood>craft_wood_plank>grab_pickaxe>mine_iron_ore&mine_coal>craft_iron_ingot>craft_shears>mine_wool>craft_bed',
    'grab_axe>mine_wood>craft_wood_plank>craft_boat',
    '(grab_axe>mine_wood>craft_wood_plank>craft_stick)>(grab_pickaxe>mine_iron_ore&mine_coal>craft_iron_ingot)>craft_sword',
    '(grab_axe>mine_wood>craft_wood_plank>craft_stick)>(grab_pickaxe>mine_iron_ore&mine_coal>craft_iron_ingot)>craft_sword>mine_feather>mine_wood>craft_wood_plank>craft_stick>craft_arrow',
    'mine_potato&(grab_pickaxe>mine_coal)>craft_cooked_potato',
    'grab_pickaxe>mine_iron_ore&mine_coal>craft_iron_ingot',
    'grab_pickaxe>mine_gold_ore&mine_coal>craft_gold_ingot',
    'grab_axe>mine_wood>craft_wood_plank>craft_bowl',
    'mine_beetroot&(grab_axe>mine_wood>craft_wood_plank>craft_bowl)>craft_beetroot_soup',
    'mine_sugar_cane>craft_paper',

    'grab_pickaxe>mine_gold_ore',
    'grab_pickaxe>mine_iron_ore',
    'mine_sugar_cane',
    'grab_pickaxe>mine_coal',
    'grab_axe>mine_wood',
    '(grab_axe>mine_wood>craft_wood_plank>craft_stick)>(grab_pickaxe>mine_iron_ore&mine_coal>craft_iron_ingot)>craft_sword>mine_feather',
    'grab_pickaxe>mine_iron_ore&mine_coal>craft_iron_ingot>craft_shears>mine_wool',
    'mine_potato',
    'mine_beetroot',
]

COMPLEX_INSTRUCTIONS = [
    ('grab_pickaxe', tuple()),
    ('grab_axe', tuple()),
    ('grab_key', tuple()),

    ('toggle_switch', tuple()),

    ('mine_wood>craft_wood_plank', ('axe',)),
    ('craft_wood_plank>craft_stick', ('wood',)),
    ('craft_iron_ingot|craft_gold_ingot>craft_shears', ('iron_ore', 'gold_ore', 'coal')),
    ('mine_wool&craft_wood_plank>craft_bed', ('shears', 'wood')),
    ('craft_wood_plank>craft_boat', ('wood',)),
    ('craft_iron_ingot&craft_stick>craft_sword', ('iron_ore', 'coal', 'wood_plank')),
    ('mine_feather&craft_stick>craft_arrow', ('sword', 'wood_plank')),
    ('mine_potato&mine_coal>craft_cooked_potato', ('pickaxe',)),
    ('mine_iron_ore&mine_coal>craft_iron_ingot', ('pickaxe',)),
    ('mine_gold_ore&mine_coal>craft_gold_ingot', ('pickaxe',)),
    ('craft_wood_plank|craft_iron_ingot>craft_bowl', ('wood', 'iron_ore', 'coal')),
    ('craft_bowl&mine_beetroot>craft_beetroot_soup', ('wood_plank',)),
    ('mine_sugar_cane>craft_paper', tuple()),

    ('grab_pickaxe>mine_gold_ore', tuple()),
    ('grab_pickaxe>mine_iron_ore', tuple()),
    ('grab_pickaxe|grab_axe>mine_sugar_cane', tuple()),
    ('grab_pickaxe>mine_coal', tuple()),
    ('grab_axe>mine_wood', tuple()),
    ('craft_sword>mine_feather', ('stick', 'iron_ingot')),
    ('craft_shears|craft_sword>mine_wool', ('iron_ingot', 'gold_ingot', 'stick')),
    ('grab_axe|mine_coal>mine_potato', ('pickaxe',)),
    ('grab_axe|grab_pickaxe>mine_beetroot', tuple()),
]

PRIMITIVES_AND_INTEGRATED = PRIMITIVES[:]
for x in COMPLEX_INSTRUCTIONS:
    if x[0] not in PRIMITIVES_AND_INTEGRATED:
        PRIMITIVES_AND_INTEGRATED.append(x[0])

NOVEL_INSTRUCTIONS = [
    # Novel craft
    'mine_sugar_cane>craft_paper',
    'mine_potato&(grab_pickaxe>mine_coal)>craft_cooked_potato',
    'mine_beetroot&(grab_axe>mine_wood>craft_wood_plank>craft_bowl)>craft_beetroot_soup',
    'grab_axe>mine_wood>craft_wood_plank>grab_pickaxe>mine_iron_ore&mine_coal>craft_iron_ingot>craft_shears>mine_wool>craft_bed',
    '(grab_axe>mine_wood>craft_wood_plank>craft_stick)>(grab_pickaxe>mine_iron_ore&mine_coal>craft_iron_ingot)>craft_sword>mine_feather>mine_wood>craft_wood_plank>craft_stick>craft_arrow',

    # With obstacles

    'grab_key>grab_axe',  # 2
    'toggle_switch>mine_beetroot',  # 2
    'grab_axe>mine_wood>craft_wood_plank>craft_boat>mine_sugar_cane',  # 5
    'grab_axe>mine_wood>craft_wood_plank>craft_boat>grab_pickaxe',  # 5
    'grab_key>grab_axe>mine_wood>craft_wood_plank>craft_boat>mine_potato',  # 6
    'grab_key|(grab_axe>mine_wood>craft_wood_plank>craft_boat)>grab_pickaxe>mine_gold_ore',  # 7
    'grab_axe>mine_wood>craft_wood_plank>craft_boat>grab_key|toggle_switch>grab_pickaxe>mine_iron_ore&mine_coal>craft_iron_ingot',
    # 9
]

PLAN_SEARCH_INSTRUCTIONS = [
        'grab_axe>mine_wood',
        'mine_sugar_cane>craft_paper',
        'mine_beetroot&craft_bowl>craft_beetroot_soup',
        'craft_wood_plank>mine_wool>craft_bed',
        'grab_pickaxe>mine_gold_ore&mine_coal>craft_gold_ingot',
        'grab_axe>mine_wood>craft_wood_plank>craft_boat',
        'grab_pickaxe>mine_coal>mine_potato>craft_cooked_potato',
        'grab_pickaxe>mine_coal>mine_iron_ore>craft_iron_ingot>craft_shears',
]
