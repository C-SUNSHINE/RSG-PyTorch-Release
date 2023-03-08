#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Mine and Craft rules are formatted as dictionaries
# Other world rules are implemented in the engine


MINING_RULES = [
    dict(
        rule_name='mine_iron_ore',
        create='iron_ore',
        action='mine',
        location='iron_ore_vein',
        recipe=[],
        holding=['pickaxe'],
    ),
    dict(
        rule_name='mine_coal',
        create='coal',
        action='mine',
        location='coal_ore_vein',
        recipe=[],
        holding=['pickaxe'],
    ),
    dict(
        rule_name='mine_cobblestone',
        create='cobblestone',
        action='mine',
        location='cobblestone_stash',
        recipe=[],
        holding=['pickaxe'],
    ),
    dict(
        rule_name='mine_wood',
        create='wood',
        action='mine',
        location='tree',
        recipe=[],
        holding=['axe'],
    ),
    dict(
        rule_name='mine_feather',
        create='feather',
        action='mine',
        location='chicken',
        recipe=[],
        holding=['sword'],
    ),
    dict(
        rule_name='mine_wool',
        create='wool',
        action='mine',
        location='sheep',
        recipe=[],
        holding=['shears'],
    ),
    dict(
        rule_name='mine_wool',
        create='wool',
        action='mine',
        location='sheep',
        recipe=[],
        holding=['sword'],
    ),
    dict(
        rule_name='mine_potato',
        create='potato',
        action='mine',
        location='potato_plant',
        recipe=[],
        holding=[],
    ),
    dict(
        rule_name='mine_beetroot',
        create='beetroot',
        action='mine',
        location='beetroot_crop',
        recipe=[],
        holding=[],
    ),
    dict(
        rule_name='mine_gold_ore',
        create='gold_ore',
        action='mine',
        location='gold_ore_vein',
        recipe=[],
        holding=['pickaxe'],
    ),
    dict(
        rule_name='mine_sugar_cane',
        create='sugar_cane',
        action='mine',
        location='sugar_cane_plant',
        recipe=[],
        holding=[],
    ),
]

CRAFTING_RULES = [
    dict(
        rule_name='craft_wood_plank',
        create='wood_plank',
        action='craft',
        location='work_station',
        recipe=['wood'],
        holding=[],
    ),
    dict(
        rule_name='craft_stick',
        create='stick',
        action='craft',
        location='work_station',
        recipe=['wood_plank'],
        holding=[],
    ),
    dict(
        rule_name='craft_arrow',
        create='arrow',
        action='craft',
        location='weapon_station',
        recipe=['stick', 'feather'],
        holding=[],
    ),
    dict(
        rule_name='craft_sword',
        create='sword',
        action='craft',
        location='weapon_station',
        recipe=['stick', 'iron_ingot'],
        holding=[],
    ),
    dict(
        rule_name='craft_shears',
        create='shears',
        action='craft',
        location='tool_station',
        recipe=['iron_ingot'],
        holding=[],
    ),
    dict(
        rule_name='craft_shears',
        create='shears',
        action='craft',
        location='tool_station',
        recipe=['gold_ingot'],
        holding=[],
    ),
    dict(
        rule_name='craft_iron_ingot',
        create='iron_ingot',
        action='craft',
        location='furnace',
        recipe=['iron_ore', 'coal'],
        holding=[],
    ),
    dict(
        rule_name='craft_gold_ingot',
        create='gold_ingot',
        action='craft',
        location='furnace',
        recipe=['gold_ore', 'coal'],
        holding=[],
    ),
    dict(
        rule_name='craft_bed',
        create='bed',
        action='craft',
        location='bed_station',
        recipe=['wood_plank', 'wool'],
        holding=[],
    ),
    dict(
        rule_name='craft_boat',
        create='boat',
        action='craft',
        location='boat_station',
        recipe=['wood_plank'],
        holding=[],
    ),
    dict(
        rule_name='craft_bowl',
        create='bowl',
        action='craft',
        location='food_station',
        recipe=['wood_plank'],
        holding=[],
    ),
    dict(
        rule_name='craft_bowl',
        create='bowl',
        action='craft',
        location='food_station',
        recipe=['iron_ingot'],
        holding=[],
    ),
    dict(
        rule_name='craft_cooked_potato',
        create='cooked_potato',
        action='craft',
        location='furnace',
        recipe=['potato', 'coal'],
        holding=[],
    ),
    dict(
        rule_name='craft_beetroot_soup',
        create='beetroot_soup',
        action='craft',
        location='food_station',
        recipe=['beetroot', 'bowl'],
        holding=[],
    ),
    dict(
        rule_name='craft_paper',
        create='paper',
        action='craft',
        location='work_station',
        recipe=['sugar_cane'],
        holding=[],
    )
]

ALL_RULES = MINING_RULES + CRAFTING_RULES

if __name__ == '__main__':
    import jacinle

    jacinle.stprint(ALL_RULES)
