#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import re


class WorldObject(object):
    STR_TO_OBJECT_CLS = {}
    CLS_INDEX = {}
    DEFAULT_NAME = '_invalid'
    TYPENAME = '_invalid'

    def __init__(self, name=None):
        if name is None:
            name = type(self).DEFAULT_NAME

        self._name = name

        self.unique = False
        self._pickable = False
        self._walkable = False

    @property
    def name(self):
        return self._name

    @property
    def type(self):
        return type(self).TYPENAME

    def can_pickup(self):
        return self._pickable

    def can_walk(self, agent=None):
        return self._walkable

    def get_status(self):
        return 0

    def set_status(self, status):
        pass

    def get_type_index(self) -> int:
        return WorldObject.CLS_INDEX[type(self)]

    @classmethod
    def from_string(cls, s, name=None):
        if s == '_invalid':
            return WorldObject()
        return cls.STR_TO_OBJECT_CLS[s](name=name)

    def __init_subclass__(cls, name=None, register=True, **kwargs):
        if name is None:
            name = re.sub( '(?<!^)(?=[A-Z])', '_', cls.__name__).lower()
        cls.DEFAULT_NAME = name
        cls.TYPENAME = name
        if register:
            WorldObject.STR_TO_OBJECT_CLS[name] = cls
            WorldObject.CLS_INDEX[cls] = len(WorldObject.CLS_INDEX)

    def encode(self):
        return (self.get_type_index(), 0, self.get_status())


class Agent(object):
    def __init__(self, capacity=5):
        self.name = 'agent'
        self.inventory = []
        self.capacity = capacity

    @property
    def type(self):
        return 'agent'

    def holding(self, name):
        return any(obj.name == name for obj in self.inventory)

    def pop(self, name):
        for i in range(len(self.inventory)):
            if self.inventory[i].name == name:
                self.inventory.pop(i)
                return

    def empty(self):
        return len(self.inventory) == 0

    def full(self):
        return len(self.inventory) == self.capacity

    def push(self, obj):
        assert not self.full()
        self.inventory.append(obj)


class WorldObjectItem(WorldObject, register=False):
    def __init__(self, name=None):
        super().__init__(name)
        self._pickable = True
        self._walkable = True


class WorldObjectStructure(WorldObject, register=False):
    def __init__(self, name=None):
        super().__init__(name)
        self._pickable = False
        self._walkable = False


class WorldObjectResource(WorldObject, register=False):
    def __init__(self, name=None):
        super().__init__(name)
        self._pickable = False
        self._walkable = True


class WorldObjectStation(WorldObject, register=False):
    def __init__(self, name=None):
        super().__init__(name)
        self._pickable = False
        self._walkable = True


class Empty(WorldObjectStructure):
    def __init__(self, name=None):
        super().__init__(name)
        self._walkable = True


class Wall(WorldObjectStructure):
    def __init__(self, name=None):
        super().__init__(name)
        self._walkable = False


class Door(WorldObjectStructure):
    def __init__(self, is_open=False, name=None):
        super().__init__(name)
        self._open = is_open

    def can_walk(self, agent=None):
        if agent:
            return self._open or agent.holding('key')
        return self._open

    def is_open(self):
        return self._open

    def toggle(self):
        self._open = not self._open

    def set(self, status):
        self._open = status

    def get_status(self):
        return 1 if self._open else 0

    def set_status(self, status):
        self._open = (status == 1)


class Switch(WorldObjectStructure):
    def __init__(self, on=False, name=None):
        super().__init__(name)
        self._walkable = True
        self._on = on

    def is_on(self):
        return self._on

    def toggle(self):
        self._on = not self._on

    def get_status(self):
        return 1 if self._on else 0

    def set_status(self, status):
        self._on = (status == 1)


class Water(WorldObjectStructure):
    def __init__(self, name=None):
        super().__init__(name)

    def can_walk(self, agent=None):
        if agent is not None:
            return agent.holding('boat')
        return False


class Key(WorldObjectItem): ...
class WorkStation(WorldObjectStation): ...
class Pickaxe(WorldObjectItem): ...
class IronOreVein(WorldObjectResource): ...
class IronOre(WorldObjectItem): ...
class IronIngot(WorldObjectItem): ...
class CoalOreVein(WorldObjectResource): ...
class Coal(WorldObjectItem): ...
class GoldOreVein(WorldObjectResource): ...
class GoldOre(WorldObjectItem): ...
class GoldIngot(WorldObjectItem): ...
class CobblestoneStash(WorldObjectResource): ...
class Cobblestone(WorldObjectItem): ...
class Axe(WorldObjectItem): ...
class Tree(WorldObjectResource): ...
class Wood(WorldObjectItem): ...
class WoodPlank(WorldObjectItem): ...
class Stick(WorldObjectItem): ...
class WeaponStation(WorldObjectStation): ...
class Sword(WorldObjectItem): ...
class Chicken(WorldObjectResource): ...
class Feather(WorldObjectItem): ...
class Arrow(WorldObjectItem): ...
class ToolStation(WorldObjectStation): ...
class Shears(WorldObjectItem): ...
class Sheep(WorldObjectResource): ...
class Wool(WorldObjectItem): ...
class Bed(WorldObjectItem): ...
class BedStation(WorldObjectStation): ...
class BoatStation(WorldObjectStation): ...
class Boat(WorldObjectItem): ...
class SugarCanePlant(WorldObjectResource): ...
class SugarCane(WorldObjectItem): ...
class Paper(WorldObjectItem): ...
class Furnace(WorldObjectStation): ...
class FoodStation(WeaponStation): ...
class Bowl(WorldObjectItem): ...
class PotatoPlant(WorldObjectResource): ...
class Potato(WorldObjectItem): ...
class CookedPotato(WorldObjectItem): ...
class BeetrootCrop(WorldObjectResource): ...
class Beetroot(WorldObjectItem): ...
class BeetrootSoup(WorldObjectItem): ...

class Hypothetical(WorldObjectItem): ...
class Trash(WorldObjectItem): ...
