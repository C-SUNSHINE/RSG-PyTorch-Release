#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# This file was originally written by Caelan Garrett, modified a bit by Tomas Lozano-Perez.

from random import random, randint, uniform
from math import pi, sin, cos
from hacl.algorithms.space import BoxConfigurationSpace, BoxConstrainedConfigurationSpace, ProblemSpace
from hacl.utils.geometry_2d import AABB, Point, Polygon, Object, convex_hull

__all__ = ['ToyRobotConfigurationSpace', 'ToyRobotConstrainedConfigurationSpace', 'ToyRobotProblemSpace']


class ToyRobotConfigurationSpace(BoxConfigurationSpace):
    def __init__(self, robot, cspace_ranges, cspace_max_stepdiff):
        super().__init__(cspace_ranges, cspace_max_stepdiff)
        self.robot = robot

    def difference(self, one: tuple, two: tuple):
        return tuple(two[i] - one[i] for i in range(len(one)))

    def distance(self, one: tuple, two: tuple):
        a = [self.cspace_ranges[i].difference(one[i], two[i]) for i in range(len(self.cspace_ranges))]
        return self.robot.distance(tuple(a))


class ToyRobotConstrainedConfigurationSpace(BoxConstrainedConfigurationSpace):
    @property
    def robot(self):
        return self.proxy.robot


class ToyRobotProblemSpace(ProblemSpace):
    def __init__(
        self, cspace, obstacles, map_x, map_y, start_state, goal_state, regions=None, regions_desc=None, **kwargs
    ):
        super().__init__(cspace)
        self.obstacles = obstacles
        self.regions = {}
        if regions is not None:
            self.regions.update(regions)
        self.regions_desc = {}
        if regions_desc is not None:
            self.regions_desc.update(regions_desc)
        self.map_x = map_x
        self.map_y = map_y
        self.map_region = AABB(Point(0, 0), Point(map_x, map_y))
        self.start_state = start_state
        self.goal_state = goal_state

    @property
    def robot(self):
        return self.cspace.robot

    def validate_config(self, configuration):
        return self.cspace.validate_config(configuration) and not self.collide(configuration)

    def try_extend_path(self, start, end):  # returns subset of path that is safe
        success, path = self.cspace.gen_path(start, end)
        if path is None:
            return False, start, path
        safe_path = []
        for configuration in path:
            if self.collide(configuration):
                return False, safe_path[-1], safe_path
            safe_path.append(configuration)
        return success, safe_path[-1], safe_path

    def collide(self, configuration):  # Check collision for configuration
        config_robot = self.robot.configuration(configuration)
        return any([config_robot.collides(obstacle) for obstacle in self.obstacles]) or not self.map_region.contains(
            config_robot
        )

    def in_region(self, configuration, name=None):
        if name is None:
            return {region_name: self.in_region(configuration, name=region_name) for region_name in self.regions}
        else:
            config_robot = self.robot.configuration(configuration)
            return config_robot.collides(self.regions[name])

    # Generate a regular polygon that does not collide with other
    # obstacles or the robot at start and goal.
    def generate_random_regular_poly(self, num_verts, radius, angle=uniform(-pi, pi)):
        """
        Generates a regular polygon that does not collide with other
            obstacles or the robot at start and goal. This polygon is added
            to self.obstacles. To make it random, keep the default angle
            argument.

        Args:
            num_verts: int. the number of vertices of the polygon >= 3
            radius: float. the distance from the center of the polygon
                to any vertex > 0
            angle: float. the angle in radians between the origin and
                the first vertex. the default is a random value between
                -pi and pi.

        """
        (min_verts, max_verts) = num_verts
        (min_radius, max_radius) = radius
        assert not (min_verts < 3 or min_verts > max_verts or min_radius <= 0 or min_radius > max_radius)
        reference = Point(random() * self.map_x, random() * self.map_y)
        distance = uniform(min_radius, max_radius)
        sides = randint(min_verts, max_verts)
        obj = Object(
            reference,
            [
                Polygon(
                    [
                        Point(distance * cos(angle + 2 * n * pi / sides), distance * sin(angle + 2 * n * pi / sides))
                        for n in range(sides)
                    ]
                )
            ],
        )
        if (
            any([obj.collides(current) for current in self.obstacles])
            or obj.collides(self.robot.configuration(self.start_state))
            or obj.collides(self.robot.configuration(self.goal_state))
            or not self.map_region.contains(obj)
        ):
            self.generate_random_regular_poly(num_verts, radius, angle=angle)
        else:
            self.obstacles.append(obj)

    # Generate a polygon that does not collide with other
    # obstacles or the robot at start and goal.
    def generate_random_poly(self, num_verts, radius):
        """Generates a random polygon that does not collide with other
             obstacles or the robot at start and goal. This polygon is added
             to self.obstacles.

        Args:
            num_verts: int. the number of vertices of the polygon >= 3
            radius: float. a reference distance between the origin and some
                vertex of the polygon > 0
        """
        (min_verts, max_verts) = num_verts
        (min_radius, max_radius) = radius
        assert not (min_verts < 3 or min_verts > max_verts or min_radius <= 0 or min_radius > max_radius)

        reference = Point(random() * self.map_x, random() * self.map_y)
        verts = randint(min_verts, max_verts)
        points = [Point(0, 0)]
        for i in range(verts):
            angle = 2 * pi * random()
            points.append(((max_radius - min_radius) * random() + min_radius) * Point(cos(angle), sin(angle)))
        obj = Object(reference, [Polygon(convex_hull(points))])
        if (
            any([obj.collides(current) for current in self.obstacles])
            or obj.collides(self.robot.configuration(self.start_state))
            or obj.collides(self.robot.configuration(self.goal_state))
            or not self.map_region.contains(obj)
        ):
            self.generate_random_poly(num_verts, radius)
        else:
            self.obstacles.append(obj)
