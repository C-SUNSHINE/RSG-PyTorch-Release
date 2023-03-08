#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from math import sqrt
from hacl.utils.geometry_2d import Point, Pose, Polygon, Object


class Robot(object):
    """
    A Robot is a set of convex polygons.
    """

    def __init__(self, polys):
        self.polys = polys
        self.radius = None

    # Creates an Object with reference point at the specified conf.
    def configuration(self, configuration):  # [x, y, theta]
        assert len(configuration) == 3
        return Object(Point(configuration[0], configuration[1]), [poly.rotate(configuration[2]) for poly in self.polys])

    def get_radius(self):
        if self.radius is None:
            self.radius = max([poly.get_radius() for poly in self.polys])
        return self.radius

    def distance(self, configuration):  # configuration distance
        return sqrt(
            configuration[0] * configuration[0] + configuration[1] * configuration[1]
        ) + self.get_radius() * abs(configuration[2])

    def __repr__(self):
        return 'Robot: (' + str(self.polys) + ')'

    def __hash__(self):
        return str(self).__hash__()

    __str__ = __repr__


class RobotArm(object):
    """
    A robot arm with rotating joints.
    """

    def __init__(self, reference, joints):
        self.reference = reference  # a Point
        # The joint Point is location of joint at zero configuration
        # The joint Polygon is link relative to that location
        # This definition could be generalized a bit...
        self.joints = joints  # [(Point, Polygon)...]

    # Creates an instance of robot at the specified configuration
    def configuration(self, configuration):  # [theta_1, theta_2, ..., theta_n]
        assert len(configuration) == len(self.joints)
        polys = []
        origin = None
        angle = None
        for i in range(len(configuration)):
            (joint, link) = self.joints[i]
            if origin == None:
                angle = configuration[i]
                origin = joint.rotate(angle)
            else:
                origin += joint.rotate(angle)
                angle += configuration[i]
            polys.append(link.at_pose(Pose(origin, angle)))
        return Object(self.reference, polys)

    def distance(self, configuration):  # configuration distance
        assert len(configuration) == len(self.joints)
        return max([abs(configuration[i]) * self.joints[i][1].get_radius() for i in range(len(configuration))])

    def __repr__(self):
        return 'RobotArm: (' + self.reference + ', ' + str(self.joints) + ')'

    def __hash__(self):
        return str(self).__hash__()

    __str__ = __repr__
