#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from hacl.utils.geometry_2d import Point, Polygon, Object
import math

__all__ = ['DEFAULT_ENV_ARGS_V1']

OBJECT_SHAPES = dict(
    music_button_on=[
        Polygon(
            [
                Point(-0.5, -0.5),
                Point(0.5, -0.5),
                Point(0.5, 0.5),
                Point(-0.5, 0.5),
            ]
        ),
    ],
    music_button_off=[
        Polygon(
            [
                Point(-0.5, -0.5),
                Point(0.5, -0.5),
                Point(0.5, 0.5),
                Point(-0.5, 0.5),
            ]
        ),
    ],
    bell=[
        Polygon(
            [
                Point(-0.45, -0.5),
                Point(0.45, -0.5),
                Point(0.3, 0.4),
                Point(-0.3, 0.4),
            ]
        ),
        Polygon(
            [
                Point(-0.1, 0.4),
                Point(0.1, 0.4),
                Point(0.1, 0.5),
                Point(-0.1, 0.5),
            ]
        ),
    ],
    ball=[
        Polygon([Point(0.5, 0).rotate(i / 12 * math.pi * 2) for i in range(12)]),
    ],
    light_switch=[
        Polygon(
            [
                Point(-0.4, -0.5),
                Point(0.4, -0.5),
                Point(0.4, 0.5),
                Point(-0.4, 0.5),
            ]
        ),
        Polygon(
            [
                Point(-0.1, -0.3),
                Point(0.1, -0.3),
                Point(0.1, 0.3),
                Point(-0.1, 0.3),
            ]
        ),
    ],
    monkey=[
        Polygon([Point(0.45, 0).rotate(i / 12 * math.pi * 2) for i in range(12)]),
        Polygon([Point(0.45, 0).rotate(math.pi / 4) + Point(0.15, 0).rotate(i / 8 * math.pi * 2) for i in range(8)]),
        Polygon(
            [Point(0.45, 0).rotate(3 * math.pi / 4) + Point(0.15, 0).rotate(i / 8 * math.pi * 2) for i in range(8)]
        ),
    ],
)

OBJECT_DESCS = dict(
    music_button_on=dict(color='green'),
    music_button_off=dict(color='red'),
    bell=dict(color='yellow'),
    ball=dict(color='#a00000'),
    light_switch=dict(color='grey'),
    monkey=dict(color='brown'),
)

DEFAULT_ENV_ARGS_V1 = dict(
    regions_empty=dict(
        dataset='regions',
        robot_size=2.0,
        max_step_diff=1.0,
        max_x=20,
        max_y=20,
        obstacles=[],
        regions=dict(
            A=Object(Point(15, 10), [Polygon([Point(-1, -1), Point(1, -1), Point(1, 1), Point(-1, 1)])]),
            B=Object(Point(10, 15), [Polygon([Point(-1, -1), Point(1, -1), Point(1, 1), Point(-1, 1)])]),
            C=Object(Point(5, 10), [Polygon([Point(-1, -1), Point(1, -1), Point(1, 1), Point(-1, 1)])]),
            D=Object(Point(10, 5), [Polygon([Point(-1, -1), Point(1, -1), Point(1, 1), Point(-1, 1)])]),
        ),
        regions_desc=dict(
            A=dict(color='orange'),
            B=dict(color='yellow'),
            C=dict(color='purple'),
            D=dict(color='green'),
        ),
    ),
    regions_empty_corner=dict(
        dataset='regions',
        robot_size=2.0,
        max_step_diff=1.0,
        max_x=20,
        max_y=20,
        obstacles=[],
        regions=dict(
            A=Object(Point(1.5, 1.5), [Polygon([Point(-1, -1), Point(1, -1), Point(1, 1), Point(-1, 1)])]),
            B=Object(Point(18.5, 1.5), [Polygon([Point(-1, -1), Point(1, -1), Point(1, 1), Point(-1, 1)])]),
            C=Object(Point(18.5, 18.5), [Polygon([Point(-1, -1), Point(1, -1), Point(1, 1), Point(-1, 1)])]),
            D=Object(Point(1.5, 18.5), [Polygon([Point(-1, -1), Point(1, -1), Point(1, 1), Point(-1, 1)])]),
        ),
        regions_desc=dict(
            A=dict(color='orange'),
            B=dict(color='yellow'),
            C=dict(color='purple'),
            D=dict(color='green'),
        ),
    ),
    regions_Xshape=dict(
        dataset='regions',
        parent='regions_empty',
        obstacles=[
            Object(Point(7, 10), [Polygon([Point(-7, 7), Point(-7, -7), Point(0, 0)])]),
            Object(Point(10, 7), [Polygon([Point(-7, -7), Point(7, -7), Point(0, 0)])]),
            Object(Point(13, 10), [Polygon([Point(7, -7), Point(7, 7), Point(0, 0)])]),
            Object(Point(10, 13), [Polygon([Point(7, 7), Point(-7, 7), Point(0, 0)])]),
        ],
        regions=dict(
            A=Object(Point(2, 2), [Polygon([Point(-1, -1), Point(1, -1), Point(1, 1), Point(-1, 1)])]),
            B=Object(Point(18, 2), [Polygon([Point(-1, -1), Point(1, -1), Point(1, 1), Point(-1, 1)])]),
            C=Object(Point(18, 18), [Polygon([Point(-1, -1), Point(1, -1), Point(1, 1), Point(-1, 1)])]),
            D=Object(Point(2, 18), [Polygon([Point(-1, -1), Point(1, -1), Point(1, 1), Point(-1, 1)])]),
        ),
    ),
    regions_maze1=dict(
        dataset='regions',
        parent='regions_empty',
        obstacles=[
            Object(Point(3, 15), [Polygon([Point(-2, -0.5), Point(2, -0.5), Point(2, 0.5), Point(-2, 0.5)])]),
            Object(Point(15, 15), [Polygon([Point(-4, -0.5), Point(4, -0.5), Point(4, 0.5), Point(-4, 0.5)])]),
            Object(Point(5, 5), [Polygon([Point(-4, -0.5), Point(4, -0.5), Point(4, 0.5), Point(-4, 0.5)])]),
            Object(Point(17, 5), [Polygon([Point(-2, -0.5), Point(2, -0.5), Point(2, 0.5), Point(-2, 0.5)])]),
            Object(Point(10, 10), [Polygon([Point(-6, -0.5), Point(6, -0.5), Point(6, 0.5), Point(-6, 0.5)])]),
        ],
        regions=dict(
            A=Object(Point(2, 2), [Polygon([Point(-1, -1), Point(1, -1), Point(1, 1), Point(-1, 1)])]),
            B=Object(Point(18, 2), [Polygon([Point(-1, -1), Point(1, -1), Point(1, 1), Point(-1, 1)])]),
            C=Object(Point(18, 18), [Polygon([Point(-1, -1), Point(1, -1), Point(1, 1), Point(-1, 1)])]),
            D=Object(Point(2, 18), [Polygon([Point(-1, -1), Point(1, -1), Point(1, 1), Point(-1, 1)])]),
        ),
    ),
    playroom_default=dict(
        dataset='playroom',
        robot_size=1.0,
        max_step_diff=1.0,
        max_x=20,
        max_y=20,
        obstacles=[
        ],
        regions=dict(
            music_button_on=Object(Point(3, 5), OBJECT_SHAPES['music_button_on']),
            music_button_off=Object(Point(8, 6), OBJECT_SHAPES['music_button_off']),
            bell=Object(Point(12, 12), OBJECT_SHAPES['bell']),
            ball=Object(Point(5, 11), OBJECT_SHAPES['ball']),
            light_switch=Object(Point(9, 15), OBJECT_SHAPES['light_switch']),
            monkey=Object(Point(15, 15), OBJECT_SHAPES['monkey']),
        ),
        regions_desc={k: v for k, v in OBJECT_DESCS.items()},
        env_variables=dict(light=False, ring=False, music=False, monkey_cry=False),
    ),

    playroom_fourrooms=dict(
        dataset='playroom',
        parent='playroom_default',
        obstacles=[
            Object(Point(2.5, 10), [Polygon([Point(-2.5, -.25), Point(2.5, -.25), Point(2.5, .25), Point(-2.5, .25)])]),
            Object(Point(12, 10), [Polygon([Point(-4, -.25), Point(4, -.25), Point(4, .25), Point(-4, .25)])]),
            Object(Point(19.5, 10), [Polygon([Point(-.5, -.25), Point(.5, -.25), Point(.5, .25), Point(-.5, .25)])]),
            Object(Point(10, 2), [Polygon([Point(.25, -2), Point(.25, 2), Point(-.25, 2), Point(-.25, -2)])]),
            Object(Point(10, 10), [Polygon([Point(.25, -3), Point(.25, 3), Point(-.25, 3), Point(-.25, -3)])]),
            Object(Point(10, 18), [Polygon([Point(.25, -2), Point(.25, 2), Point(-.25, 2), Point(-.25, -2)])]),
        ],
    ),

    playroom_maze1=dict(
        dataset='playroom',
        parent='playroom_default',
        obstacles=[
            Object(Point(3, 8), [Polygon([Point(-4, -.25), Point(4, -.25), Point(4, .25), Point(-4, .25)]).rotate(math.pi/6)]),
            Object(Point(15, 6), [Polygon([Point(-6, -.25), Point(6, -.25), Point(6, .25), Point(-6, .25)]).rotate(-math.pi/12)]),
            Object(Point(10, 15), [Polygon([Point(-7, -.25), Point(7, -.25), Point(7, .25), Point(-7, .25)]).rotate(-math.pi/3)]),
            Object(Point(10, 3), [Polygon([Point(-5, -.25), Point(5, -.25), Point(5, .25), Point(-5, .25)]).rotate(0)]),
            Object(Point(5, 15), [Polygon([Point(-2, -.25), Point(2, -.25), Point(2, .25), Point(-2, .25)]).rotate(-math.pi/10)]),
        ],
    ),
)
