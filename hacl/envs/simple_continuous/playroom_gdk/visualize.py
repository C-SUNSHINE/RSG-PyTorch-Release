#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import six
from pprint import pprint

from hacl.algorithms.rrt.rrt import traverse_rrt_bfs
from hacl.engine.tk.drawing_window import DrawingWindow
from hacl.utils.geometry_2d import Line, Point

__all__ = ['visualize_rrt', 'visualize_rrt_graph', 'visualize_robot_path', 'visualize_problem_and_solution']


def visualize_rrt(pspace, rrt, window=None, color='black', node_color=None):
    if window is None or isinstance(window, six.string_types):
        window_name = window or 'RRT'
        window = DrawingWindow(600, 600, 0, pspace.map_x, 0, pspace.map_y, window_name)

    for node_id, node in enumerate(traverse_rrt_bfs(rrt.roots)):
        node_point = Point(node.config[0], node.config[1])
        for child in node.children:
            child_point = Point(child.config[0], child.config[1])
            Line(node_point, child_point).draw(window, color=color, width=1)

            if node_color is not None:
                node_point.draw(window, color=node_color[node_id])

    return window


def visualize_rrt_graph(
    pspace,
    graph: 'RRTizedEnv',
    highlight_path: list,
    window=None,
    color='black',
    highlight_color='red',
    node_color=None,
):
    if window is None or isinstance(window, six.string_types):
        window_name = window or 'RRT'
        window = DrawingWindow(600, 600, 0, pspace.map_x, 0, pspace.map_y, window_name)

    highlight_path_set = set(highlight_path)

    index2state = graph.get_state_list_by_index()
    for x, es in graph.edges.items():
        node1 = Point(index2state[x][0], index2state[x][1])
        for y, _, _ in es:
            if index2state[x] not in highlight_path_set and index2state[y] not in highlight_path_set:
                node2 = Point(index2state[y][0], index2state[y][1])
                Line(node1, node2).draw(window, color=color, width=1)

    for x, y in zip(highlight_path[:-1], highlight_path[1:]):
        Line(Point(x[0], x[1]), Point(y[0], y[1])).draw(window, color=highlight_color, width=2)

    for node_id, config in enumerate(graph.states):
        if node_color is not None:
            Point(config[0], config[1]).draw(window, color=node_color[node_id])

    return window


def visualize_robot_path(pspace, path, window, color=None, start=None, goal=None, play_step=True):
    if start:
        pspace.robot.configuration(path[0]).draw(window, 'orange')
        if play_step:
            input('Next?')
    for i in range(1, len(path) - 1):  # don't draw start and end
        pspace.robot.configuration(path[i]).draw(window, color)
        window.update()
        if play_step:
            input('Next?')
    if goal:
        pspace.robot.configuration(path[-1]).draw(window, 'green')
        if play_step:
            input('Next?')


def visualize_problem(pspace, path, window=None, **kwargs):
    if window is None or isinstance(window, six.string_types):
        window_name = window or 'RRT Planing'
        window = DrawingWindow(600, 600, 0, pspace.map_x, 0, pspace.map_y, window_name)

    for obs in pspace.obstacles:
        obs.draw(window, 'red')
    for name, region in pspace.regions.items():
        color = 'blue'
        if name in pspace.regions_desc:
            if 'color' in pspace.regions_desc[name]:
                color = pspace.regions_desc[name]['color']
        region.draw(window, color)
    start_state = path[0]
    pspace.robot.configuration(start_state).draw(window, 'orange')

    return window


def visualize_problem_and_solution(pspace, path=None, rrt=None, window=None, play_step=True):
    if window is None or isinstance(window, six.string_types):
        window_name = window or 'RRT Planing'
        window = DrawingWindow(600, 600, 0, pspace.map_x, 0, pspace.map_y, window_name)

    def draw_problem():
        for obs in pspace.obstacles:
            obs.draw(window, 'red')
        for name, region in pspace.regions.items():
            color = 'blue'
            if name in pspace.regions_desc:
                if 'color' in pspace.regions_desc[name]:
                    color = pspace.regions_desc[name]['color']
            region.draw(window, color)
        if pspace.start_state is not None:
            pspace.robot.configuration(pspace.start_state).draw(window, 'orange')
        elif path is not None and len(path) > 0 and path[0] is not None:
            pspace.robot.configuration(path[0]).draw(window, 'orange')
        else:
            print("Start state missing.")
        if pspace.goal_state is not None:
            pspace.robot.configuration(pspace.goal_state).draw(window, 'green')
        elif path is not None and len(path) > 0 and path[-1] is not None:
            pass
        else:
            print("Goal state missing.")

    print('Visualizing the problem space.')
    draw_problem()

    if rrt is not None:
        input('Press any key to visualize the RRT tree(s).')

        print('Visualizing the RRT tree(s).')
        if isinstance(rrt, (tuple, list)):
            if len(rrt) == 2:
                visualize_rrt(pspace, rrt[0], window, color='orange')
                visualize_rrt(pspace, rrt[1], window, color='green')
            else:
                for r in rrt:
                    visualize_rrt(pspace, r, window)
        else:
            visualize_rrt(pspace, rrt, window)

    if path is not None:
        input('Press any key to visualize the solution.')

        print('Visualizing the solution.')
        spath = []
        if len(path) <= 1:
            spath = path
        else:
            for i in range(1, len(path)):
                _, sub_path = pspace.cspace.gen_path(path[i - 1], path[i])
                spath.extend(sub_path)

        # make sure path is collision free
        if any([pspace.collide(c) for c in spath]):
            print('Collision in smoothed path.')

        pprint(spath)

        window.clear()
        draw_problem()
        visualize_robot_path(
            pspace, spath, window, color='yellow', start=pspace.start_state is None, goal=pspace.goal_state is None, play_step=play_step
        )

    while input('End? (y or n)') != 'y':
        pass
