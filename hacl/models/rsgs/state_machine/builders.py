#! /usr/bin/env python3
# -*- coding: utf-8 -*-

__all__ = ['default_primitive_constructor', 'get_incompact_state_machine_from_label', 'get_compact_state_machine_from_label']

from hacl.models.rsgs.state_machine import StateMachine


def default_primitive_constructor(obj_name):
    a = StateMachine()
    s = a.add_node()
    t = a.add_node()
    a.add_edge(s, t, obj_name)
    return a, s, t

def get_incompact_state_machine_from_label(label):
    def primitive_constructor(obj_name):
        a = StateMachine()
        s = a.add_node()
        t = a.add_node()
        a.add_edge(s, t, 'eff_' + obj_name)
        return a, s, t

    return StateMachine.from_expression(label, primitive_constructor=primitive_constructor)


def get_compact_state_machine_from_label(label):
    def primitive_constructor(obj_name):
        a = StateMachine()
        s = a.add_node()
        t = a.add_node()
        a.add_edge(s, t, obj_name)
        return a, s, t

    origin = StateMachine.from_expression(label, primitive_constructor=primitive_constructor)
    # print(origin)
    s = StateMachine()
    act2node = dict()
    for x in origin.get_topological_sequence():
        if len(origin.radjs[x]) == 0:  # head
            for (y, l) in origin.adjs[x]:
                t = s.add_node()
                s.add_start(t, 'pre_' + l)
                act2node[(x, l)] = [t]
        elif len(origin.adjs[x]) == 0:  # tail
            for (w, l) in origin.radjs[x]:
                t = s.add_node()
                s.add_end(t)
                for p in act2node[(w, l)]:
                    s.add_edge(p, t, 'eff_' + l)
        else:
            for (w, lw) in origin.radjs[x]:
                for (y, ly) in origin.adjs[x]:
                    t = s.add_node()
                    for p in act2node[(w, lw)]:
                        s.add_edge(p, t, 'eff_' + lw + '&' + 'pre_' + ly)
                    if (x, ly) not in act2node:
                        act2node[(x, ly)] = [t]
                    else:
                        act2node[(x, ly)].append(t)
    # print(s)
    # input()
    return s
