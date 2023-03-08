#! /usr/bin/env python3
# -*- coding: utf-8 -*-


class StateMachine:
    def __init__(self):
        self.node_index = 0
        self.edge_index = dict()
        self.nodes = list()
        self.edges = list()
        self.node2index = dict()
        self.label2edge = dict()
        self.adjs, self.radjs = dict(), dict()
        self.starts = list()
        self.ends = list()
        self.start_note = dict()
        self.end_note = dict()
        self.potential = None

    def __str__(self):
        node_str = ', '.join(
            [
                ('<' + str(x) + (' (%s)' % str(self.start_note[x]) if self.start_note[x] is not None else '') + '>')
                if x in self.starts
                else (
                    ('[' + str(x) + (' (%s)' % str(self.end_note[x]) if self.end_note[x] is not None else '') + ']')
                    if x in self.ends
                    else str(x)
                )
                for x in self.nodes
            ]
        )
        edge_str = '\n'.join(['(%s, %s, %s)' % (str(e[0]), str(e[1]), str(e[2])) for e in self.edges])
        return node_str + '\n' + edge_str

    def add_node(self, label=None):
        if label is None:
            while self.node_index in self.nodes:
                self.node_index += 1
            label = self.node_index
        assert label not in self.adjs.keys()
        self.node2index[label] = len(self.nodes)
        self.nodes.append(label)
        self.adjs[label] = list()
        self.radjs[label] = list()
        self.edge_index[label] = 0
        return label

    def add_edge(self, u, v, label=None):
        assert u in self.adjs.keys() and v in self.adjs.keys()
        if label is None:
            while (u, self.edge_index[u]) in self.label2edge:
                self.edge_index[u] += 1
            label = self.edge_index[u]
        self.edges.append((u, v, label))
        self.label2edge[(u, label)] = v
        self.adjs[u].append((v, label))
        self.radjs[v].append((u, label))
        return label

    def add_start(self, label, note=None):
        assert label in self.nodes
        self.starts.append(label)
        self.start_note[label] = note

    def add_end(self, label, note=None):
        assert label in self.nodes
        self.ends.append(label)
        self.end_note[label] = note

    def set_starts(self, labels, start_note=None):
        self.starts = list()
        self.start_note = dict()
        if start_note is None:
            start_note = dict()
        for label in labels:
            self.add_start(label, note=start_note[label] if label in start_note else None)

    def set_ends(self, labels, end_note=None):
        self.ends = list()
        self.end_note = dict()
        if end_note is None:
            end_note = dict()
        for label in labels:
            self.add_end(label, note=end_note[label] if label in end_note else None)

    def get_edge_label_set(self):
        return set(label for u, v, label in self.edges)

    def get_topological_sequence(self):
        q = []
        d = {v: 0 for v in self.nodes}
        for (u, v, l) in self.edges:
            d[v] += 1
        for v in d:
            if d[v] == 0:
                q.append(v)
        h = 0
        while h < len(q):
            x = q[h]
            h += 1
            for (y, l) in self.adjs[x]:
                d[y] -= 1
                if d[y] == 0:
                    q.append(y)
        assert len(q) == len(self.nodes)
        return tuple(q)

    def critical_path_embed(self):
        n = len(self.nodes)
        s = [0 for i in range(n)]
        lb = 0
        for x in reversed(self.get_topological_sequence()):
            lb = min(lb, s[self.node2index[x]])
            for y, label in self.radjs[x]:
                s[self.node2index[y]] = min(s[self.node2index[y]], s[self.node2index[x]] - 1)
        for x in self.starts:
            s[self.node2index[x]] = lb
        for i in range(n):
            s[i] = (s[i] - lb + 1) / (1 - lb)
        return s

    def update_potential(self):
        self.potential = self.critical_path_embed()

    def embed_state_machine(self, u, v, a, s, t):
        """
        :param u: from node u
        :param v: to node v
        :param a: embed state machine a between node u and node v
        :param s: start state of a, matching u
        :param t: end state of a, matching v
        :return: None, operation done inplace
        """
        node_map = {s: u, t: v}
        for x in a.nodes:
            if x not in node_map:
                node_map[x] = self.add_node()
        for (x, y, label) in a.edges:
            xx, yy = node_map[x], node_map[y]
            self.add_edge(xx, yy, label)

    @classmethod
    def _operate(cls, operators, elements, primitive_constructor, str_rep=False):
        assert len(operators) > 0 and len(operators) + 1 == len(elements)
        if str_rep:
            return '(' + ''.join([elements[i] + operators[i] for i in range(len(operators))]) + elements[-1] + ')'
        elements = [primitive_constructor(e) if isinstance(e, str) else e for e in elements]
        if '&' in operators:
            for o in operators:
                assert o == '&'
            n = len(elements)
            a = cls()
            s, t = 0, 2 ** n - 1
            for mask in range(2 ** n):
                a.add_node()
            for mask in range(2 ** n):
                for i, e in enumerate(elements):
                    if ((mask >> i) & 1) == 0:
                        new_mask = mask | 1 << i
                        a.embed_state_machine(mask, new_mask, e[0], e[1], e[2])
            return a, s, t
        if '|' in operators:
            for o in operators:
                assert o == '|'
            a = cls()
            s = a.add_node()
            t = a.add_node()
            for i, e in enumerate(elements):
                a.embed_state_machine(s, t, e[0], e[1], e[2])
            return a, s, t
        if '>' in operators:
            for o in operators:
                assert o == '>'
            a = cls()
            t = s = a.add_node()
            for i, e in enumerate(elements):
                new_t = a.add_node()
                a.embed_state_machine(t, new_t, e[0], e[1], e[2])
                t = new_t
            return a, s, t
        raise NotImplementedError()

    @classmethod
    def default_primitive_constructor(cls, desc):
        a = StateMachine()
        s = a.add_node()
        t = a.add_node()
        a.add_edge(s, t, desc)
        return a, s, t

    @classmethod
    def from_expression(cls, exp, primitive_constructor, str_rep=False):
        opt_pri = {'&': 0, '|': 1, '>': 2}
        stack = []
        cur = None
        for c in exp:
            if c in opt_pri or c in ('(', ')'):
                if c == '(':
                    assert cur is None
                    stack.append(['(', None])
                elif c == ')':
                    assert cur is not None
                    while stack[-1][0] != '(':
                        stack[-1][1].append(cur)
                        cur = cls._operate(*stack.pop(), primitive_constructor, str_rep)
                    stack.pop()
                else:
                    assert cur is not None
                    while len(stack) > 0 and stack[-1][0] != '(' and opt_pri[c] > opt_pri[stack[-1][0][0]]:
                        stack[-1][1].append(cur)
                        cur = cls._operate(*stack.pop(), primitive_constructor, str_rep)
                    if len(stack) > 0 and stack[-1][0] != '(' and opt_pri[c] == opt_pri[stack[-1][0][0]]:
                        stack[-1][0].append(c)
                        stack[-1][1].append(cur)
                    else:
                        stack.append([[c], [cur]])
                    cur = None
            else:
                cur = c if cur is None else cur + c
        assert cur is not None
        while len(stack) > 0:
            stack[-1][1].append(cur)
            cur = cls._operate(*stack.pop(), primitive_constructor, str_rep)
        if isinstance(cur, str):
            cur = primitive_constructor(cur)
        a, s, t = cur
        a.set_starts([s])
        a.set_ends([t])
        return a


if __name__ == '__main__':

    def primitive_constructor(desc):
        a = StateMachine()
        s = a.add_node()
        t = a.add_node()
        a.add_edge(s, t, desc)
        return a, s, t

    print(StateMachine.from_expression('A&(B>C|D)', primitive_constructor, str_rep=False)[0])
