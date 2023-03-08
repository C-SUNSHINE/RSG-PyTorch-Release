#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from hacl.algorithms.planning.astar_planning_config import SKILL_DEPENDENCIES

from queue import PriorityQueue

from tqdm import tqdm

class AstarPlanningEngine(object):
    class Node(object):
        def __init__(self, state, t, heuristic=0, parent=None, action=None):
            self.state = state
            self.t = t
            self.heuristic = heuristic
            self.parent = parent
            self.action = action

    class StepAction(object):
        def __init__(self, x, y):
            self.x = x
            self.y = y

    def __init__(
        self, env_name, env_args, broadcast_env, init_values, label, state_machine, action_set=None, action_cost=None, prune=True, **kwargs
    ):
        self.env_name = env_name
        self.env_args = env_args
        self.broadcast_env = broadcast_env
        self.env_args = self.broadcast_env.env.complete_env_args(self.env_args)
        self.init_values = init_values
        self.label = label
        self.state_machine = state_machine
        self.action_set = action_set
        self.action_cost = action_cost
        assert self.action_cost > 0
        self.kept_args = kwargs
        self.prune = prune

        self.st2node = None
        self.heap = None
        self.openings = None
        self.closed = None

    @classmethod
    def get_env_transitions(cls, s, env, action_set, action_cost):
        trans = []
        for a in action_set:
            env.load_from_symbolic_state(s)
            success = env.action(a)
            if success:
                new_s = env.get_symbolic_state()
                trans.append((new_s, -action_cost, a))
        return trans

    def augment(self, node):
        s, t, h = node.state, node.t, node.heuristic
        trans = self.get_env_transitions(s, self.broadcast_env.env, self.action_set, self.action_cost)
        trans = [(ns, t, c, a) for (ns, c, a) in trans]
        for u, el in self.state_machine.adjs[t]:
            dh = self.init_values(el, self.broadcast_env.states2tensor([s]), ignore_pre=True).item()
            a = self.StepAction(t, u)
            trans.append((s, u, dh, a))

        for new_s, new_t, dh, a in trans:
            new_node = self.Node(new_s, new_t, h + dh, parent=node, action=a)
            if new_s not in self.st2node[new_t] or h + dh > self.st2node[new_t][new_s].heuristic:
                self.st2node[new_t][new_s] = new_node
                self.heap[new_t].put((-new_node.heuristic, new_s))

    def _close_node(self, t):
        if not self.closed[t]:
            self.closed[t] = True
            for u, el in self.state_machine.radjs[t]:
                self.openings[u] -= 1
            if t in self.state_machine.ends:
                self.openings[t] -= 1

    class TreeNode(object):
        def __init__(self, index, parent=None, skill=None):
            self.index = index
            self.parent = parent
            self.skill = skill
            self.skill_sequence = '*' if parent is None else parent.skill_sequence + '>' + skill
            self.st2node = dict()
            self._active_states = None

        def add_node(self, node):
            s, t, h = node.state, node.t, node.heuristic
            if s not in self.st2node or self.st2node[s].heuristic < h:
                self.st2node[s] = node
                return True
            return False

        @classmethod
        def check_terminate(cls, states, terminate_checker):
            return AstarPlanningEngine.check_terminate(states, terminate_checker)

        @classmethod
        def find_terminate(cls, states, terminate_checker):
            return AstarPlanningEngine.find_terminate(states, terminate_checker)

        def augment(self, env, action_set, action_cost, n_iters=5000, use_tqdm=False, terminate_checker=None):
            search_count = 0
            if self.check_terminate(self.st2node.keys(), terminate_checker):
                return 0
            heap = PriorityQueue()
            self._active_states = set()
            for state, node in self.st2node.items():
                heap.put((-node.heuristic, node.state))
            if use_tqdm:
                iters = tqdm(range(n_iters), total=n_iters)
                iters.set_description(self.skill_sequence)
            else:
                iters = range(n_iters)
            for it in iters:
                v, s = None, None
                while not heap.empty():
                    v, s = heap.get()
                    if -self.st2node[s].heuristic != v:
                        v, s = None, None
                        continue
                    break
                if v is None:
                    break
                search_count += 1
                trans = AstarPlanningEngine.get_env_transitions(s, env, action_set, action_cost)
                env.load_from_symbolic_state(s)
                node = self.st2node[s]
                self._active_states.add(s)
                for ns, c, a in trans:
                    if ns not in self.st2node or node.heuristic + c > self.st2node[ns].heuristic:
                        new_node = AstarPlanningEngine.Node(ns, self.index, node.heuristic + c, parent=node, action=a)
                        self.st2node[ns] = new_node
                        heap.put((-new_node.heuristic, new_node.state))
                        if terminate_checker(ns):
                            return search_count
            if use_tqdm:
                iters.close()
            return search_count

        def skill_transit(self, new_tree_node, skill, broadcast_env=None, init_values=None):
            state_list = list(self.st2node.keys() if self._active_states is None else self._active_states)
            s_tensor = broadcast_env.states2tensor(state_list)
            edge_label = 'eff_' + skill
            rewards = init_values(edge_label, s_tensor).detach().tolist()
            best_heuristic = None
            for i in range(len(state_list)):
                state, reward = state_list[i], rewards[i]
                node = self.st2node[state]
                new_node = AstarPlanningEngine.Node(
                    state,
                    new_tree_node.index,
                    node.heuristic + reward,
                    parent=node,
                    action=AstarPlanningEngine.StepAction(self.index, new_tree_node.index)
                )
                new_tree_node.add_node(new_node)
                if best_heuristic is None or new_node.heuristic > best_heuristic:
                    best_heuristic = new_node.heuristic
            return best_heuristic

    def backward(self, end_node, search_count=0):
        states, actions = [], []
        node = end_node
        while node is not None:
            if not isinstance(node.action, self.StepAction):
                states.append(node.state)
                if node.action is not None:
                    actions.append(node.action)
            node = node.parent
        return tuple(reversed(states)), tuple(reversed(actions)), search_count

    @classmethod
    def check_terminate(cls, states, terminate_checker):
        return any(terminate_checker(s) for s in states)

    @classmethod
    def find_terminate(cls, states, terminate_checker):
        res = list(filter(terminate_checker, states))
        if len(res) > 0:
            return res[0]
        return None

    def search_plan_brute(self, start_state, n_iters=None, use_tqdm=False, terminate_checker=None):
        tree = [AstarPlanningEngine.TreeNode(0)]
        add_node_success = tree[0].add_node(AstarPlanningEngine.Node(start_state, 0, 0))
        assert add_node_success
        search_count = 0
        search_count += tree[0].augment(self.broadcast_env.env, self.action_set, self.action_cost, n_iters=n_iters,
                                        use_tqdm=use_tqdm, terminate_checker=terminate_checker)
        if self.check_terminate(tree[0].st2node.keys(), terminate_checker):
            return self.backward(
                tree[0].st2node[self.find_terminate(tree[0].st2node, terminate_checker)],
                search_count=search_count
            )
        else:
            return (start_state,), tuple(), search_count

    def search_plan_hierarchical(self, start_state, n_iters=None, use_tqdm=False, terminate_checker=None):
        primitives = self.env_args['search_primitives'][self.label][:]
        tree = [AstarPlanningEngine.TreeNode(0)]
        add_node_success = tree[0].add_node(AstarPlanningEngine.Node(start_state, 0, 0))
        assert add_node_success
        search_count = 0
        heap = PriorityQueue()
        heap.put((0, 0))
        for k in range(120):
            heu, t = heap.get()
            tree_node = tree[t]
            search_count += tree_node.augment(self.broadcast_env.env, self.action_set, self.action_cost,
                                              n_iters=n_iters, use_tqdm=use_tqdm, terminate_checker=terminate_checker)
            if self.check_terminate(tree_node.st2node.keys(), terminate_checker):
                return self.backward(
                    tree_node.st2node[self.find_terminate(tree_node.st2node.keys(), terminate_checker)],
                    search_count=search_count
                )

            for skill in primitives:
                if skill not in tree_node.skill_sequence:
                    new_tree_node = AstarPlanningEngine.TreeNode(len(tree), parent=tree_node, skill=skill)
                    best_heu = tree_node.skill_transit(new_tree_node, skill, self.broadcast_env, self.init_values)
                    tree.append(new_tree_node)
                    heap.put((-best_heu, new_tree_node.index))

                    if self.check_terminate(new_tree_node.st2node.keys(), terminate_checker):
                        return self.backward(
                            new_tree_node.st2node[self.find_terminate(new_tree_node.st2node, terminate_checker)],
                            search_count=search_count
                        )
        return (start_state,), tuple(), search_count


    def _skill_sequence_generator(self, final_skills, dependencies):
        q = [[s] for s in final_skills]
        qh = 0
        while True:
            s = q[qh]
            yield s
            qh += 1
            s_dep = set()
            for si in s:
                s_dep.update(set(dependencies[si] if si in dependencies else []))
            for t in s_dep:
                if t not in s:
                    q.append([t] + s)

    def _skill_sequence_recommender(self, final_skills, dependencies_score):
        dependencies = {s: {s_ for s_ in dependencies_score if dependencies_score[s][s_] > 1e-9} for s in dependencies_score}
        n_max_depth = 6
        length_bias_coef = 0.9
        q = {i + 1: [] for i in range(n_max_depth)}
        q_all = []
        for seq in self._skill_sequence_generator(final_skills, dependencies):
            if len(seq) > n_max_depth:
                break
            score = 1
            for i in range(len(seq) - 2, -1, -1):
                not_dep = 1
                for j in range(i + 1, len(seq)):
                    not_dep *= (1 - dependencies_score[seq[j]][seq[i]])
                score *= (1 - not_dep) * length_bias_coef
            q[len(seq)].append((seq, score))
            q_all.append((seq, score))
        for i in range(1, n_max_depth + 1):
            q[i] = list(reversed(sorted(q[i], key=lambda x: x[1])))
        q_all = list(reversed(sorted(q_all, key=lambda x: x[1])))
        ptr = {i: 0 for i in range(1, n_max_depth + 1)}

        method = 'v1'
        if method == 'v1':
            for (seq, score) in q_all:
                yield seq
        elif method == 'v2':
            while True:
                for i in range(1, n_max_depth + 1):
                    if ptr[i] < len(q[i]):
                        yield q[i][ptr[i]][0]
                        ptr[i] += 1

    def search_plan_dependency(self, start_state, n_iters=None, use_tqdm=False, terminate_checker=None,
                               dependency_base=False):
        final_skills = self.env_args['final_skills'][self.label][:]
        if not dependency_base:
            dependencies_score = SKILL_DEPENDENCIES
        else:
            dependencies_score = {
                s1: {s2: 1 if SKILL_DEPENDENCIES[s1][s2] > 1e-5 else 0 for s2 in SKILL_DEPENDENCIES[s1]} for s1 in
                SKILL_DEPENDENCIES
            }
        search_count = 0

        print("Recommender for ", final_skills, " :")
        for i, c in enumerate(self._skill_sequence_recommender(final_skills, dependencies_score)):
            if i == 20:
                break
            print('%2d' % (i + 1), ':', c)

        skill_sequences_rec = iter(self._skill_sequence_recommender(final_skills, dependencies_score))

        tree = [AstarPlanningEngine.TreeNode(0)]
        add_node_success = tree[0].add_node(AstarPlanningEngine.Node(start_state, 0, 0))
        assert add_node_success

        seq_to_tree_node = {'*': tree[0]}
        augmented = set()

        for skill_seq in skill_sequences_rec:
            if search_count >= 100000:
                print('search_count exceeded')
                break
            # heap = PriorityQueue()
            # heap.put((0, 0))
            cur_seq = '*'
            for skill in skill_seq:
                tree_node = seq_to_tree_node[cur_seq]
                cur_seq += '>' + skill
                if tree_node.skill_sequence not in augmented:
                    search_count += tree_node.augment(self.broadcast_env.env, self.action_set, self.action_cost,
                                                      n_iters=n_iters, use_tqdm=use_tqdm, terminate_checker=terminate_checker)
                    augmented.add(tree_node.skill_sequence)
                    if self.check_terminate(tree_node.st2node.keys(), terminate_checker):
                        return self.backward(
                            tree_node.st2node[self.find_terminate(tree_node.st2node.keys(), terminate_checker)],
                            search_count=search_count
                        )

                assert skill not in tree_node.skill_sequence
                if cur_seq not in seq_to_tree_node:
                    new_tree_node = AstarPlanningEngine.TreeNode(len(tree), parent=tree_node, skill=skill)
                    tree_node.skill_transit(new_tree_node, skill, self.broadcast_env, self.init_values)
                    tree.append(new_tree_node)
                    seq_to_tree_node[cur_seq] = new_tree_node

                    if self.check_terminate(new_tree_node.st2node.keys(), terminate_checker):
                        return self.backward(
                            new_tree_node.st2node[self.find_terminate(new_tree_node.st2node, terminate_checker)],
                            search_count=search_count
                        )
        return (start_state,), tuple(), search_count

    def search_plan(self, start_state, n_iters=5000, use_tqdm=False, search_policy=None, terminate_checker=None, **kwargs):
        if search_policy == 'brute':
            return self.search_plan_brute(start_state, n_iters=n_iters, use_tqdm=use_tqdm, terminate_checker=terminate_checker)
        elif search_policy == 'hierarchical':
            return self.search_plan_hierarchical(start_state, n_iters=n_iters, use_tqdm=use_tqdm, terminate_checker=terminate_checker)
        elif search_policy == 'dependency':
            return self.search_plan_dependency(start_state, n_iters=n_iters, use_tqdm=use_tqdm, terminate_checker=terminate_checker)
        elif search_policy == 'dependency_base':
            return self.search_plan_dependency(start_state, n_iters=n_iters, use_tqdm=use_tqdm, terminate_checker=terminate_checker, dependency_base=True)
        else:
            raise ValueError('Invalid search plan method')

    def search(self, constraint, n_iters=5000, use_tqdm=True, plan_search=False, **kwargs):
        if plan_search:
            return self.search_plan(constraint[0], n_iters=n_iters, use_tqdm=use_tqdm, search_policy=plan_search, terminate_checker=constraint[1], **kwargs)
        else:
            start_state = constraint
        self.st2node = {t: dict() for t in self.state_machine.nodes}
        self.heap = {t: PriorityQueue() for t in self.state_machine.nodes}

        self.openings = {
            t: len(self.state_machine.adjs[t]) + (1 if t in self.state_machine.ends else 0)
            for t in self.state_machine.nodes
        }
        self.closed = {t: False for t in self.state_machine.nodes}

        self.broadcast_env.env.load_from_symbolic_state(start_state)

        for t in self.state_machine.starts:
            if self.state_machine.start_note[t] is not None:
                h = float(
                    self.init_values(
                        self.state_machine.start_note[t], self.broadcast_env.states2tensor([start_state]), ignore_pre=True,
                    ).item()
                )
            else:
                h = 0
            self.st2node[t][start_state] = self.Node(start_state, t, heuristic=h, parent=None, action=None)
            self.heap[t].put((-h, start_state))
            self._close_node(t)

        topo_seq = self.state_machine.get_topological_sequence()
        self.state_machine.update_potential()
        n_non_ends = len([t for t in self.state_machine.nodes if t not in self.state_machine.ends])
        n_iters = round(n_iters * (n_non_ends ** 1.00))
        n_iters_each = n_iters // n_non_ends
        iter_tqdm = tqdm(total=n_iters) if use_tqdm else None
        search_count = 0
        for t in topo_seq:
            if t not in self.state_machine.ends:
                for it in range(n_iters_each):
                    iter_tqdm.update(1)
                    if self.heap[t].empty():
                        continue
                    v, s = self.heap[t].get()
                    if -self.st2node[t][s].heuristic != v:
                        continue
                    self.augment(self.st2node[t][s])
                    search_count += 1
        if use_tqdm:
            iter_tqdm.close()
        best_h = None
        best_end = None

        for t in self.state_machine.ends:
            for node in self.st2node[t].values():
                if best_h is None or node.heuristic > best_h:
                    best_h = node.heuristic
                    best_end = node

        return self.backward(best_end, search_count=search_count)
