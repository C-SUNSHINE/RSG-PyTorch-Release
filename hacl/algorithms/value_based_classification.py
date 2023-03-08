#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List, Tuple, Union

import jacinle
import jacinle.random as jacrandom
import torch
from scipy.special import logsumexp

from .graph_search import GraphizedEnv
from hacl.models.rsgs.state_machine.algorithms import max_sum_multistep, max_sum_state_machine

"""
The environments need to support a "get_symbolic_state" call.
"""


class OnlineActionClassifier(object):
    def __init__(self):
        self.current_state = None
        self.action_cummulated_likelihood = None
        self.debug = '-----'

    def restart(self, state=None):
        raise NotImplementedError()

    def action(self, next_state):
        raise NotImplementedError()

    def display(self):
        fmt = '[Classifier] {}; '.format(
            max(self.action_cummulated_likelihood, key=self.action_cummulated_likelihood.get)
        )
        for k, v in self.action_cummulated_likelihood.items():
            fmt += f'{k} = {v:.3f} '

        fmt = fmt.strip() + '\n[Classifier] ' + self.debug
        return fmt.strip()

    def inject(self, env):
        old_restart = env._restart

        def new_restart(*args, **kwargs):
            rv = old_restart(*args, **kwargs)
            self.restart(env.get_symbolic_state())
            return rv

        env._restart = new_restart

        old_action = env._action

        def new_action(*args, **kwargs):
            rv = old_action(*args, **kwargs)
            self.action(env.get_symbolic_state())
            return rv

        env._action = new_action

        old_emulate_cli_info = env.emulate_cli_info

        def new_emulate_cli_info():
            rv = old_emulate_cli_info()
            return rv + '\n' + self.display()

        env.emulate_cli_info = new_emulate_cli_info


class ValueBasedActionClassifier(OnlineActionClassifier):
    def __init__(
        self,
        graph: Union[GraphizedEnv, Tuple],
        action_value_tensors: dict,
        step_value_tensors=None,
        n_stages=None,
        # stage_indicator=None,
        action_state_machine=None,
    ):
        super().__init__()
        self.graph = graph
        self.action_value_tensors = action_value_tensors
        self.step_value_tensors = step_value_tensors
        self.n_stages = n_stages
        # self.stage_indicator = stage_indicator
        self.action_state_machine = action_state_machine

    def restart(self, state):
        self.current_state = self.graph.state2index[state]
        self.action_cummulated_likelihood = {k: 0 for k in self.action_value_tensors}

    def action(self, next_state):
        edges = self.graph.get_by_index(self.current_state)
        next_state = self.graph.state2index[next_state]

        this_step_likelihood = dict()
        for action_type, value_tensor in self.action_value_tensors.items():
            this_action_value = None
            all_action_values = list()
            for v, _, _ in edges:
                all_action_values.append(value_tensor[v])
                if v == next_state:
                    this_action_value = value_tensor[v]

            # essentially, Softmax(QValue[s, a])
            if this_action_value is None:
                this_step_likelihood[action_type] = None
            else:
                this_step_likelihood[action_type] = this_action_value - logsumexp(all_action_values)

        self.debug = 'this_step_likelihood=' + str(this_step_likelihood)

        if None not in tuple(this_step_likelihood.values()):
            for k, v in this_step_likelihood.items():
                self.action_cummulated_likelihood[k] += v

        self.current_state = next_state

    def prob(self, states, actions):
        if self.n_stages is not None:
            return self.prob_multistep(states, actions)
        if self.action_state_machine is not None:
            return self.prob_state_machine(states, actions)
        results = dict()
        for action_type, value_tensor in self.action_value_tensors.items():
            results[action_type] = 0
            for s, a in zip(states, actions):
                this_action_value = None
                all_action_values = list()
                edges = self.graph.get_by_index(self.graph.state2index[s])
                for v, aa, _ in edges:
                    v_value = value_tensor[v]
                    all_action_values.append(v_value)
                    if aa == a:
                        this_action_value = v_value

                # essentially, Softmax(QValue[s, a])
                if this_action_value is None:
                    pass
                else:
                    results[action_type] += this_action_value - torch.logsumexp(torch.stack(all_action_values), dim=-1)
        return results

    def prob_multistep(self, states, actions, cuda=True):
        device = torch.device('cuda:0') if cuda and torch.cuda.is_available() else torch.device('cpu')
        results = dict()
        for action_type, value_tensor in self.action_value_tensors.items():
            assert len(value_tensor) == self.n_stages
            results[action_type] = torch.zeros(self.n_stages, len(actions)).to(device)
            for i, (s, a) in enumerate(zip(states, actions)):
                for stage in range(self.n_stages):
                    this_action_value = None
                    all_action_values = list()
                    edges = self.graph.get_by_index(self.graph.state2index[s])
                    for v, aa, _ in edges:
                        v_value = value_tensor[stage][v]
                        all_action_values.append(v_value)
                        if aa == a:
                            this_action_value = v_value

                    # essentially, Softmax(QValue[s, a])
                    if this_action_value is None:
                        pass
                    else:
                        results[action_type][stage, i] = this_action_value - torch.logsumexp(
                            torch.stack(all_action_values), dim=-1
                        )
            results[action_type] = max_sum_multistep(results[action_type])
        return results

    def prob_state_machine(self, states, actions, keep_step_results=False, device='cuda:0'):
        results = dict()
        step_results = dict()
        for action_type, value_tensor in self.action_value_tensors.items():
            step_value_tensor = self.step_value_tensors[action_type]
            state_machine = self.action_state_machine[action_type]
            assert set(value_tensor.keys()) == set(state_machine.label2edge.keys())
            assert set(step_value_tensor.keys()) == set(state_machine.label2edge.keys())
            edges = set(state_machine.label2edge.keys())
            action_prob = {e: torch.zeros(len(actions)).to(device) for e in edges}
            step_prob = {e: torch.zeros(len(states)).to(device) for e in edges}
            if not isinstance(self.graph, GraphizedEnv):  # TODO replace this by trajptr
                states_tensor, transition, action_set = self.graph
                nr_states = states_tensor.size(0)
                traj_state_tensor = GridWorldBroadcastEngine.states2tensor(states).to(states_tensor.device)

                def locate_state(state_tensor):
                    comp = (
                        1
                        - states_tensor.eq(
                        state_tensor.unsqueeze(0).repeat(nr_states, *([1] * len(state_tensor.size())))
                    ).type(torch.long)
                    ).sum(dim=(1, 2))
                    for pos in range(nr_states):
                        if comp[pos] == 0:
                            break
                    assert pos < nr_states
                    return pos

                cur = locate_state(traj_state_tensor[0])
                action2actionid = {a: i for i, a in enumerate(action_set)}
                state_index = {}
                for k, (s, a) in enumerate(zip(states, actions + [None])):
                    state_index[s] = cur
                    assert torch.equal(states_tensor[cur], traj_state_tensor[k])
                    if a is not None:
                        aid = action2actionid[a]
                        if transition[cur, aid] == -1:
                            cur = None
                            break
                        cur = transition[cur, aid]
                if cur is None:  # flattened tree
                    state_index = {}
                    for k, s in enumerate(states):
                        state_index[s] = locate_state(traj_state_tensor[k])

            for i, (s, a) in enumerate(zip(states, actions + [None])):
                for e in edges:
                    this_action_value, this_action_index = None, None
                    all_action_values = list()
                    if isinstance(self.graph, GraphizedEnv):
                        cur = self.graph.state2index[s]
                        for v, aa, _ in self.graph.get_by_index(cur):
                            v_value = value_tensor[e][v]
                            all_action_values.append(v_value)
                            if aa == a:
                                this_action_value = v_value
                                this_action_index = len(all_action_values) - 1
                        step_value = step_value_tensor[e][cur]
                    else:
                        _, transition, action_set = self.graph
                        cur = state_index[s]
                        for aid, aname in enumerate(action_set):
                            to = transition[cur][aid]
                            v_value = value_tensor[e][to] if to != -1 else torch.zeros_like(value_tensor[e][to]) - 1e9
                            all_action_values.append(v_value)
                            if aname == a:
                                this_action_value = v_value
                                this_action_index = len(all_action_values) - 1
                        step_value = step_value_tensor[e][cur]
                    all_action_value_tensor = torch.stack(all_action_values)
                    all_action_prob_tensor = torch.log_softmax(
                        all_action_value_tensor - all_action_value_tensor.max(dim=-1)[0], dim=-1
                    )
                    if this_action_value is not None:
                        # print(action_prob[e][i].size(), all_action_prob_tensor.size(), this_action_index)
                        action_prob[e][i] = all_action_prob_tensor[this_action_index]
                    # Compute the probability of switching stage TODO check validity
                    max_action_prob, max_action_id = all_action_prob_tensor.max(dim=-1)
                    step_prob[e][i] = torch.min(step_value - max_action_prob, torch.zeros_like(step_value))
            results[action_type] = max_sum_state_machine(action_prob, step_prob, state_machine)
            step_results[action_type] = (action_prob, step_prob)
        if keep_step_results:
            return results, step_results
        return results


class ValueBasedActionSamplerMode(jacinle.JacEnum):
    BOLTZMANN = 'boltzmann'
    EPSGREEDY = 'epsgreedy'


class ValueBasedActionSampler(object):
    def __init__(
        self,
        graph: GraphizedEnv,
        value: Union[torch.Tensor, List[torch.Tensor]],
        mode: Union[str, ValueBasedActionSamplerMode] = 'epsgreedy',
        rng=None,
        n_stages=None,
        stage_indicator=None,
        rationality=None,
    ):
        self.graph = graph
        self.value = value
        self.mode = ValueBasedActionSamplerMode.from_string(mode)
        self.rng = rng
        self.n_stages = n_stages
        self.stage_indicator = stage_indicator
        self.rationality = rationality if rationality is not None else 0.9
        self._stage = None

        if self.rng is None:
            import random

            self.rng = random.Random(jacrandom.gen_seed())

    def init_stage(self):
        if self.n_stages is not None:
            self._stage = 0

    def clear_stage(self):
        self._stage = None

    def _get_stage(self, state):
        if self._stage is None:
            stage = self.stage_indicator(state)
        else:
            indicated_stage = self.stage_indicator(state)
            if indicated_stage <= self._stage + 1:
                self._stage = indicated_stage
            stage = self._stage
        return stage

    def __call__(self, state):
        state_index = self.graph.state2index[state]
        edges = self.graph.get_by_index(state_index)
        all_action_values = dict()
        for v, action, _ in edges:
            if self.n_stages is None:
                all_action_values[action] = self.value[v]
            else:
                stage = self._get_stage(state)
                assert stage < self.n_stages
                all_action_values[action] = self.value[stage][v]

        if self.mode is ValueBasedActionSamplerMode.EPSGREEDY:
            if self.rng.random() > self.rationality:
                return self.rng.choice(list(all_action_values.keys()))
            else:
                return max(all_action_values, key=all_action_values.get)
        else:
            raise ValueError('Unknown action sampler mode: {}.'.format(self.mode))

    def get_reward(self, state):
        if self.n_stages is None:
            return self.value[self.graph.state2index[state]]
        else:
            stage = self._get_stage(state)
            return self.value[stage][self.graph.state2index[state]]


class ValueBasedTrajectoryCollector(object):
    def __init__(self, env, sampler, terminal_func, max_steps=10, with_rewards=False):
        self.env = env
        self.sampler = sampler
        self.terminal_func = terminal_func
        self.max_steps = max_steps
        self.with_rewards = with_rewards

    def collect(self, nr_examples):
        env = self.env
        trajs = list()
        for i in range(nr_examples):
            env.restart()
            traj_states = [env.get_symbolic_state()]
            traj_actions = list()
            reward = None
            self.sampler.init_stage()
            for j in range(self.max_steps):
                action = self.sampler(traj_states[-1])
                if self.with_rewards:
                    reward = self.sampler.get_reward(traj_states[-1])
                env.action(action)

                traj_actions.append(action)
                traj_states.append(env.get_symbolic_state())

                if self.terminal_func(traj_states[-1]):
                    # print('term', traj_states[-1])
                    break
            trajs.append((traj_states, traj_actions, reward) if self.with_rewards else (traj_states, traj_actions))
        return trajs


class QValueBasedMultistepActionClassifier(OnlineActionClassifier):
    def __init__(self, graph: GraphizedEnv, primitive_actions: List[int], nr_steps: int, action_qvalue_tensors: dict):
        super().__init__()

        self.graph = graph
        self.primitive_actions = primitive_actions
        self.nr_steps = nr_steps
        self.action_qvalue_tensors = action_qvalue_tensors
        self.action_cummulated_likelihood_multistep = None

    def restart(self, state):
        super().restart(state)
        self.action_cummulated_likelihood = {k: 0 for k in self.action_qvalue_tensors}
        self.action_cummulated_likelihood_multistep = {
            k: [-1e9 if i != 0 else 0 for i in range(self.nr_steps)] for k in self.action_qvalue_tensors
        }

    def action(self, next_state):
        edges = self.graph.get_by_index(self.current_state)
        next_state = self.graph.state2index[next_state]
        nA = self.graph.nr_actions

        self.debug = ''
        this_step_likelihood = dict()
        results = self.action_cummulated_likelihood_multistep  # a "reference"
        for action_type, qvalue_tensor in self.action_qvalue_tensors.items():
            this_action = None
            for v, a, _ in edges:
                if v == next_state:
                    this_action = a
            a = this_action

            if this_action is not None:
                old_results = self.action_cummulated_likelihood_multistep[action_type].copy()
                for j in range(self.nr_steps):
                    action_qvalue_tensor = qvalue_tensor[j, self.current_state]
                    boltzman1 = action_qvalue_tensor[a] - torch.logsumexp(action_qvalue_tensor[:nA], dim=-1)
                    results[action_type][j] = old_results[j] + boltzman1
                    self.debug += (
                        f'\na={action_type}, j={j}, boltzman1=' + format(boltzman1) + f', Q={action_qvalue_tensor[:nA]}'
                    )

                for j in range(self.nr_steps):
                    if j != self.nr_steps - 1:
                        action_qvalue_tensor = qvalue_tensor[j, self.current_state]
                        boltzman2 = action_qvalue_tensor[a + nA] - torch.logsumexp(action_qvalue_tensor[:], dim=-1)
                        results[action_type][j + 1] = torch.max(results[action_type][j + 1], old_results[j] + boltzman2)
                        self.debug += (
                            f'\na={action_type}, j={j}, boltzman2='
                            + format(boltzman2)
                            + f', Q={action_qvalue_tensor[nA:]}'
                        )

        self.current_state = next_state
        self.action_cummulated_likelihood = {k: max(v) for k, v in self.action_cummulated_likelihood_multistep.items()}
        self.debug += '; ' + format(self.action_cummulated_likelihood_multistep)

    @property
    def nr_primitive_actions(self):
        return len(self.primitive_actions)

    def prob(self, states, actions):
        results = dict()
        nA = self.nr_primitive_actions
        for action_type, qvalue_tensor in self.action_value_tensors.items():
            results[action_type] = [-1e9 for _ in range(self.nr_steps)]
            results[action_type][0] = 0
            for s, a in zip(states, actions):
                old_results = results.copy()
                for j in range(self.nr_steps):
                    action_qvalue_tensor = qvalue_tensor[j, s]
                    boltzman1 = action_qvalue_tensor[a] - torch.logsumexp(action_qvalue_tensor[:], dim=-1)
                    results[action_type][j] = old_results[j] + boltzman1
                for j in range(self.nr_steps):
                    if j != self.nr_steps - 1:
                        action_qvalue_tensor = qvalue_tensor[j, s]
                        boltzman2 = action_qvalue_tensor[a + nA] - torch.logsumexp(action_qvalue_tensor[:], dim=-1)
                        results[action_type][j + 1] = torch.max(results[action_type][j + 1], old_results[j] + boltzman2)

        return results


class QValueBasedActionClassifier(OnlineActionClassifier):
    def __init__(self, labels, label2state_machines, broadcast_engine=None, ptrajonly=False):
        super().__init__()
        print('QValueBasedActionClassifier,', 'ptrajonly=', ptrajonly)
        self.ptrajonly = ptrajonly
        self.labels = labels
        self.label2state_machines = label2state_machines
        self.broadcast_engine = broadcast_engine
        for s in self.label2state_machines.values():
            s.update_potential()

    def prob(
        self,
        states,
        actions,
        action_set,
        qvalues,
        init_values,
        labels=None,
        record_path=False,
        record_step_value=False,
        device='cuda:0',
    ):
        if labels is None:
            labels = self.labels
        state_machines = [self.label2state_machines[label] for label in labels]
        res = []
        paths = []
        step_values_list = []
        action2index = {a: aid for aid, a in enumerate(action_set)}
        states_tensor = self.broadcast_engine.states2tensor(states).to(device)
        for l, (state_machine, qvalue) in enumerate(zip(state_machines, qvalues)):
            out_degree = max(len(state_machine.adjs[x]) for x in state_machine.nodes)
            topo_seq = state_machine.get_topological_sequence()
            # f = torch.zeros(len(states), len(state_machine.nodes), device=device) - 1e9
            f = [[None for j in range(len(state_machine.nodes))] for i in range(len(states))]
            r = [[None for j in range(len(state_machine.nodes))] for i in range(len(states))]
            step_values = dict()

            for i in range(len(states) - 1, -1, -1):
                q = qvalue[i]
                n_actions = q.size(0)
                assert i == len(actions) or action2index[actions[i]] < n_actions
                prob_q = torch.log_softmax(q, dim=0)
                # Compute prob for actions[i], and tranfer to i+1
                if i + 1 < len(states):
                    action_prob = prob_q[action2index[actions[i]]]
                    for j in range(len(state_machine.nodes)):
                        if f[i + 1][j] is not None:
                            f[i][j] = f[i + 1][j] + action_prob[j]
                            r[i][j] = ('next', round(float(action_prob[j]), 4))
                else:
                    for v in state_machine.ends:
                        f[i][state_machine.node2index[v]] = torch.zeros(1, device=device)[0]
                        r[i][state_machine.node2index[v]] = ('end',)
                # Compute state transfer action
                for u in reversed(topo_seq):
                    for k, (v, label) in enumerate(state_machine.adjs[u]):
                        # Make exactly the path (last transition must be on the last state)
                        if v in state_machine.ends and i + 1 < len(states):
                            continue
                        x, y = state_machine.node2index[u], state_machine.node2index[v]
                        delta = state_machine.potential[y] - state_machine.potential[x]
                        if not self.ptrajonly:
                            step_value = delta * init_values(label, states_tensor[i]) #+ prob_q[n_actions + k, x]
                        else:
                            step_value = prob_q[n_actions + k, x]
                        upd = f[i][y] + step_value
                        if f[i][x] is None or upd > f[i][x]:
                            f[i][x] = upd
                            r[i][x] = (
                                'step',
                                y,
                                round(delta, 4),
                                round(float(step_value), 4),
                            )
                        if record_step_value:
                            if (x, y) not in step_values:
                                step_values[(x, y)] = torch.zeros(len(states))
                            else:
                                step_values[(x, y)][i] = step_value.cpu().item()

            step_values_list.append(step_values)
            sbest = None
            vmax = None
            for u in state_machine.starts:
                x = state_machine.node2index[u]
                note = state_machine.start_note[u]
                delta = state_machine.potential[x]
                sprob = f[0][x] + (delta * init_values(note, states_tensor[0]) if note is not None else 0)
                # vmax = sprob if vmax is None else torch.max(vmax, sprob)
                if vmax is None or sprob.item() > vmax.item():
                    vmax = sprob
                    sbest = x
            res.append(vmax)
            if record_path:
                path = []
                i, j = 0, sbest
                while i < len(states):
                    path.append((i, j, r[i][j]))
                    if r[i][j][0] in ('next', 'end'):
                        i += 1
                    elif r[i][j][0] in ('step',):
                        j = r[i][j][1]
                    else:
                        raise ValueError()
                paths.append(path)

        if record_path:
            if record_step_value:
                return torch.stack(res).view(-1), paths, step_values_list
            return torch.stack(res).view(-1), paths
        return torch.stack(res).view(-1)


class QValueGraphBasedActionClassifier(OnlineActionClassifier):
    def __init__(self, labels, label2state_machines):
        super().__init__()
        self.labels = labels
        self.label2state_machines = label2state_machines
        for s in self.label2state_machines.values():
            s.update_potential()

    def prob(self, states_tensor, qvalues, init_values, labels=None, record_path=False, device='cuda:0'):
        # Note that qvalues is already normalized, so qvalue[l,i] is the equal to log probability.
        if labels is None:
            labels = self.labels
        state_machines = [self.label2state_machines[label] for label in labels]
        res = []
        paths = []
        # states_tensor = torch.Tensor(states).to(qvalues[0].device)
        n = states_tensor.size(0)
        for l, (state_machine, qvalue) in enumerate(zip(state_machines, qvalues)):
            topo_seq = state_machine.get_topological_sequence()
            # f = torch.zeros(len(states), len(state_machine.nodes), device=device) - 1e9
            f = [[None for j in range(len(state_machine.nodes))] for i in range(n)]
            r = [[None for j in range(len(state_machine.nodes))] for i in range(n)]

            for i in range(n - 1, -1, -1):
                if i + 1 < n:
                    action_prob = qvalue[i]
                    for j in range(len(state_machine.nodes)):
                        if f[i + 1][j] is not None:
                            f[i][j] = f[i + 1][j] + action_prob[j]
                            r[i][j] = ('next', round(float(action_prob[j]), 4))
                else:
                    for v in state_machine.ends:
                        f[i][state_machine.node2index[v]] = torch.zeros(1, device=device)[0]
                        r[i][state_machine.node2index[v]] = ('end',)
                # Compute state transfer action
                # print("step", i)
                # print(f[i], q0)
                # f[i] += q0
                for u in reversed(topo_seq):
                    for k, (v, label) in enumerate(state_machine.adjs[u]):
                        # Make exactly the path (last transition must be on the last state)
                        if v in state_machine.ends and i + 1 < n:
                            continue
                        x, y = state_machine.node2index[u], state_machine.node2index[v]
                        delta = state_machine.potential[y] - state_machine.potential[x]
                        upd = f[i][y] + delta * init_values(label, states_tensor[i])
                        # print(label, states_tensor[i], init_values(label, states_tensor[i]))
                        if f[i][x] is None or upd > f[i][x]:
                            f[i][x] = upd
                            r[i][x] = (
                                'step',
                                y,
                                round(delta, 4),
                                round(float(delta * init_values(label, states_tensor[i])), 4),
                            )
                # f[i] -= q0  # compare state transfer with best action
                # print(f[i])
                # input()
            sbest = None
            vmax = None
            for u in state_machine.starts:
                x = state_machine.node2index[u]
                note = state_machine.start_note[u]
                delta = state_machine.potential[x]
                sprob = f[0][x] + (delta * init_values(note, states_tensor[0]) if note is not None else 0)
                # vmax = sprob if vmax is None else torch.max(vmax, sprob)
                if vmax is None or sprob.item() > vmax.item():
                    vmax = sprob
                    sbest = x
            res.append(vmax)
            if record_path:
                path = []
                i, j = 0, sbest
                while i < n:
                    path.append((i, j, r[i][j]))
                    if r[i][j][0] in ('next', 'end'):
                        i += 1
                    elif r[i][j][0] in ('step',):
                        j = r[i][j][1]
                    else:
                        raise ValueError()
                paths.append(path)
        if record_path:
            return torch.stack(res).view(-1), paths
        return torch.stack(res).view(-1)
