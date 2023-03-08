#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn

from hacl.algorithms.astar.utils import transfer


# def _set_worker_device_name_from_queue(q):
#     global worker_device
#     worker_device = q.get()
#     # print('initialized:', worker_device)


class GraphAstarModule(nn.Module):
    def __init__(self, env, estimator, state_machine, init_value, s_dim=2, search_unique=False, gamma=0.98):
        """
        :param env: broadcast game env
        :param estimator: value estimators, a dict from each node t of state machine to estimate V(s,t|L)
        :param state_machine: state_machine describing L
        :param init_value: a network from (primitive_task_name, states) to the init_value of states of primitive task
        """
        super().__init__()
        self.env = env
        self.estimator = estimator
        self.state_machine = state_machine
        self.topo_seq = self.state_machine.get_topological_sequence()
        self.init_value = init_value
        self.state_machine.update_potential()
        self.s_dim = s_dim
        self.gamma = gamma
        self.search_unique = search_unique

    def transfer(self, states, f, reverse=False, inplace=True):
        return transfer(
            states,
            f,
            self.state_machine,
            self.init_value,
            topo_seq=self.topo_seq,
            reverse=reverse,
            inplace=inplace,
            s_dim=self.s_dim,
        )

    def action(self, states, g, act, cost, inplace=True):
        if not inplace:
            g = g.clone()
        states, valid = self.env.action(states, int(act), inplace=inplace)
        g += cost
        g = self.transfer(states, g, inplace=True)
        return states, valid, g

    def compress_dim(self, x, dim, d):
        s = 1
        for i in range(d):
            s *= x.size(dim + i)
        return x.view(*x.size()[:dim], s, *x.size()[dim + d:])

    def uncompress_dim(self, x, dim, dims):
        s = 1
        for i in range(len(dims)):
            s *= dims[i]
        assert s == x.size(dim)
        return x.view(*x.size()[:dim], *dims, *x.size()[dim + 1:])

    def get_init_visited(self, batch):
        return [set() for i in range(batch)]

    def remove_visited(self, visited, states, valid, dim=0):
        assert len(valid.size()) == 3 and dim == 1
        batch = valid.size(0)
        flatten_states = states.clone().view(*valid.size(), -1).detach().cpu()
        new_visited = [dict() for i in range(batch)]
        old_valid = valid.clone()

        for index in valid.nonzero(as_tuple=False):
            state_tuple = tuple(flatten_states[index[0], index[1], index[2]].tolist())
            if state_tuple in visited[index[0]]:
                valid[index[0], index[1], index[2]] = False
                old_valid[index[0], index[1], index[2]] = False
            else:
                if state_tuple not in new_visited[index[0]]:
                    new_visited[index[0]][state_tuple] = None
                else:
                    valid[index[0], index[1], index[2]] = False

        transition = valid.type(torch.long).clone()
        transition_mask = transition.clone().type(torch.long)
        transition = (self.compress_dim(transition, dim, 2).cumsum(dim=dim) - 1).view_as(transition_mask)
        transition = transition * transition_mask - (1 - transition_mask)

        for index in old_valid.nonzero(as_tuple=False):
            state_tuple = tuple(flatten_states[index[0], index[1], index[2]].tolist())
            if state_tuple in new_visited[index[0]]:
                if new_visited[index[0]][state_tuple] is None:
                    new_visited[index[0]][state_tuple] = transition[index[0], index[1], index[2]]
                    assert transition_mask[index[0], index[1], index[2]]
                else:
                    transition[index[0], index[1], index[2]] = new_visited[index[0]][state_tuple]
                    transition_mask[index[0], index[1], index[2]] = True

        for i in range(batch):
            visited[i].update(new_visited[i].keys())

        return states, valid, transition, transition_mask

    def build_transition(self, states, valid, g, dim=0, inplace=True, visited=None):
        if not inplace:
            states, valid, g = states.clone(), valid.clone(), g.clone()
        device = states.device
        # print(states.size(), valid.size(), g.size(), dim)

        # Compute transition
        if visited is None:
            transition = valid.type(torch.long).clone()
            transition_mask = transition.clone().type(torch.long)
            transition = (self.compress_dim(transition, dim, 2).cumsum(dim=dim) - 1).view_as(transition_mask)
            transition = transition * transition_mask - (1 - transition_mask)
        else:
            states, valid, transition, transition_mask = self.remove_visited(visited, states, valid, dim=dim)

        # Compute new_states, new_g
        valid = self.compress_dim(valid, dim, 2)
        nonzero_count = valid.type(torch.uint8).gt(0).sum(dim).max()
        index = valid.type(torch.uint8).topk(nonzero_count, dim=dim, sorted=False)[1]
        if index.size(0) == 0:
            states, depth = None, None
        else:
            valid = valid.gather(dim=dim, index=index)
            states_index = (
                index.view(-1, *([1] * self.s_dim))
                    .repeat(1, *states.size()[-self.s_dim:])
                    .view(*index.size(), *states.size()[-self.s_dim:])
            )
            states = self.compress_dim(states, dim, 2).gather(dim=dim, index=states_index)
            g_index = index.view(-1, 1).repeat(1, g.size(-1)).view(*index.size(), g.size(-1))
            g = self.compress_dim(g, dim, 2).gather(dim=dim, index=g_index)

        return states, valid, g, transition

    def expand_by_actions(self, states, valid, g, actions, action_cost, dim=1):
        expanded = [self.action(states, g, a, action_cost[a], inplace=False) for a in actions]
        new_states = torch.stack([x[0] for x in expanded], dim=dim)
        new_valid = torch.stack([(x[1] if valid is None else x[1] & valid) for x in expanded], dim=dim)
        new_g = torch.stack([x[2] for x in expanded], dim=dim)
        return new_states, new_valid, new_g

    def keep_best_k_origin(self, states, valid, g, k):
        size = valid.size()
        states = states.view(-1, *states.size()[-2:])
        valid = valid.view(-1)
        g = g.view(-1, g.size(-1))
        indices = valid.type(torch.uint8).nonzero().view(-1)
        valid_states = states.index_select(dim=0, index=indices)
        valid_g = g.index_select(dim=0, index=indices)
        h = self.estimator(valid_states)
        assert valid_g.size() == h.size()
        f = (valid_g + h * self.gamma).max(dim=1)[0]
        topk_values, topk_indices = f.topk(k, sorted=False)
        preserved_indices = indices.index_select(0, topk_indices)
        valid_topk = torch.zeros_like(valid.view(-1))
        valid_topk[preserved_indices] = True
        return valid_topk.reshape(size)

    def keep_best_k(self, states, valid, g, k, dim=None):
        if dim is None:
            return self.keep_best_k_origin(states, valid, g, k)
        valid_ext = valid.unsqueeze(-1)
        masked_h = self.estimator(states) * valid_ext.type(torch.float) + (valid_ext.type(torch.float) - 1.0) * 1e9
        masked_g = g * valid_ext.type(torch.float) + (valid_ext.type(torch.float) - 1.0) * 1e9
        f = masked_h * self.gamma + masked_g
        valid_topk = torch.zeros_like(valid)
        for t in range(f.size(-1)):
            siz = f.size()[:-1]
            ft = f.view(-1, f.size(-1))[:, t].view(siz)
            topk_values, topk_indices = ft.topk(k, dim=dim, sorted=False)
            valid_topk = valid_topk.scatter_(
                dim=dim, index=topk_indices, src=torch.ones_like(topk_indices).type(torch.bool)
            )
        return valid_topk

    def astar_single(
        self,
        s_tensor,
        a_tensor,
        actions,
        action_cost,
        brute_depth,
        explore_depth,
        shallow_filter=True,
        max_branches=None,
    ):
        """
        Astar from single state start
        :param s_tensor: symbolic tensor of trajectory
        :param a_tensor: actions on the trajectory
        :param actions: action set
        :param action_cost: action->cost cost should usually be negative
        :param brute_depth: search with no condition
        :param explore_depth: after brute_depth, fix #nodes for each search tree layer while keeping the best solutions
        with estimation from vnets.
        :param shallow_filter: True if return only brute depth values, otherwise all values, or number to specify depth
        :param max_branches: if not None, will limit the number of states for explore
        :return: Q[j, t], state_tensor[:, n_obj + 1, obj_s_dim], value[:, t]; each representing means
        Q(s_-1, actions_j, t at state machine), list of states we visited, value of those states at machine node t
        """
        traj_length = s_tensor.size(0)
        device = s_tensor.device
        cost_tensor = torch.Tensor([action_cost[a] for a in actions]).to(device)
        g_tensor = torch.zeros(traj_length, len(self.state_machine.nodes), device=device) - 1e9
        for x in self.state_machine.starts:
            note = self.state_machine.start_note[x]
            delta = self.state_machine.potential[self.state_machine.node2index[x]]
            if note is None:
                g_tensor[0, self.state_machine.node2index[x]] = delta * 0
            else:
                g_tensor[0, self.state_machine.node2index[x]] = delta * self.init_value(note, s_tensor[0])
        for i in range(traj_length - 1):
            g_tensor[i + 1: i + 2] = self.action(
                s_tensor[i: i + 1], g_tensor[i: i + 1], a_tensor[i], cost_tensor[a_tensor[i]], False
            )[2]

        s_tensor = s_tensor[-1:]
        g_tensor = g_tensor[-1:]

        states_queue = [s_tensor]
        g_queue = [g_tensor]
        transitions_queue = []

        for i in range(brute_depth + explore_depth):
            # print('compute depth', i, states_queue[-1].device)
            states = states_queue[-1]
            g = g_queue[-1]
            new_states, valid, new_g = self.expand_by_actions(states, None, g, actions, action_cost)
            new_g = self.transfer(new_states, new_g, inplace=True)

            if i >= brute_depth:
                n_branches = states.size(0) if max_branches is None else min(max_branches, int(states.size(0)))
                print(n_branches)
                valid &= self.keep_best_k(new_states, valid, new_g, n_branches)

            new_states, new_valid, new_g, transition = self.build_transition(new_states, valid, new_g, inplace=True)
            states_queue.append(new_states)
            g_queue.append(new_g)
            transitions_queue.append(transition)

        values_queue = [None for g in g_queue]
        end_mask = torch.zeros(len(self.state_machine.nodes), device=device)
        for x in self.state_machine.ends:
            end_mask[self.state_machine.node2index[x]] = 1
        for i in range(len(values_queue) - 1, -1, -1):
            states = states_queue[i]
            values = torch.zeros_like(g_queue[i]) + (end_mask.unsqueeze(0) - 1) * 1e9  # only keep completed states
            if i + 1 < len(values_queue):
                nex_v = values_queue[i + 1]
                transition = transitions_queue[i]
                transition_mask = transition.ge(0).type(torch.long)
                eff_v = (
                    nex_v[transition + (1 - transition_mask)] * transition_mask.unsqueeze(-1).type(torch.float)
                    + (transition_mask - 1).unsqueeze(-1).type(torch.float) * 1e9
                    + cost_tensor.view(1, -1, 1)
                )
                values = torch.max(values, eff_v.max(dim=1)[0])
            # if i==0:
            #     print(i, values, states)
            values = self.transfer(states, values, reverse=True, inplace=True)
            values_queue[i] = values
            # if i==0:
            #     print(i, values)

        # Compute Qvalue
        qvalue = torch.zeros(len(actions), len(self.state_machine.nodes), device=device)
        for aid, a in enumerate(actions):
            to = transitions_queue[0][0, aid]
            if to == -1:
                qvalue[aid] = -1e9
            else:
                # print(qvalue[aid].size(),  values_queue[1][to].size())
                qvalue[aid] = values_queue[1][to] + action_cost[a]

        # Get states, values pair
        value_depth_cut = (
            (brute_depth + 1 if shallow_filter else explore_depth + 1)
            if isinstance(shallow_filter, bool)
            else shallow_filter
        )
        states_all = torch.cat(states_queue[:value_depth_cut], dim=0)
        values_all = torch.cat(values_queue[:value_depth_cut], dim=0)

        return qvalue, states_all, values_all

    def get_g_function(self, s_tensor, a_tensor, actions, action_cost):
        length = s_tensor.size(0)
        device = s_tensor.device
        cost_tensor = torch.Tensor([action_cost[a] for a in actions]).to(device)
        g_tensor = torch.zeros(length, len(self.state_machine.nodes), device=device) - 1e9
        start_s_tensors = []
        start_g_tensors = []
        for x in self.state_machine.starts:
            note = self.state_machine.start_note[x]
            delta = self.state_machine.potential[self.state_machine.node2index[x]]
            if note is None:
                g_tensor[0, self.state_machine.node2index[x]] = delta * 0
            else:
                g_tensor[0, self.state_machine.node2index[x]] = delta * self.init_value(note, s_tensor[0])
        for i in range(length - 1):
            g_tensor[i + 1: i + 2] = self.action(
                s_tensor[i: i + 1], g_tensor[i: i + 1], a_tensor[i], cost_tensor[a_tensor[i]], False
            )[2]
        return g_tensor

    def value_iteration_search_tree(self, states_queue, g_queue, transitions_queue, cost_tensor, use_estimate_value=False):
        values_queue = [None for g in g_queue]
        end_mask = torch.zeros(len(self.state_machine.nodes), device=states_queue[0].device)
        for x in self.state_machine.ends:
            end_mask[self.state_machine.node2index[x]] = 1
        nex_values = None
        for i in range(len(values_queue) - 1, -1, -1):
            states = states_queue[i]
            values = torch.zeros_like(g_queue[i]) + (end_mask.view(1, 1, -1) - 1) * 1e9
            if i + 1 < len(values_queue):
                if use_estimate_value:
                    nex_values = nex_values * self.gamma
                transition = transitions_queue[i]
                transition_mask = transition.ge(0).type(torch.long)
                transit_index = (
                    (transition + (1 - transition_mask))
                        .view(-1, 1)
                        .repeat(1, *nex_values.size()[2:])
                        .view(*transition.size(), *nex_values.size()[2:])
                )

                transit_value = self.uncompress_dim(
                    nex_values.gather(dim=1, index=self.compress_dim(transit_index, 1, 2)),
                    dim=1,
                    dims=transition.size()[1:3],
                )
                eff_v = (
                    transit_value * transition_mask.unsqueeze(-1).type(torch.float)
                    + (transition_mask.unsqueeze(-1) - 1).type(torch.float) * 1e9
                    + cost_tensor.view(1, 1, -1, 1)
                )
                values = torch.max(values, eff_v.max(dim=2)[0])

            values = self.transfer(states, values, reverse=True, inplace=True)
            values_queue[i] = values
            nex_values = values
            if use_estimate_value:
                nex_values = torch.max(nex_values, self.estimator(states))
        return values_queue

    def astar_parallel(
        self,
        s_tensors,
        a_tensors,
        lengths,
        actions,
        action_cost,
        brute_depth,
        explore_depth,
        shallow_filter=True,
        cached_g_tensors=None,
        max_branches=None,
        use_estimate_value=False,
    ):
        """
        Astar from single state start
        :param s_tensors: symbolic tensor of trajectory, a batch
        :param a_tensors: actions on the trajectory, a batch
        :param lengths: lengths of trajectory, a batch
        :param actions: action set
        :param action_cost: action->cost cost should usually be negative
        :param brute_depth: search with no condition
        :param explore_depth: after brute_depth, fix #nodes for each search tree layer while keeping the best solutions
        with estimation from vnets.
        :param shallow_filter: True if return only brute depth values, otherwise all values, or number to specify depth
        :param max_branches: if not None, will limit the number of states for exploration
        :return: Q[j, t], state_tensor[:, n_obj + 1, obj_s_dim], value[:, t]; each representing means
        Q(s_-1, actions_j, t at state machine), list of states we visited, value of those states at machine node t
        """
        device = s_tensors.device
        self.env = self.env.to(device)

        batch = s_tensors.size(0)
        cost_tensor = torch.Tensor([action_cost[a] for a in actions]).to(device)
        start_s_tensors = []
        start_g_tensors = []
        for k in range(batch):
            start_s_tensors.append(s_tensors[k, lengths[k] - 1])
            start_g_tensors.append(cached_g_tensors[k, lengths[k] - 1])

        start_s_tensors = torch.stack(start_s_tensors, dim=0).unsqueeze(1)
        start_g_tensors = torch.stack(start_g_tensors, dim=0).unsqueeze(1)
        assert start_s_tensors.size(0) == batch and start_s_tensors.size(1) == 1
        assert (
            start_g_tensors.size(0) == batch
            and start_g_tensors.size(1) == 1
            and start_g_tensors.size(2) == len(self.state_machine.nodes)
        )

        states_queue = [start_s_tensors]
        g_queue = [start_g_tensors]
        valid_queue = [torch.ones(*start_s_tensors.size()[:2], dtype=torch.bool, device=device)]
        transitions_queue = []

        visited = self.get_init_visited(batch) if self.search_unique else None
        for i in range(brute_depth + explore_depth):
            # print('depth', i)
            states = states_queue[-1]
            valid = valid_queue[-1]
            g = g_queue[-1]
            new_states, new_valid, new_g = self.expand_by_actions(states, valid, g, actions, action_cost, dim=2)

            if i >= brute_depth:
                n_states = states.size(1)
                best_k = max(10, n_states // len(self.state_machine.nodes))
                if max_branches is not None:
                    best_k = min(int(best_k), max_branches)
                new_states = self.compress_dim(new_states, 1, 2)
                new_valid = self.compress_dim(new_valid, 1, 2)
                new_g = self.compress_dim(new_g, 1, 2)
                new_valid &= self.keep_best_k(new_states, new_valid, new_g, best_k, dim=1)
                new_states = self.uncompress_dim(new_states, 1, (n_states, len(actions)))
                new_valid = self.uncompress_dim(new_valid, 1, (n_states, len(actions)))
                new_g = self.uncompress_dim(new_g, 1, (n_states, len(actions)))

            new_states, new_valid, new_g, transition = self.build_transition(
                new_states, new_valid, new_g, dim=1, inplace=True, visited=visited if i < brute_depth else None
            )
            if min(new_valid.size()) == 0 or new_valid.max() == 0:
                break
            states_queue.append(new_states)
            g_queue.append(new_g)
            valid_queue.append(new_valid)
            transitions_queue.append(transition)

        # Value iteration on search tree
        values_queue = self.value_iteration_search_tree(
            states_queue,
            g_queue,
            transitions_queue,
            cost_tensor,
            use_estimate_value=False
        )
        if use_estimate_value:
            estimate_values_queue = self.value_iteration_search_tree(
                states_queue,
                g_queue,
                transitions_queue,
                cost_tensor,
                use_estimate_value=True
            )
        else:
            estimate_values_queue = None

        # Compute Qvalue
        qvalue = torch.zeros(batch, len(actions), len(self.state_machine.nodes), device=device)
        for k in range(batch):
            for aid, a in enumerate(actions):
                to = transitions_queue[0][k, 0, aid]
                if to == -1:
                    qvalue[k, aid] = -1e9
                else:
                    qvalue[k, aid] = values_queue[1][k, to] + action_cost[a]

        # Get states, values pair
        value_depth_cut = (
            (brute_depth + 1 if shallow_filter else explore_depth + 1)
            if isinstance(shallow_filter, bool)
            else shallow_filter
        )
        states_all = []
        values_all = []
        for d in range(value_depth_cut):
            states_set = states_queue[d]
            values_set = values_queue[d] if not use_estimate_value else estimate_values_queue[d]
            valid_set = valid_queue[d]
            index = valid_set.type(torch.long).view(-1).nonzero().view(-1)
            states_all.append(states_set.view(-1, *states_set.size()[2:])[index])
            values_all.append(values_set.view(-1, *values_set.size()[2:])[index])
        states_all = torch.cat(states_all, dim=0)
        values_all = torch.cat(values_all, dim=0)

        return qvalue, states_all, values_all

    def full_traj_value_iteration(self, s_tensors, a_tensors, lengths, actions, action_cost, estimate_end_state=False):
        batch = s_tensors.size(0)
        cost_tensor = torch.Tensor([action_cost[a] for a in actions]).to(s_tensors.device)
        end_mask = torch.zeros(len(self.state_machine.nodes), device=s_tensors.device)
        for x in self.state_machine.ends:
            end_mask[self.state_machine.node2index[x]] = 1
        # Value iteration on given trajectories.
        max_length = s_tensors.size(1)
        mask = torch.stack(
            [torch.arange(max_length, device=s_tensors.device).lt(length) for length in lengths], dim=0
        ).type(torch.long)
        traj_values_queue = [None for i in range(max_length)]
        next_values = None
        for i in range(max_length - 1, -1, -1):
            states = s_tensors[:, i]
            values = (
                torch.zeros(batch, len(self.state_machine.nodes), device=s_tensors.device)
                + (end_mask.unsqueeze(0) - 1) * 1e9
            )

            if i + 1 < max_length:
                next_values = next_values + cost_tensor[a_tensors[:, i]].unsqueeze(1)
                values = torch.max(values, next_values)
            # print(i, values, states, self.init_value('eff_A', states), self.state_machine)
            values = self.transfer(states, values, reverse=True, inplace=True)
            # print(values)
            values = values * mask[:, i].unsqueeze(1) + (mask[:, i].unsqueeze(1) - 1) * 1e9
            traj_values_queue[i] = values
            next_values = torch.max(values, self.estimator(states) * mask[:, i].unsqueeze(1) + (mask[:, i].unsqueeze(1) - 1) * 1e9) if estimate_end_state else values

        # Get traj states, traj values pair
        traj_values_queue = torch.stack(traj_values_queue, dim=1)
        traj_states_all, traj_values_all = [], []
        for k in range(batch):
            traj_states_all.append(s_tensors[k, : lengths[k]])
            traj_values_all.append(traj_values_queue[k, : lengths[k]])

        # print(traj_states_all[0], traj_values_all[0][:, 2])
        # input()
        # traj_states_all = torch.cat(traj_states_all, dim=0)
        # traj_values_all = torch.cat(traj_values_all, dim=0)
        return traj_states_all, traj_values_all

    def forward(self, traj_states_batch, traj_actions_batch, traj_length_batch, *args, **kwargs):
        """
        Same with evaluators, except for handling multiple starts.
        """
        parallel_astar = True
        if parallel_astar:
            qvalue, states_all, values_all = self.astar_parallel(
                traj_states_batch, traj_actions_batch, traj_length_batch, *args, **kwargs
            )
            return qvalue, states_all, values_all
        else:
            # print(traj_length_batch.size(0))
            q_value_list, states_all_list, values_all_list = [], [], []
            for k in range(traj_length_batch.size(0)):
                traj_length = traj_length_batch[k]
                traj_states = traj_states_batch[k][:traj_length]
                traj_actions = traj_actions_batch[k][: traj_length - 1]
                qvalue, states_all, values_all = self.astar_single(traj_states, traj_actions, *args, **kwargs)
                q_value_list.append(qvalue)
                states_all_list.append(states_all)
                values_all_list.append(values_all)
            return torch.stack(q_value_list, dim=0), torch.cat(states_all_list), torch.cat(values_all_list)

    def expand_at_dim(self, x, dim, siz):
        size = list(x.size())
        size[dim] = siz
        y = torch.zeros(*size, dtype=x.dtype, device=x.device)
        return torch.cat((x, y), dim=dim)

    def iterative_astar(
        self,
        trajs,
        actions,
        action_cost,
        n_iters=5,
        n_epochs=3,
        brute_depth=4,
        explore_depth=30,
        shallow_filter=True,
        flatten_tree=True,
        max_branches=None,
        tune_by_true_traj=False,
        tune_by_value_estimation=False,
        true_traj_indicators=None,
        label=None,
        progress=None,
        device='cpu',
    ):
        """
        :param trajs: trajectories
        :param actions: action set
        :param action_cost: action->cost cost should usually be negative
        :param n_iters: number of iterations for search
        :param n_epochs: number of epochs of training vnet after each iteration.
        :param brute_depth: search with no condition
        :param explore_depth: after brute_depth, fix #nodes for each search tree layer while keeping the best solutions
        with estimation from vnets.
        :param shallow_filter: same as it in evaluators
        :param max_branches: if not None, will limit @ states for exploration
        :return: final results List of Q[i, j, t] indicating Q(S_i, action_j, t) for each traj
        """
        # print('111')
        action2index = {a: i for i, a in enumerate(actions)}
        self.env.to(device)

        traj_state_tensors = []
        traj_action_tensors = []
        g_function_tensors = []
        for traj_id in range(len(trajs)):
            traj_state_tensors.append(self.env.states2tensor(trajs[traj_id][0]).to(device))
            traj_action_tensors.append(torch.LongTensor(trajs[traj_id][1]).to(device))
            g_function_tensors.append(
                self.get_g_function(traj_state_tensors[-1], traj_action_tensors[-1], actions, action_cost)
            )
        # for i, state in enumerate(trajs[0][0]):
        #     print(state)
        #     for act in ['pre_+r', 'eff_+r', 'pre_+g', 'eff_+g', 'pre_=R', 'eff_=R']:
        #         print(act, self.init_value(act, traj_state_tensors[0][i:i+1]).item())
        # input()
        for it in range(n_iters):
            print('running iteration %d/%d' % (it + 1, n_iters))
            # print('prepare')
            qvalues_tensors = [[] for traj in trajs]
            state_tensors = []
            value_tensors = []

            # Make inputs to parallel
            traj_id_batch = []
            traj_states_batch = []
            traj_actions_batch = []
            traj_length_batch = []
            g_function_batch = []
            full_traj_states_batch, full_traj_actions_batch, full_traj_length_batch = [], [], []
            for traj_id, traj in enumerate(trajs):
                full_traj_states_batch.append(traj_state_tensors[traj_id])
                full_traj_actions_batch.append(traj_action_tensors[traj_id])
                full_traj_length_batch.append(len(traj[0]))
                for i in range(len(traj[0])):
                    traj_id_batch.append(traj_id)
                    traj_states_batch.append(traj_state_tensors[traj_id])
                    traj_actions_batch.append(traj_action_tensors[traj_id])
                    traj_length_batch.append(i + 1)
                    g_function_batch.append(g_function_tensors[traj_id])
            max_traj_length = max(traj_length_batch)
            for i in range(len(traj_length_batch)):
                traj_states_batch[i] = self.expand_at_dim(
                    traj_states_batch[i], 0, max_traj_length - traj_states_batch[i].size(0)
                )
                traj_actions_batch[i] = self.expand_at_dim(
                    traj_actions_batch[i], 0, max_traj_length - traj_actions_batch[i].size(0)
                )
                g_function_batch[i] = self.expand_at_dim(
                    g_function_batch[i], 0, max_traj_length - g_function_batch[i].size(0)
                )
            for i in range(len(full_traj_length_batch)):
                full_traj_states_batch[i] = self.expand_at_dim(
                    full_traj_states_batch[i], 0, max_traj_length - full_traj_states_batch[i].size(0)
                )
                full_traj_actions_batch[i] = self.expand_at_dim(
                    full_traj_actions_batch[i], 0, max_traj_length - full_traj_actions_batch[i].size(0)
                )

            traj_states_batch = torch.stack(traj_states_batch, dim=0)
            traj_actions_batch = torch.stack(traj_actions_batch, dim=0)
            traj_length_batch = torch.LongTensor(traj_length_batch).to(device)
            g_function_batch = torch.stack(g_function_batch, dim=0)

            full_traj_states_batch = torch.stack(full_traj_states_batch, dim=0)
            full_traj_actions_batch = torch.stack(full_traj_actions_batch, dim=0)
            full_traj_length_batch = torch.LongTensor(full_traj_length_batch).to(device)

            # Parallel apply
            # print('start_parallel_apply')
            # print('prarllel' if torch.cuda.is_available() else 'not parallel')
            parallel_model = nn.parallel.DataParallel(self) if torch.cuda.is_available() else self
            results = parallel_model(
                traj_states_batch,
                traj_actions_batch,
                traj_length_batch,
                actions,
                action_cost,
                brute_depth,
                explore_depth,
                shallow_filter=shallow_filter,
                cached_g_tensors=g_function_batch,
                max_branches=max_branches,
                use_estimate_value=tune_by_value_estimation,
            )
            # print('end_parallel_apply')

            # print('222')

            # Aggregate results
            for i, traj_id in enumerate(traj_id_batch):
                qvalues_tensors[traj_id].append(results[0][i])
            state_tensors = results[1]
            value_tensors = results[2]
            traj_states = []
            traj_values = []
            for traj_id in range(len(trajs)):
                qvalue_tensor = torch.stack(qvalues_tensors[traj_id], dim=0)
                if not flatten_tree:
                    for i, (s, a) in reversed(list(enumerate(zip(*trajs[traj_id])))[:-1]):
                        qvalue_tensor[i, action2index[a]] = qvalue_tensor[i + 1].max(dim=0)[0] + action_cost[a]
                qvalues_tensors[traj_id] = qvalue_tensor
                for i in range(len(trajs[traj_id][0])):
                    traj_states.append(traj_state_tensors[traj_id][i])
                    traj_values.append(qvalue_tensor[i].max(dim=0)[0])
            # Value Iteration on given trajectories if we need to tune on it as the true path
            if tune_by_true_traj and any(true_traj_indicators):
                # print(333)
                true_traj_states, true_traj_values = self.full_traj_value_iteration(
                    full_traj_states_batch,
                    full_traj_actions_batch,
                    full_traj_length_batch,
                    actions,
                    action_cost,
                    estimate_end_state=False,
                )
                # print(444)
                true_traj_states = torch.cat([states.detach() for i, states in enumerate(true_traj_states) if true_traj_indicators[i]], dim=0).detach()
                true_traj_values = torch.cat([values.detach() for i, values in enumerate(true_traj_values) if true_traj_indicators[i]], dim=0).detach()
                # if progress > 0:
                #     print(true_traj_indicators)
                #     print(true_traj_states.size(0))
                #     for i in range(true_traj_states.size(0)):
                #         print(label, self.env.tensor2states(true_traj_states[i:i + 1])[0], true_traj_values[i].tolist())
                #         print(self.init_value('eff_A', true_traj_states[i:i + 1]))
                #     input()
                torch.set_num_threads(28)
                for j in range(n_epochs):
                    self.estimator.tune(true_traj_states, true_traj_values)
            self.estimator.zero_grad()
            # print(555)
            # if tune_by_value_estimation:
            #     approx_traj_states, approx_traj_values = self.full_traj_value_iteration(
            #         full_traj_states_batch,
            #         full_traj_actions_batch,
            #         full_traj_length_batch,
            #         actions,
            #         action_cost,
            #         estimate_end_state=True,
            #     )
            #     true_traj_states = torch.cat(true_traj_states, dim=0).detach()
            #     true_traj_values = torch.cat(true_traj_values, dim=0).detach()
            #     for j in range(n_epochs):
            #         self.estimator.tune(approx_traj_states, approx_traj_values)
            # Tune estimator
            if tune_by_value_estimation:
                if len(traj_states) > 0 and False:
                    traj_states = torch.stack(traj_states, dim=0)
                    traj_values = torch.stack(traj_values, dim=0)
                    tune_input = torch.cat([state_tensors, traj_states], dim=0)
                    tune_output = torch.cat([value_tensors, traj_values], dim=0)
                else:
                    tune_input = state_tensors
                    tune_output = value_tensors
                if self.estimator.training:
                    for j in range(n_epochs):
                        self.estimator.tune(tune_input.to(device).detach(), tune_output.to(device).detach())
            if it + 1 == n_iters:
                return qvalues_tensors
