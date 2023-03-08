#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn

from hacl.algorithms.astar.astar import GraphAstarModule
from hacl.algorithms.planning.init_value_planner import InitValuePlanner
from hacl.envs.gridworld.crafting_world.broadcast_engine import CraftingWorldBroadcastEngine
from hacl.envs.gridworld.crafting_world.v20210515 import CraftingWorldV20210515
from hacl.models.rsgs.evaluators.evaluator import Evaluator
from hacl.models.rsgs.init_value_models.craftingworld_init_value_net import CraftingWorldInitValueNet
from hacl.models.rsgs.init_value_models.wrapper import InitValueExpressionWrapper
from hacl.models.rsgs.value_models.wrapper import ValueEstimatorStateMachineWrapper
from hacl.models.rsgs.state_machine.builders import (
    get_compact_state_machine_from_label,
    get_incompact_state_machine_from_label,
)
from hacl.models.rsgs.value_models.craftingworld_value_net import CraftingWorldValueNet

class AstarEvaluator(Evaluator):
    _EXTRA_DICT_KEY = []

    def __init__(
        self,
        env_args=None,
        ptrajonly=False,
        use_not_goal=False,
        use_true_init=False,
        action_cost=None,
        state_machine_type='compact',
        setting='gridworld',
        add_labels=None,
        search_unique=False,
        plan=False,
        flatten_tree=True,
    ):
        super().__init__()
        if env_args is None:
            env_args = dict()
        self.env_args = env_args
        self.setting = setting
        if 'craftingworld' in setting:
            self.env_args = CraftingWorldV20210515.complete_env_args(env_args)
            self.broadcast_env = CraftingWorldBroadcastEngine(env_args)
            self.init_values = InitValueExpressionWrapper(
                CraftingWorldInitValueNet if not use_true_init else CraftingWorldInitValueNet, ptrajonly=ptrajonly, use_not_goal=use_not_goal
            )
        else:
            raise ValueError()
        self.use_true_init = use_true_init
        self.state_machine_type = state_machine_type
        self.search_unique = search_unique
        self.flatten_tree = flatten_tree
        self.action_cost = action_cost
        assert action_cost > 0
        self.labels = []
        self.state_machines = dict()
        self.vnets = nn.ModuleDict()
        self.value_net_tune_by_true_traj = False
        self.value_net_tune_by_self = False

        if 'skills' in self.env_args:
            all_labels = add_labels
            for skill in self.env_args['skills']:
                all_labels.add(skill)
        else:
            all_labels = add_labels if add_labels else set()
        for label in all_labels:
            self.add_label(label, exist_ok=True)

    def get_planner(self):
        return InitValuePlanner(env_name=self.setting, env_args=self.env_args, init_value=self.init_values, device=self.device)

    def get_training_parameters(self):
        if self.training and not self.use_true_init:
            return self.init_values.parameters()
        return None

    def prepare_epoch(self, progress=None, lr=None, *args, **kwargs):
        self.value_net_tune_by_true_traj = True
        self.value_net_tune_by_self = True
        for vnet in self.vnets.values():
            vnet.set_optimizer(lr=lr)

    @property
    def qvalue_based(self):
        return True

    def extra_state_dict(self):
        return {key: getattr(self, key) for key in self._EXTRA_DICT_KEY}

    def load_extra_state_dict(self, extra_dict):
        for key in self._EXTRA_DICT_KEY:
            setattr(self, key, extra_dict[key])

    def add_edge_label(self, edge_label):
        self.init_values.add_edge_labels(edge_label)

    def add_label(self, label, state_machine=None, vnet=None, exist_ok=False):
        if label in self.labels:
            if not exist_ok:
                raise ValueError("label already added")
            return
        if state_machine is None:
            if self.state_machine_type == 'incompact':
                state_machine = get_incompact_state_machine_from_label(label)
            elif self.state_machine_type == 'compact':
                state_machine = get_compact_state_machine_from_label(label)
            else:
                raise ValueError()
        if vnet is None:
            if 'craftingworld' in self.setting:
                vnet = ValueEstimatorStateMachineWrapper(state_machine, CraftingWorldValueNet)
        self.labels.append(label)
        self.init_values.add_edge_labels(self.get_edge_label_set_from_state_machines([state_machine]))
        self.state_machines[label] = state_machine
        self.vnets[label] = vnet

    def forward(
        self,
        trajs,
        action_set=None,
        n_iters=2,
        n_epochs=2,
        brute_depth=3,
        explore_depth=20,
        shallow_filter=1,
        max_branches=None,
        map_id=None,
        labels=None,
        answer_labels=None,
        training=False,
        progress=None,
        **kwargs
    ):
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
        if labels is None:
            labels = self.labels
        if 'craftingworld' in self.setting:
            broadcast_env = self.broadcast_env
            s_dim = 1
        else:
            raise ValueError()
        astar_engines = [
            GraphAstarModule(
                broadcast_env,
                self.vnets[label],
                self.state_machines[label],
                self.init_values,
                s_dim=s_dim,
                search_unique=self.search_unique,
            )
            for label in labels
        ]
        action_cost = {a: -self.action_cost for a in action_set}

        action_qvalues = [
            astar_engine.iterative_astar(
                trajs,
                action_set,
                action_cost,
                n_iters=n_iters,
                n_epochs=n_epochs,
                brute_depth=brute_depth,
                explore_depth=explore_depth,
                shallow_filter=shallow_filter,
                flatten_tree=self.flatten_tree,
                max_branches=max_branches,
                tune_by_true_traj=self.value_net_tune_by_true_traj and training,
                tune_by_value_estimation=self.value_net_tune_by_self and training,
                true_traj_indicators=[answer_label == label for answer_label in answer_labels] if training else None,
                label=label,
                progress=progress,
                device=device,
            )
            for astar_engine, label in zip(astar_engines, labels)
        ]
        return action_qvalues

    def plan(self, constraints, labels, action_set=None, plan_search=False, terminate_checkers=None, **kwargs):
        planner = self.get_planner()
        state_machines = [self.state_machines[label] for label in labels]
        trajs = planner.plan(
            constraints, labels, state_machines,
            action_set=action_set,
            action_cost=1e-5,  # self.action_cost,
            plan_search=plan_search,
            terminate_checkers=terminate_checkers,
            **kwargs
        )
        return trajs

    def test_goal_classifier(self, states, skill):
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
        if 'eff_' + skill in self.init_values:
            value = self.init_values('eff_' + skill, self.broadcast_env.states2tensor(states).to(device))
            return value
        return None

    def get_edge_label_set_from_state_machines(self, state_machines):
        res = set()
        for s in state_machines:
            for l in s.get_edge_label_set():
                res.add(l)
            for st in s.starts:
                res.add(s.start_note[st])
        res = list(sorted(list(filter(lambda x: x is not None, res))))
        return res

    def init_values_requires_grad(self, mode):
        for param in self.init_values.parameters():
            param.requires_grad = mode

    def values_requires_grad(self, mode):
        for param in self.vnets.parameters():
            param.requires_grad = mode

    def train(self, mode=True):
        super(AstarEvaluator, self).train(mode=mode)
        self.init_values_requires_grad(mode)
        self.values_requires_grad(mode)

    def set_zero_parameter(self):
        print('set zero parameter')
        for param in self.vnets.parameters():
            param.data[:] = 0
