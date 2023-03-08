#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
from typing import Optional, List

import jacinle
import jacinle.random as jacrandom
import numpy as np
import torch
from torch.utils.data.dataset import Dataset


class LiftedSkillTransitionEnvConfiguration(object):
    nr_objects: int


class LiftedSkillTransitionEnvState(object):
    def clone(self) -> 'LiftedSkillTransitionEnvState':
        return copy.deepcopy(self)

    def as_tuple(self) -> tuple:
        return tuple(*self)

    def vectorize(self) -> np.ndarray:
        return np.stack(self.as_tuple(), axis=-1)


class LiftedSkillTransitionEnv(object):
    def __init__(self, config: Optional[LiftedSkillTransitionEnvConfiguration] = None):
        self.config = config
        if self.config is None:
            self.config = self._make_default_config()

    def _make_default_config(self) -> LiftedSkillTransitionEnvConfiguration:
        raise NotImplementedError()

    def all_valid_states(self) -> List[LiftedSkillTransitionEnvState]:
        raise NotImplementedError()

    def skill_transition(self, state: LiftedSkillTransitionEnvState, skill_name: str, skill_object_id: int) -> (bool, LiftedSkillTransitionEnvState):
        raise NotImplementedError()

    def gen_all_actions(self, actions):
        for action_id, action_obj in enumerate(actions):
            for target_obj_id in range(self.config.nr_objects):
                yield action_id, action_obj, target_obj_id

    def build_graph_data(self, actions):
        all_states = list(self.all_valid_states())
        all_states_tensor = torch.tensor([x.vectorize() for x in all_states], dtype=torch.float32)
        state2id = {v.as_tuple(): idx for idx, v in enumerate(all_states)}

        feasible_actions = list()
        feasible_action_transitions = list()

        for state in all_states:
            feasible_actions.append(list())
            feasible_action_transitions.append(list())
            for action_id, action_obj, target_obj_id in self.gen_all_actions(actions):
                success, next_state = self.skill_transition(state, action_obj, target_obj_id)
                if success:
                    feasible_actions[-1].append((action_id, target_obj_id))
                    feasible_action_transitions[-1].append(state2id[next_state.as_tuple()])

        return all_states, all_states_tensor, feasible_actions, feasible_action_transitions

    def collect_success_data(self, actions, verbose_stat=True):
        all_states = list(self.all_valid_states())
        all_states_tensor = torch.tensor([x.vectorize() for x in all_states], dtype=torch.float32)
        all_success_tensor = list()
        all_next_states_tensor = list()
        all_next_success_tensor = list()

        for state in all_states:
            all_success_tensor.append(list())
            all_next_states_tensor.append(list())
            all_next_success_tensor.append(list())
            for _, action_obj, target_obj_id in self.gen_all_actions(actions):
                success, next_state = self.skill_transition(state, action_obj, target_obj_id)
                all_success_tensor[-1].append(int(success))
                all_next_states_tensor[-1].append(next_state.vectorize())

                all_next_success_tensor[-1].append(list())
                for _, action_obj2, target_obj_id2 in self.gen_all_actions(actions):
                    success2, _ = self.skill_transition(next_state, action_obj2, target_obj_id2)
                    all_next_success_tensor[-1][-1].append(int(success2))

        all_success_tensor = torch.tensor(all_success_tensor, dtype=torch.int64)
        all_next_states_tensor = torch.tensor(all_next_states_tensor, dtype=torch.float32)
        all_next_success_tensor = torch.tensor(all_next_success_tensor, dtype=torch.int64)

        nr_states = all_states_tensor.size(0)
        nr_actions = all_success_tensor.size(1)

        all_states_tensor = all_states_tensor.view((nr_states, -1))
        all_success_tensor = all_success_tensor.view((nr_states, nr_actions))
        all_next_states_tensor = all_next_states_tensor.view((nr_states, nr_actions, -1))
        all_next_success_tensor = all_next_success_tensor.view((nr_states, nr_actions, nr_actions))

        if verbose_stat:
            print('all_states_tensor', all_states_tensor.size())
            print('all_success_tensor', all_success_tensor.size())
            print('all_next_states_tensor', all_next_states_tensor.size())
            print('all_next_success_tensor', all_next_success_tensor.size())

        return all_states_tensor, all_success_tensor, all_next_states_tensor, all_next_success_tensor

    def make_trajectory_dataset(self, actions: list, traj_length: int, epoch_size: int):
        return LiftedSkillTrajectoryDataset(self, actions, traj_length, epoch_size)


class LiftedSkillTrajectoryDataset(Dataset):
    def __init__(self, env: LiftedSkillTransitionEnv, actions: list, traj_length: int, epoch_size: int):
        self.env = env
        self.actions = actions
        self.traj_length = traj_length
        self.epoch_size = epoch_size

        self.grounded_actions = list(self.env.gen_all_actions(actions))
        self.all_states, self.all_states_tensor, self.feasible_actions, self.feasible_action_transitions = self.env.build_graph_data(self.actions)
        self.rng = jacrandom.gen_rng()
        self.state_size = self.all_states_tensor.shape[1]

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, item):
        feed_dict = dict()

        state = self.rng.randint(0, len(self.all_states))
        traj_states = [self.all_states_tensor[state]]
        traj_actions = list()

        for i in range(self.traj_length):
            valid_actions = self.feasible_actions[state]
            feasible_action_id = self.rng.randint(0, len(valid_actions))
            action_id, object_id = valid_actions[feasible_action_id]
            next_state = self.feasible_action_transitions[state][feasible_action_id]

            traj_actions.append((action_id, object_id))
            traj_states.append(self.all_states_tensor[next_state])

            state = next_state

        traj_states = torch.stack(traj_states, dim=0)
        traj_actions = torch.tensor(traj_actions, dtype=torch.int64)

        return {
            'traj_states': traj_states,
            'traj_actions': traj_actions
        }

    def all_one_step_transitions(self):
        traj_states = []
        traj_actions = []
        for this_state_index in range(len(self.all_states)):
            this_state = self.all_states_tensor[this_state_index]
            for (action_id, object_id), next_state_index in zip(self.feasible_actions[this_state_index], self.feasible_action_transitions[this_state_index]):
                next_state = self.all_states_tensor[next_state_index]
                traj_actions.append([(action_id, object_id)])
                traj_states.append(torch.stack([this_state, next_state], dim=0))
        traj_states = torch.stack(traj_states, dim=0)
        traj_actions = torch.tensor(traj_actions, dtype=torch.int64)
        return {
            'traj_states': traj_states,
            'traj_actions': traj_actions
        }

    def make_dataloader(self, batch_size, nr_workers=4):
        from jactorch.data.dataloader import JacDataLoader
        from jactorch.data.collate import VarLengthCollateV2

        collate_guide = {}

        return JacDataLoader(
            self, batch_size,
            shuffle=True, drop_last=True, num_workers=nr_workers, pin_memory=True,
            collate_fn=VarLengthCollateV2(collate_guide)
        )

