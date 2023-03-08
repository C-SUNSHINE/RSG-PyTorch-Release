#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import optim
import torch.nn.functional as F
import random

from hacl.models.rsgs.evaluators.evaluator import Evaluator
from hacl.models.rsgs.encoders.label_encoder import LabelEncoder
from hacl.models.rsgs.dqn import DQN
from hacl.models.rsgs.utils import Transition, Episode, EpisodeReplayMemory, growing_sampler
from hacl.models.rsgs.encoders import StateEncoder
from hacl.models.rsgs.encoders import TrajEncoder
from hacl.p.rsgs.unified_broadcast_engine import UnifiedBroadcastEngine


class DQNEvaluator(Evaluator):
    _EXTRA_DICT_KEY = [
        'env_name',
    ]

    def __init__(
        self,
        env_name='gridworld',
        env_args=None,
        add_labels=None,
        h_dim=256,
        use_lstm=False,
        train_by_classification=True,
        gamma=0.95,
        eps_start=0.9,
        eps_end=0.05,
        n_episodes_per_traj=3,
        max_steps=50,
        step_reward=-0.1,
        final_reward=10,
        update_per_episode=10,
        rho=0.4,
    ):
        super().__init__()
        if env_args is None:
            env_args = dict()
        self.env_args = env_args
        self.env_name = env_name
        self.h_dim = h_dim

        self.use_lstm = use_lstm

        self.train_by_classification = train_by_classification

        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.n_episodes_per_traj = n_episodes_per_traj
        self.max_steps = max_steps
        self.step_reward = step_reward
        self.final_reward = final_reward
        self.update_per_episode = update_per_episode
        self.episode_count = 0
        self.rho = rho

        self.broadcast_env = UnifiedBroadcastEngine(env_name, env_args)
        self.traj_encoder = TrajEncoder(env_name, env_args, require_encoder=False)

        self.action_dim = self.broadcast_env.get_action_dim()

        self.policy_net = DQN(
            in_dim=None,
            out_dim=self.action_dim,
            input_encoder_gen=lambda: StateEncoder(env_name, env_args, s_dim=h_dim),
            label_encoder_gen=lambda: LabelEncoder(add_labels, h_dim=h_dim // 2),
            use_lstm=use_lstm,
        )
        self.target_net = DQN(
            in_dim=None,
            out_dim=self.action_dim,
            input_encoder_gen=lambda: StateEncoder(env_name, env_args, s_dim=h_dim),
            label_encoder_gen=lambda: LabelEncoder(add_labels, h_dim=h_dim // 2),
            use_lstm=use_lstm,
        )

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.dqn_optimizer = None

        self.memory = EpisodeReplayMemory(100000, has_expert=True, expert_capacity=100000)

    def get_training_parameters(self):
        if self.train_by_classification:
            return self.policy_net.parameters()
        else:
            return None

    @property
    def qvalue_based(self):
        return False

    @property
    def online_optimizer(self):
        # return None
        return self.dqn_optimizer

    def extra_state_dict(self):
        return {key: getattr(self, key) for key in self._EXTRA_DICT_KEY}

    def load_extra_state_dict(self, extra_dict):
        for key in self._EXTRA_DICT_KEY:
            setattr(self, key, extra_dict[key])

    def prepare_epoch(self, *args, lr=0.02, **kwargs):
        self.dqn_optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.policy_net.parameters()), lr=lr)

    @classmethod
    def compute_label_scores(cls, net, trajs, lens, packed_states, labels=None, use_lstm=False, **kwargs):
        n = len(trajs)
        device = packed_states.device
        label2embeddings = {label: net.label_encoder([label]) for label in labels}
        qvalues_all = {
            label: net(
                packed_states,
                embeddings=embedding.repeat(packed_states.size(0), 1),
                hidden=net.get_zero_lstm_state(packed_states) if use_lstm else None,
            )
            for label, embedding in label2embeddings.items()
        }
        if use_lstm:
            qvalues_all = {k: v[0] for k, v in qvalues_all.items()}
        action_logprobs_all = {label: F.log_softmax(qvalues_all[label], dim=2) for label in labels}

        ptr = 0
        scores = torch.zeros(n, len(labels), device=device)
        for i in range(len(lens)):
            for j, label in enumerate(labels):
                action_logprobs = action_logprobs_all[label][i, : lens[i] - 1]
                action_tensor = torch.LongTensor(trajs[i][1]).to(device)
                assert action_logprobs.size(0) == action_tensor.size(0)
                policy_logprobs = action_logprobs.gather(dim=1, index=action_tensor.unsqueeze(1)).squeeze(1)
                scores[i, j] = policy_logprobs.sum()
            ptr += lens[i]
        return scores

    def forward(self, trajs, labels=None, training=False, answer_labels=None, progress=None, **kwargs):
        assert self.env_name in ['craftingworld']

        states_list = [self.traj_encoder.traj_to_tensor(traj, **kwargs) for traj in trajs]
        lens = [states.size(0) for states in states_list]
        max_len = max(lens)
        packed_states = torch.stack(
            [F.pad(states, (0, 0, 0, max_len - length)) for states, length in zip(states_list, lens)], dim=0
        )

        if training:
            self.self_training(trajs, answer_labels=answer_labels, progress=progress, **kwargs)

        return self.compute_label_scores(
            self.policy_net, trajs, lens, packed_states, labels=labels, use_lstm=self.use_lstm
        )

        # label2embeddings = {label: self.policy_net.label_encoder([label]) for label in labels}
        # qvalues_all = {
        #     label: self.policy_net(
        #         packed_states, embeddings=embedding.repeat(packed_states.size(0), 1),
        #         hidden=self.policy_net.get_zero_lstm_state(packed_states) if self.use_lstm else None
        #     ) for label, embedding in label2embeddings.items()
        # }
        # if self.use_lstm:
        #     qvalues_all = {k: v[0] for k, v in qvalues_all.items()}
        # action_logprobs_all = {label: F.log_softmax(qvalues_all[label], dim=2) for label in labels}
        #
        # ptr = 0
        # scores = torch.zeros(n, len(labels), device=self.device)
        # for i in range(len(lens)):
        #     for j, label in enumerate(labels):
        #         action_logprobs = action_logprobs_all[label][i, :lens[i] - 1]
        #         action_tensor = torch.LongTensor(trajs[i][1]).to(self.device)
        #         assert action_logprobs.size(0) == action_tensor.size(0)
        #         policy_logprobs = action_logprobs.gather(dim=1, index=action_tensor.unsqueeze(1)).squeeze(1)
        #         scores[i, j] = policy_logprobs.sum()
        #     ptr += lens[i]
        # return scores

    def train(self, mode=True):
        super(DQNEvaluator, self).train(mode=mode)

    def read_expert_traj(self, states_list, actions_list, label_list):
        for states, actions, label in zip(states_list, actions_list, label_list):
            rewards = torch.zeros_like(actions, dtype=torch.float) + self.step_reward
            rewards[-1] = self.final_reward
            self.memory.push_expert(Episode(states[:-1], actions, rewards, label))

    def sample_start_state(self):
        transition = self.memory.sample_transitions(1, 0)[0]
        state, label = transition.state, transition.label
        return state, label

    @classmethod
    def dqn_sample_action(cls, net, state, label, eps, hidden=None, use_lstm=False):
        with torch.no_grad():
            if use_lstm:
                output, hidden = net(state.unsqueeze(0).unsqueeze(0), labels=[label], hidden=hidden)
            else:
                output = net(state.unsqueeze(0).unsqueeze(0), labels=[label])

        if random.random() > eps:
            action = int(output.max(2)[1].view(1)[0])
        else:
            action = random.randint(0, output.size(2) - 1)

        if use_lstm:
            return action, hidden
        else:
            return action

    def sample_action(self, state, label, eps, hidden=None):
        return self.dqn_sample_action(self.policy_net, state, label, eps, hidden=hidden, use_lstm=self.use_lstm)

    def perform_action(self, state, action, **kwargs):
        new, valid = self.broadcast_env.action(state.unsqueeze(0), action, inplace=False)
        return new.squeeze(0), valid.item()

    def optimize_dqn(self, batch_size=128, rho=None, minimum_batch_size=32):
        assert not self.use_lstm
        if rho is None:
            rho = self.rho
        transitions = growing_sampler(self.memory.sample_transitions, minimum_batch_size, batch_size, rho=rho)
        if transitions is None:
            return

        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool
        )
        non_final_next_states = torch.stack([s for s in batch.next_state if s is not None], dim=0)
        state_batch = torch.stack(batch.state, dim=0)
        action_batch = torch.LongTensor(batch.action).to(self.device)
        reward_batch = torch.Tensor(batch.reward).to(self.device)
        label_batch = batch.label
        non_final_label_batch = [label for s, label in zip(batch.next_state, label_batch) if s is not None]

        state_action_values = (
            self.policy_net(state_batch.unsqueeze(1), labels=label_batch)
            .squeeze(1)
            .gather(1, action_batch.unsqueeze(1))
        )

        next_state_values = torch.zeros(batch_size, device=self.device)
        # print(non_final_next_states.size(), label_batch_embeddings[non_final_mask].size())
        next_state_values[non_final_mask] = (
            self.target_net(non_final_next_states.unsqueeze(1), labels=non_final_label_batch)
            .squeeze(1)
            .max(1)[0]
            .detach()
        )

        # print(next_state_values.size(), reward_batch.size())
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        self._average_qvalue = state_action_values.mean().detach().cpu().item()

        # Optimize the model
        self.dqn_optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            if param.requires_grad:
                param.grad.data.clamp_(-1, 1)
        self.dqn_optimizer.step()

    def optimize_lstm_dqn(self, batch_size=128, rho=0.5, minimum_batch_size=16):
        assert self.use_lstm
        episodes = growing_sampler(self.memory.sample_episodes, minimum_batch_size, batch_size, rho=rho)
        if episodes is None:
            return

        lens = [episode.states.size(0) for episode in episodes]
        max_len = max(lens)

        packed_states = torch.stack(
            [F.pad(episode.states, (0, 0, 0, max_len - length)) for episode, length in zip(episodes, lens)], dim=0
        )
        packed_actions = torch.stack(
            [F.pad(episode.actions, (0, max_len - length)) for episode, length in zip(episodes, lens)], dim=0
        )
        packed_rewards = torch.stack(
            [F.pad(episode.rewards, (0, max_len - length)) for episode, length in zip(episodes, lens)], dim=0
        )
        labels = [episode.label for episode in episodes]
        mask = torch.stack([torch.arange(max_len, device=packed_states.device).lt(length) for length in lens], dim=0)

        state_action_values = (
            self.policy_net(packed_states, labels=labels, hidden=self.policy_net.get_zero_lstm_state(packed_states))[0]
            .gather(2, packed_actions.unsqueeze(2))
            .squeeze(2)
        )

        next_state_values = self.target_net(
            packed_states, labels=labels, hidden=self.policy_net.get_zero_lstm_state(packed_states)
        )[0].max(2)[0]
        next_state_values[~mask] = 0
        next_state_values = torch.cat(
            [next_state_values[:, 1:], torch.zeros_like(next_state_values)[:, :1]], dim=1
        ).detach()
        expected_state_action_values = next_state_values * self.gamma + packed_rewards

        loss = F.smooth_l1_loss(expected_state_action_values[mask], state_action_values[mask])
        self._average_qvalue = state_action_values[mask].mean().detach().cpu().item()
        # Optimize the model
        self.dqn_optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            if param.requires_grad:
                param.grad.data.clamp_(-1, 1)
        self.dqn_optimizer.step()

    def self_training(self, trajs, answer_labels, progress=0, **kwargs):
        n = len(trajs)
        n_episodes = self.n_episodes_per_traj * n
        max_steps = self.max_steps
        states_list = [self.traj_encoder.traj_to_tensor(traj, **kwargs) for traj in trajs]
        actions_list = [torch.LongTensor(traj[1]).to(self.device) for traj in trajs]

        self.read_expert_traj(states_list, actions_list, answer_labels)

        eps = self.eps_start + progress * (self.eps_end - self.eps_start)

        for i_episode in range(1, n_episodes + 1):
            if (self.episode_count + 1) % 250 == 0:
                print('episode', self.episode_count + 1)
                if hasattr(self, '_average_qvalue'):
                    print('_average_qvalue =', self._average_qvalue)
            # Get start state
            state, label = self.sample_start_state()
            if self.use_lstm:
                hidden = self.policy_net.get_zero_lstm_state(state.unsqueeze(0).unsqueeze(0))
            explore_states = []
            explore_actions = []
            explore_rewards = []
            for t in range(max_steps):
                if self.use_lstm:
                    action, hidden = self.sample_action(state, label, eps, hidden=hidden)
                else:
                    action = self.sample_action(state, label, eps)
                explore_states.append(state)
                explore_actions.append(action)
                explore_rewards.append(self.step_reward)

                next_state, valid = self.perform_action(state, action)
                done = not valid
                state = next_state

                # Optimize the model
                if self.use_lstm:
                    self.optimize_lstm_dqn()
                else:
                    self.optimize_dqn()

                if done:
                    explore_states = torch.stack(explore_states, dim=0)
                    explore_actions = torch.LongTensor(explore_actions).to(self.device)
                    explore_rewards = torch.Tensor(explore_rewards).to(self.device)
                    self.memory.push(Episode(explore_states, explore_actions, explore_rewards, label))
                    break

            self.episode_count += 1
            if i_episode == n_episodes or self.episode_count % self.update_per_episode == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
