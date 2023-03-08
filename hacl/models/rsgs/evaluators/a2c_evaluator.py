#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import optim
import torch.nn.functional as F
import random
import math

from hacl.models.rsgs.evaluators.evaluator import Evaluator
from hacl.models.rsgs.encoders.label_encoder import LabelEncoder
from hacl.models.rsgs.a2c import A2C
from hacl.models.rsgs.utils import Episode, EpisodeReplayMemory, growing_sampler
from hacl.models.rsgs.encoders.state_encoder import StateEncoder
from hacl.models.rsgs.encoders.traj_encoder import TrajEncoder
from hacl.p.rsgs.unified_broadcast_engine import UnifiedBroadcastEngine
from hacl.utils.math import gauss_log_prob

class A2CEvaluator(Evaluator):
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
        n_episodes_per_traj=5,
        max_steps=50,
        step_reward=-0.1,
        final_reward=10,
        entropy_beta=1e-4,
        rho=0.4,
    ):
        super().__init__()
        if env_args is None:
            env_args = dict()
        self.env_args = env_args
        self.env_name = env_name

        if env_name in ['toyrobot']:
            self.continuous = True
        else:
            raise ValueError()

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
        self.episode_count = 0
        self.entropy_beta = entropy_beta
        self.rho = rho

        self.broadcast_env = UnifiedBroadcastEngine(env_name, env_args)
        self.traj_encoder = TrajEncoder(env_name, env_args, require_encoder=False)

        self.action_dim = self.broadcast_env.get_action_dim()

        self.net = A2C(
            in_dim=None,
            out_dim=self.action_dim,
            input_encoder_gen=lambda: StateEncoder(env_name, env_args, s_dim=h_dim),
            label_encoder_gen=lambda: LabelEncoder(add_labels, h_dim=h_dim // 2),
            use_lstm=use_lstm,
            continuous=self.continuous,
        )

        self.a2c_optimizer = None

        self.memory = EpisodeReplayMemory(100000, has_expert=True, expert_capacity=100000)

    def get_training_parameters(self):
        if self.train_by_classification:
            return self.net.parameters()
        else:
            return None

    @property
    def qvalue_based(self):
        return False

    @property
    def online_optimizer(self):
        # return None
        return self.a2c_optimizer

    def extra_state_dict(self):
        return {key: getattr(self, key) for key in self._EXTRA_DICT_KEY}

    def load_extra_state_dict(self, extra_dict):
        for key in self._EXTRA_DICT_KEY:
            setattr(self, key, extra_dict[key])

    def prepare_epoch(self, *args, lr=0.02, **kwargs):
        self.a2c_optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr=lr)

    @classmethod
    def compute_label_scores(
        cls, net, trajs, lens, packed_states, actions_list, labels=None, use_lstm=False, continuous=False, **kwargs
    ):
        n = len(trajs)
        device = packed_states.device

        label2embeddings = {label: net.label_encoder([label]) for label in labels}

        if not continuous:
            action_dist_all = {
                label: net(
                    packed_states,
                    embeddings=embedding.repeat(packed_states.size(0), 1),
                    hidden=net.get_zero_lstm_state(packed_states) if use_lstm else None,
                )[0]
                for label, embedding in label2embeddings.items()
            }
            action_dist_all = {label: F.log_softmax(action_dist_all[label], dim=2) for label in labels}
        else:
            action_dist_all = {
                label: net(
                    packed_states,
                    embeddings=embedding.repeat(packed_states.size(0), 1),
                    hidden=net.get_zero_lstm_state(packed_states) if use_lstm else None,
                )[0]
                for label, embedding in label2embeddings.items()
            }

        ptr = 0
        scores = torch.zeros(n, len(labels), device=device)
        for i in range(len(lens)):
            for j, label in enumerate(labels):
                if not continuous:
                    action_tensor = actions_list[i]
                    action_dist = action_dist_all[label][i, : lens[i] - 1]
                    assert action_tensor.size(0) == action_dist.size(0)
                    policy_logprobs = action_dist.gather(dim=1, index=action_tensor.unsqueeze(1)).squeeze(1)
                else:
                    action_tensor = actions_list[i]
                    action_mu, action_var = (
                        action_dist_all[label][0][i, : lens[i] - 1],
                        action_dist_all[label][1][i, : lens[i] - 1],
                    )
                    policy_logprobs = cls.compute_logprob(action_mu, action_var, action_tensor).sum(-1)
                scores[i, j] = policy_logprobs.sum()
            ptr += lens[i]
        return scores

    def forward(self, trajs, labels=None, training=False, answer_labels=None, progress=None, **kwargs):
        states_list = [self.traj_encoder.traj_to_tensor(traj, **kwargs) for traj in trajs]
        actions_list = [self.traj_encoder.traj_to_action_tensor(traj, **kwargs) for traj in trajs]
        lens = [states.size(0) for states in states_list]
        max_len = max(lens)
        packed_states = torch.stack(
            [F.pad(states, (0, 0, 0, max_len - length)) for states, length in zip(states_list, lens)], dim=0
        )

        if training:
            self.self_training(trajs, answer_labels=answer_labels, progress=progress, **kwargs)

        return self.compute_label_scores(
            self.net,
            trajs,
            lens,
            packed_states,
            actions_list,
            labels=labels,
            use_lstm=self.use_lstm,
            continuous=self.continuous,
        )

    def train(self, mode=True):
        super(A2CEvaluator, self).train(mode=mode)

    def read_expert_traj(self, states_list, actions_list, label_list):
        for states, actions, label in zip(states_list, actions_list, label_list):
            if not self.continuous:
                rewards = torch.zeros_like(actions, dtype=torch.float) + self.step_reward
                rewards[-1] = self.final_reward
                self.memory.push_expert(Episode(states, actions, rewards, label))
            else:
                rewards = actions.sum(1)
                rewards[-1] += self.final_reward
                self.memory.push_expert(Episode(states, actions, rewards, label))

    @classmethod
    def compute_logprob(cls, mu, var, value):
        return gauss_log_prob(mu, var, value)

    def discount_reward(self, values, rewards, lens, mask=None):
        max_len = values.size(1)
        if mask == None:
            mask = torch.stack([torch.arange(max_len, device=self.device).lt(length) for length in lens], dim=0)
        assert values.size() == rewards.size()
        x = torch.zeros_like(rewards)
        x[mask] = rewards[mask]
        lens_tensor = torch.LongTensor(lens).to(self.device)
        x = x.scatter(1, lens_tensor.unsqueeze(1), values.gather(1, lens_tensor.unsqueeze(1)))

        for t in range(max_len - 2, -1, -1):
            x[:, t] += x[:, t + 1] * self.gamma

        return x.detach()

    @classmethod
    def normal_sampler(cls, mu, var):
        return torch.normal(mu, torch.sqrt(var))

    def sample_start_state(self):
        transition = self.memory.sample_transitions(1, 0)[0]
        state, label = transition.state, transition.label
        return state, label

    @classmethod
    def a2c_sample_action(
        cls, net, state, label, eps, hidden=None, use_lstm=False, continuous=False, broadcast_env=None
    ):
        with torch.no_grad():
            if use_lstm:
                output, value, hidden = net(state.unsqueeze(0).unsqueeze(0), labels=[label], hidden=hidden)
            else:
                output, value = net(state.unsqueeze(0).unsqueeze(0), labels=[label])

        if random.random() > eps:
            if not continuous:
                action = int(output.max(2)[1].view(1)[0])
            else:
                action = cls.normal_sampler(output[0], output[1]).squeeze(0).squeeze(0).detach()
        else:
            action = broadcast_env.sample_action()

        if use_lstm:
            return action, hidden
        else:
            return action

    def sample_action(self, state, label, eps, hidden=None):
        return self.a2c_sample_action(
            self.net,
            state,
            label,
            eps,
            hidden=hidden,
            use_lstm=self.use_lstm,
            continuous=self.continuous,
            broadcast_env=self.broadcast_env,
        )

    def perform_action(self, state, action):
        new, valid = self.broadcast_env.action(state.unsqueeze(0), action, inplace=False)
        return new.squeeze(0), valid.item()

    def optimize_a2c(self, batch_size=128, rho=None, minimum_batch_size=2):
        assert self.use_lstm
        if rho is None:
            rho = self.rho
        episodes = growing_sampler(self.memory.sample_episodes, minimum_batch_size, batch_size, rho=rho)
        if episodes is None:
            return

        lens = [episode.actions.size(0) for episode in episodes]
        max_len = max(lens) + 1

        packed_states = torch.stack(
            [F.pad(episode.states, (0, 0, 0, max_len - length - 1)) for episode, length in zip(episodes, lens)], dim=0
        )
        if not self.continuous:
            packed_actions = torch.stack(
                [F.pad(episode.actions, (0, max_len - length)) for episode, length in zip(episodes, lens)], dim=0
            )
        else:
            packed_actions = torch.stack(
                [F.pad(episode.actions, (0, 0, 0, max_len - length)) for episode, length in zip(episodes, lens)], dim=0
            )

        packed_rewards = torch.stack(
            [F.pad(episode.rewards, (0, max_len - length)) for episode, length in zip(episodes, lens)], dim=0
        )
        labels = [episode.label for episode in episodes]
        mask = torch.stack([torch.arange(max_len, device=packed_states.device).lt(length) for length in lens], dim=0)

        outputs = self.net(
            packed_states, labels=labels, hidden=self.net.get_zero_lstm_state(packed_states) if self.use_lstm else None
        )

        values = outputs[1]
        target_values = self.discount_reward(values, packed_rewards, lens, mask=mask).detach()
        value_loss = F.mse_loss(values[mask], target_values[mask])
        advantage = target_values - values.detach()

        if not self.continuous:
            scores = outputs[0]
            action_logprobs = F.log_softmax(scores, dim=2).gather(2, packed_actions.unsqueeze(2)).squeeze(2)[mask]
            actor_loss = -action_logprobs.mean()
            entropy_loss = 0
        else:
            (mu, var) = outputs[0]
            action_logprobs = advantage[mask] * self.compute_logprob(mu[mask], var[mask], packed_actions[mask]).sum(-1)
            actor_loss = -action_logprobs.mean()
            entropy_loss = self.entropy_beta * (-(torch.log(2 * math.pi * var) + 1) / 2).mean()

        loss = actor_loss + value_loss + entropy_loss
        self._average_value = values[mask].mean().detach().cpu().item()

        # Optimize the model
        self.a2c_optimizer.zero_grad()
        loss.backward()
        for param in self.net.parameters():
            if param.requires_grad:
                param.grad.data.clamp_(-1, 1)
        self.a2c_optimizer.step()

    def self_training(self, trajs, answer_labels, progress=0, **kwargs):
        n = len(trajs)
        n_episodes = self.n_episodes_per_traj * n
        max_steps = self.max_steps
        states_list = [self.traj_encoder.traj_to_tensor(traj, **kwargs) for traj in trajs]
        actions_list = [self.traj_encoder.traj_to_action_tensor(traj, **kwargs) for traj in trajs]

        self.read_expert_traj(states_list, actions_list, answer_labels)

        eps = self.eps_start + progress * (self.eps_end - self.eps_start)

        for i_episode in range(1, n_episodes + 1):
            if (self.episode_count + 1) % 100 == 0:
                print('episode', self.episode_count + 1)
                if hasattr(self, '_average_qvalue'):
                    print('_average_value =', self._average_qvalue)
            # Get start state
            state, label = self.sample_start_state()
            if self.use_lstm:
                hidden = self.net.get_zero_lstm_state(state.unsqueeze(0).unsqueeze(0))
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
                self.optimize_a2c()

                if done:
                    explore_states = torch.stack(explore_states, dim=0)
                    if not self.continuous:
                        explore_actions = torch.LongTensor(explore_actions).to(self.device)
                    else:
                        explore_actions = torch.stack(explore_actions, dim=0).to(self.device)
                    explore_rewards = torch.Tensor(explore_rewards).to(self.device)
                    if explore_actions.size(0) > 1:
                        self.memory.push(Episode(explore_states, explore_actions[:-1], explore_rewards[:-1], label))
                    break

            self.episode_count += 1
