#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import random

import torch
import torch.nn.functional as F
from torch import nn, optim

from hacl.algorithms.planning.rl_planner import RLPlanner
from hacl.p.rsgs.unified_broadcast_engine import UnifiedBroadcastEngine
from hacl.models.rsgs.dqn import DQN
from hacl.models.rsgs.encoders import StateEncoder, TrajEncoder
from hacl.models.rsgs.encoders.label_encoder import LabelEncoder
from hacl.models.rsgs.evaluators.dqn_evaluator import DQNEvaluator
from hacl.models.rsgs.evaluators.evaluator import Evaluator
from hacl.models.rsgs.utils import Episode, EpisodeReplayMemory, growing_sampler


class IRLContEvaluator(Evaluator):
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
        rl_model='dqn',
        gamma=0.5,
        eps_start=0.5,
        eps_end=0.05,
        n_episodes_per_traj=None,
        max_steps=None,
        update_per_episode=50,
        rho=0.5,
        classify_with_reward=False,
    ):
        super().__init__()
        if env_args is None:
            env_args = dict()
        self.env_args = env_args
        self.env_name = env_name

        print("IRLEvaluator: classify_with_reward=", classify_with_reward)

        if env_name in ['craftingworld']:
            self.continuous = False
        elif env_name in ['toyrobot']:
            self.continuous = True
        else:
            raise ValueError('Invalid env_name %s' % env_name)

        self.h_dim = h_dim

        self.use_lstm = use_lstm

        self.train_by_classification = False

        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.rho = rho
        self.n_episodes_per_traj = n_episodes_per_traj if n_episodes_per_traj is not None else 1.0 / 8
        self.max_steps = max_steps if max_steps is not None else (50 if not self.continuous else 30)
        self.update_per_episode = update_per_episode
        self.episode_count = 0
        self.classify_with_reward = classify_with_reward
        self.entropy_beta = 1e-4

        self.broadcast_env = UnifiedBroadcastEngine(env_name, env_args)
        self.traj_encoder = TrajEncoder(env_name, env_args, require_encoder=False)

        self.action_dim = self.broadcast_env.get_action_dim()

        self.rl_model = rl_model
        assert rl_model == 'dqn' and self.continuous

        # self.irl = MaxEntIRL(
        #     in_dim=None,
        #     action_dim=self.action_dim,
        #     input_encoder_gen=lambda: StateEncoder(env_name, env_args, s_dim=h_dim),
        #     label_encoder_gen=lambda: LabelEncoder(add_labels, h_dim=h_dim // 2),
        #     use_lstm=use_lstm,
        #     continuous=self.continuous,
        #     rl_model=rl_model,
        #     reward_activation='linear',
        #     reward_logprob=True,
        # )

        self.value = DQN(
            None,
            1,
            h_dim=h_dim,
            input_encoder_gen=lambda: StateEncoder(env_name, env_args, s_dim=h_dim, playroom_distance=True),
            label_encoder_gen=lambda: LabelEncoder(add_labels, h_dim=h_dim // 2),
            use_lstm=use_lstm,
        )
        self.reward = DQN(
            None,
            1,
            h_dim=h_dim,
            input_encoder_gen=lambda: StateEncoder(env_name, env_args, s_dim=h_dim, playroom_distance=True),
            label_encoder_gen=lambda: LabelEncoder(add_labels, h_dim=h_dim // 2),
            use_lstm=use_lstm,
            playroom_add=True,
            activation='logsigmoid',
        )
        print('DQN,', 'activation=logsigmoid, playroom_distance=True')

        self.reward_net_optimizer = None
        self.rl_net_optimizer = None

        self.memory = EpisodeReplayMemory(100000, has_expert=True, expert_capacity=1000)
        self.seen_labels = []

    def get_planner(self):
        if self.classify_with_reward:
            planner = RLPlanner(env_name=self.env_name, env_args=self.env_args, net=self, is_irlcont=True, gamma=self.gamma)
        else:
            planner = RLPlanner(env_name=self.env_name, env_args=self.env_args, net=self.dqn)
        return planner.to(self.device)

    def get_training_parameters(self):
        if self.train_by_classification:
            return self.reward.parameters()
        else:
            return None

    @property
    def qvalue_based(self):
        return False

    @property
    def online_optimizer(self):
        return self.reward_net_optimizer

    def extra_state_dict(self):
        return {key: getattr(self, key) for key in self._EXTRA_DICT_KEY}

    def load_extra_state_dict(self, extra_dict):
        for key in self._EXTRA_DICT_KEY:
            setattr(self, key, extra_dict[key])

    def prepare_epoch(self, *args, lr=0.02, base_lr=None, progress=None, **kwargs):
        self.reward_net_optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.reward.parameters()), lr=lr, weight_decay=1e-4
        )
        self.rl_net_optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.value.parameters()), lr=lr
        )
        self.train_by_classification = True
        if progress < .3:
            self.gamma = 0.0
        else:
            self.gamma = 0.5

    def forward(self, trajs, labels=None, training=False, answer_labels=None, progress=None, save_dir=None, **kwargs):
        n = len(trajs)
        states_list = [self.traj_encoder.traj_to_tensor(traj, **kwargs) for traj in trajs]
        actions_list = [self.traj_encoder.traj_to_action_tensor(traj, **kwargs) for traj in trajs]
        states_lens = [states.size(0) for states in states_list]
        actions_lens = [actions.size(0) for actions in actions_list]
        max_len = max(states_lens)
        packed_states = torch.stack(
            [F.pad(states, (0, 0, 0, max_len - length)) for states, length in zip(states_list, states_lens)], dim=0
        )

        if training:
            self.self_training(trajs, answer_labels=answer_labels, progress=progress, **kwargs)

        if self.classify_with_reward:
            packed_actions = torch.stack(
                [(F.pad(actions, (0, max_len - length)) if not self.continuous
                  else F.pad(actions, (0, 0, 0, max_len - length))) for actions, length in zip(actions_list, actions_lens)],
                dim=0,
            )
            label_packed_statess = packed_states.repeat(len(labels), 1, 1)
            label_packed_actions = packed_actions.repeat(len(labels), 1) if not self.continuous else packed_actions.repeat(len(labels), 1, 1)
            policy_logprobs = self.compute_policy_logprobs(
                label_packed_statess,
                label_packed_actions,
                actions_lens * len(labels),
                [labels[i // n] for i in range(len(labels) * n)],
                with_mask=False,
                detach_qvalue=False,
            ).view(len(labels), n, max_len)
            scores = []
            for i in range(n):
                scores.append(policy_logprobs[:, i, : actions_lens[i]].sum(1))
            return torch.stack(scores, dim=0)

        else:
            return DQNEvaluator.compute_label_scores(
                self.irl.dqn, trajs, states_lens, packed_states, labels=labels, use_lstm=self.use_lstm
            )

    def train(self, mode=True):
        super(IRLContEvaluator, self).train(mode=mode)

    def plan(self, start_states, labels, action_set=None, **kwargs):
        planner = self.get_planner()
        trajs = planner.plan(
            start_states, labels,
            action_set=action_set,
            action_cost=1e-5,  # self.action_cost,
            **kwargs
        )
        return trajs

    def read_expert_traj(self, states_list, actions_list, label_list):
        for states, actions, label in zip(states_list, actions_list, label_list):
            self.memory.push_expert(Episode(states, actions, None, label))

    def sample_action(self, state, label, eps, hidden_value=None, hidden_reward=None, proposed_action=None):
        assert self.use_lstm
        n = 20
        if random.random() > eps:
            best_action, best_value = None, None
            is_proposed = False
            for i in range(n):
                if i == 0 and proposed_action is not None:
                    action = proposed_action
                else:
                    action = torch.normal(torch.zeros(self.action_dim), torch.ones(self.action_dim)).to(state.device)
                    action = torch.clip(action, -1, 1)
                new, valid = self.perform_action(state, action)
                if not valid:
                    continue
                reward = self.reward(
                    state.unsqueeze(0).unsqueeze(0),
                    [label],
                    hidden=hidden_reward,
                    add=action.view(1, 1, self.action_dim)
                )[0].item()
                value = self.value(
                    torch.cat((state.unsqueeze(0).unsqueeze(0), new.unsqueeze(0).unsqueeze(0)), dim=1),
                    [label],
                    hidden=hidden_value,
                )[0][:, 1, :].item()
                if best_value is None or value + reward > best_value:
                    best_value = value + reward
                    best_action = action
                    is_proposed = (i == 0)
            return best_action, is_proposed
        else:
            for i in range(n):
                action = torch.normal(torch.zeros(self.action_dim), torch.ones(self.action_dim)).to(state.device)
                action = torch.clip(action, -1, 1)
                new, valid = self.perform_action(state, action)
                if valid:
                    return action, False
            return None

    def perform_action(self, state, action):
        new, valid = self.broadcast_env.action(state.unsqueeze(0), action, inplace=False)
        return new.squeeze(0), valid.item()

    def update_dqn(self, batch_size=128, rho=None, minimum_batch_size=16):
        if rho is None:
            rho = self.rho
        is_expert = True
        if is_expert:
            rho = 0
        episodes = growing_sampler(self.memory.sample_episodes, minimum_batch_size, batch_size, rho=rho)
        if episodes is None:
            return

        lens = [episode.actions.size(0) for episode in episodes]
        max_len = max(lens) + 1

        packed_states = torch.stack(
            [F.pad(episode.states, (0, 0, 0, max_len - length - 1)) for episode, length in zip(episodes, lens)], dim=0
        )
        packed_actions = torch.stack(
            [F.pad(episode.actions, (0, 0, 0, max_len - length)) for episode, length in zip(episodes, lens)], dim=0
        )
        labels = [episode.label if is_expert or random.random() < .5 or len(self.seen_labels) == 0 else random.choice(self.seen_labels) for episode in episodes]
        mask = torch.stack([torch.arange(max_len, device=packed_states.device).lt(length) for length in lens], dim=0)

        state_values = self.value(
            packed_states,
            labels=labels,
            hidden=self.value.get_zero_lstm_state(packed_states) if self.use_lstm else None,
        )[0]

        next_state_values = self.value(
            packed_states,
            labels=labels,
            hidden=self.value.get_zero_lstm_state(packed_states) if self.use_lstm else None,
        )[0]
        next_state_values = torch.cat(
            [next_state_values[:, 1:], torch.zeros_like(next_state_values[:, :1])], dim=1
        ).detach()

        state_rewards = self.reward(
            packed_states,
            labels=labels,
            hidden=self.reward.get_zero_lstm_state(packed_states) if self.use_lstm else None,
            add=packed_actions
        )[0]

        expected_state_values = next_state_values * self.gamma + state_rewards
        if not is_expert:
            expected_state_values = torch.max(expected_state_values, state_values)

        loss = F.l1_loss(expected_state_values[mask], state_values[mask])
        self._average_qvalue = state_values[mask].mean().detach().cpu().item()
        # Optimize the model
        self.rl_net_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.value.parameters()), 10)
        self.rl_net_optimizer.step()

    def update_rl_net(self, *args, **kwargs):
        self.update_dqn(*args, **kwargs)

    def compute_values_and_rewards(self, packed_states, packed_actions, labels, detach_qvalue=True):
        max_len = packed_states.size(1)
        step_action_values = []
        step_action_rewards = []
        hidden_value = self.value.get_zero_lstm_state(packed_states[:, :1]) if self.use_lstm else None
        hidden_reward = self.reward.get_zero_lstm_state(packed_states[:, :1]) if self.use_lstm else None
        n = 4
        for t in range(max_len):
            if self.use_lstm:
                _, hidden_value = self.value(packed_states[:, t: t + 1], labels=labels, hidden=hidden_value)
            # actions = torch.normal(
            #     torch.zeros(packed_states.size(0), n, self.action_dim, device=packed_states.device),
            #     torch.ones(packed_states.size(0), n, self.action_dim, device=packed_states.device)
            # )
            # actions = torch.clip(actions, -1, 1)
            # actions[:, 0] = packed_actions[:, t]

            base_action = packed_actions[:, t]
            actions = [
                torch.stack((
                    base_action[:, 0],
                    base_action[:, 1],
                    torch.clip(torch.normal(torch.zeros_like(base_action[:, 2]), torch.ones_like(base_action[:, 2])), -1, 1)
                ), dim=1),
                torch.stack((
                    -base_action[:, 1],
                    base_action[:, 0],
                    torch.clip(torch.normal(torch.zeros_like(base_action[:, 2]), torch.ones_like(base_action[:, 2])), -1, 1)
                ), dim=1),
                torch.stack((
                    base_action[:, 1],
                    -base_action[:, 0],
                    torch.clip(torch.normal(torch.zeros_like(base_action[:, 2]), torch.ones_like(base_action[:, 2])), -1, 1)
                ), dim=1),
                torch.stack((
                    -base_action[:, 0],
                    -base_action[:, 1],
                    torch.clip(torch.normal(torch.zeros_like(base_action[:, 2]), torch.ones_like(base_action[:, 2])), -1, 1)
                ), dim=1)
            ]
            actions = torch.stack(actions, dim=1)
            # print(actions.size())

            action_values = []
            action_rewards = []
            for i in range(n):
                action = actions[:, i:i + 1]
                packed_next_states = torch.cat((
                    packed_states[:, t:t + 1][:, :, :3] + action,
                    packed_states[:, t:t + 1][:, :, 3:]
                ), dim=2)
                # packed_next_states, _ = self.broadcast_env.action(packed_states[:, t:t + 1], action, inplace=False)
                next_values, _ = self.value(packed_next_states, labels=labels, hidden=hidden_value)
                action_values.append(next_values.squeeze(2))
                action_reward, _ = self.reward(
                    packed_states[:, t:t + 1],
                    labels=labels,
                    hidden=hidden_reward,
                    add=action
                )
                action_rewards.append(action_reward.squeeze(2))
            action_values = torch.stack(action_values, dim=2)
            if detach_qvalue:
                action_values = action_values.detach()
            action_rewards = torch.stack(action_rewards, dim=2)
            _, hidden_reward = self.reward(packed_states[:, t: t + 1], labels=labels, hidden=hidden_reward, add=actions[:, :1])

            step_action_values.append(action_values * self.gamma + action_rewards)
            step_action_rewards.append(action_rewards)
        return step_action_values, step_action_rewards

    def compute_policy_logprobs(self, packed_states, packed_actions, lens, labels, with_mask=False, detach_qvalue=True):
        max_len = max(lens) + 1
        mask = torch.stack(
            [torch.arange(max_len, device=packed_states.device).lt(length) for length in lens], dim=0
        )
        step_action_values, step_action_rewards = self.compute_values_and_rewards(packed_states, packed_actions, labels, detach_qvalue=detach_qvalue)

        for t in range(max_len - 1, -1, -1):
            if t + 1 < max_len:
                transit_value = step_action_values[t + 1].max(2)[0] * self.gamma + step_action_rewards[t][:, :, 0]
                transit_value = transit_value * mask[:, t].unsqueeze(1).type(torch.long) - 1e9 * (~mask[:, t].unsqueeze(1)).type(torch.long)
                step_action_values[t] = torch.cat([
                    torch.max(step_action_values[t][:, :, :1], transit_value.unsqueeze(2)),
                    step_action_values[t][:, :, 1:]
                ], dim=2)
        step_action_values = torch.cat(step_action_values, dim=1)
        state_action_logprob = F.log_softmax(step_action_values, dim=2)
        policy_logprob = state_action_logprob[:, :, 0]
        if with_mask:
            return policy_logprob, mask
        return policy_logprob

    def optimize_reward_net_dqn(self, batch_size=128, minimum_batch_size=16):
        episodes = growing_sampler(self.memory.sample_episodes, minimum_batch_size, batch_size, rho=1, record=True)
        if episodes is None:
            return
        our_episodes, records = episodes
        exp_episodes, lcps = [record[0] for record in records], [record[1] for record in records]
        exp_lens = [episode.actions.size(0) for episode in exp_episodes]
        our_lens = [episode.actions.size(0) for episode in our_episodes]

        max_len = max(exp_lens + our_lens) + 1
        labels = [episode.label for episode in exp_episodes]
        exp_packed_states = torch.stack(
            [F.pad(episode.states, (0, 0, 0, max_len - length - 1)) for episode, length in zip(exp_episodes, exp_lens)], dim=0
        )
        exp_packed_actions = torch.stack(
            [F.pad(episode.actions, (0, 0, 0, max_len - length)) for episode, length in zip(exp_episodes, exp_lens)], dim=0
        )
        our_packed_states = torch.stack(
            [F.pad(episode.states, (0, 0, 0, max_len - length - 1)) for episode, length in zip(our_episodes, our_lens)], dim=0
        )
        our_packed_actions = torch.stack(
            [F.pad(episode.actions, (0, 0, 0, max_len - length)) for episode, length in zip(our_episodes, our_lens)], dim=0
        )
        exp_mask = torch.stack([torch.arange(max_len, device=exp_packed_states.device).lt(length) for length in exp_lens], dim=0)
        our_mask = torch.stack([torch.arange(max_len, device=our_packed_states.device).lt(length) for length in our_lens], dim=0)

        exp_values, exp_rewards = self.compute_values_and_rewards(exp_packed_states, exp_packed_actions, labels, detach_qvalue=True)
        our_values, our_rewards = self.compute_values_and_rewards(our_packed_states, our_packed_actions, labels, detach_qvalue=True)

        t2lcpidx = {t: [k for k in range(len(lcps)) if lcps[k] - 1 == t] for t in range(max_len)}

        for t in range(max_len - 1, -1, -1):
            if t + 1 < max_len:
                exp_transit_value = exp_values[t + 1].max(2)[0] * self.gamma + exp_rewards[t][:, :, 0]
                exp_transit_value = exp_transit_value * exp_mask[:, t].unsqueeze(1).type(torch.long) - 1e9 * (~ exp_mask[:, t].unsqueeze(1)).type(torch.long)
                our_transit_value = our_values[t + 1].max(2)[0] * self.gamma + exp_rewards[t][:, :, 0]
                our_transit_value = our_transit_value * our_mask[:, t].unsqueeze(1).type(torch.long) - 1e9 * (~ our_mask[:, t].unsqueeze(1)).type(torch.long)

                exp_values[t] = torch.cat((
                    torch.max(exp_values[t][:, :, :1], exp_transit_value.unsqueeze(2)),
                    exp_values[t][:, :, 1:]
                ), dim=2)

                our_values[t] = torch.cat((
                    torch.max(our_values[t][:, :, :1], our_transit_value.unsqueeze(2)),
                    our_values[t][:, :, 1:]
                ), dim=2)

            for k in t2lcpidx[t]:
                exp_values[t] = torch.cat([
                    exp_values[t][:k],
                    torch.max(exp_values[t][k:k + 1], our_values[t][k:k + 1]),
                    exp_values[t][k + 1:]
                ], dim=0)

        state_action_qvalues = torch.cat(exp_values, dim=1)
        state_action_logprob = F.log_softmax(state_action_qvalues, dim=2)
        policy_logprob = state_action_logprob[:, :, 0]
        loss = -policy_logprob[exp_mask].mean()

        print('Exp episode loss=', float(loss))

        # Optimize the model
        self.reward_net_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.reward.parameters()), 10)
        self.reward_net_optimizer.step()

    def optimize_reward_net(self, batch_size=64, minimum_batch_size=16):
        self.optimize_reward_net_dqn(batch_size=batch_size, minimum_batch_size=minimum_batch_size)

    def self_training(self, trajs, answer_labels, progress=0, **kwargs):
        assert self.use_lstm
        for label in answer_labels:
            if label not in self.seen_labels:
                self.seen_labels.append(label)
        n = len(trajs)
        n_episodes = max(1, round(self.n_episodes_per_traj * n))
        max_steps = self.max_steps
        states_list = [self.traj_encoder.traj_to_tensor(traj, **kwargs) for traj in trajs]
        actions_list = [torch.LongTensor(traj[1]).to(self.device) for traj in trajs]

        self.read_expert_traj(states_list, actions_list, answer_labels)

        eps = self.eps_start + progress * (self.eps_end - self.eps_start)

        for i_episode in range(1, n_episodes + 1):
            # print('episode', self.episode_count+1, n_episodes)
            if (self.episode_count + 1) % 50 == 0:
                print('episode', self.episode_count + 1)
                if hasattr(self, '_average_qvalue'):
                    print('_average_qvalue =', self._average_qvalue)
            # Get start state
            exp_episode = self.memory.sample_episodes(1, 0)[0]
            state, label = exp_episode.states[0], exp_episode.label
            lcp = 1

            hidden_value = self.value.get_zero_lstm_state(state.unsqueeze(0).unsqueeze(0))
            hidden_reward = self.reward.get_zero_lstm_state(state.unsqueeze(0).unsqueeze(0))
            explore_states = [state]
            explore_actions = []
            for t in range(max_steps):
                # print(t)
                # print('start sample')
                action, is_proposed = self.sample_action(
                    state, label, eps,
                    hidden_value=hidden_value,
                    hidden_reward=hidden_reward,
                    proposed_action=exp_episode.actions[lcp - 1] if lcp <= len(exp_episode.actions) else None
                )
                _, hidden_reward = self.reward(state.unsqueeze(0).unsqueeze(0), labels=[label], hidden=hidden_reward, add=action.view(1, 1, -1))
                _, hidden_value = self.value(state.unsqueeze(0).unsqueeze(0), labels=[label], hidden=hidden_value)

                next_state, valid = self.perform_action(state, action)
                done = not valid
                state = next_state
                explore_states.append(next_state)
                explore_actions.append(action)
                if lcp < len(exp_episode.states) - 1 and is_proposed:
                    lcp += 1

                # Optimize the model
                if t % 4 == 0:
                    self.update_rl_net()
                if done:
                    break

            explore_states = torch.stack(explore_states, dim=0)
            explore_actions = torch.stack(explore_actions, dim=0).to(self.device).type(torch.float)
            self.memory.push(Episode(explore_states, explore_actions, None, label), record=(exp_episode, lcp))
            # print('episode end')
            self.episode_count += 1
            # if i_episode == n_episodes or self.episode_count % self.update_per_episode == 0:
            #     self.optimize_reward_net()
        # print('self training end')
