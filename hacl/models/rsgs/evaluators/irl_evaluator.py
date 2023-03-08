#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import random

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch_scatter import scatter_max

from hacl.algorithms.planning.rl_planner import RLPlanner
from hacl.p.rsgs.unified_broadcast_engine import UnifiedBroadcastEngine
from hacl.models.rsgs.encoders import StateEncoder, TrajEncoder
from hacl.models.rsgs.encoders.label_encoder import LabelEncoder
from hacl.models.rsgs.evaluators.a2c_evaluator import A2CEvaluator
from hacl.models.rsgs.evaluators.dqn_evaluator import DQNEvaluator
from hacl.models.rsgs.evaluators.evaluator import Evaluator
from hacl.models.rsgs.maxent_irl import MaxEntIRL
from hacl.models.rsgs.utils import Episode, EpisodeReplayMemory, growing_sampler
from hacl.utils.math import gauss_log_prob


class IRLEvaluator(Evaluator):
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
        gamma=0.8,
        eps_start=0.9,
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
        self.n_episodes_per_traj = n_episodes_per_traj if n_episodes_per_traj is not None else 1
        self.max_steps = max_steps if max_steps is not None else (50 if not self.continuous else 30)
        self.update_per_episode = update_per_episode
        self.episode_count = 0
        self.classify_with_reward = classify_with_reward
        self.entropy_beta = 1e-4

        self.broadcast_env = UnifiedBroadcastEngine(env_name, env_args)
        self.traj_encoder = TrajEncoder(env_name, env_args, require_encoder=False)

        self.action_dim = self.broadcast_env.get_action_dim()

        self.rl_model = rl_model
        self.irl = MaxEntIRL(
            in_dim=None,
            action_dim=self.action_dim,
            input_encoder_gen=lambda: StateEncoder(env_name, env_args, s_dim=h_dim),
            label_encoder_gen=lambda: LabelEncoder(add_labels, h_dim=h_dim // 2),
            use_lstm=use_lstm,
            continuous=self.continuous,
            rl_model=rl_model,
            reward_activation='linear',
            reward_logprob=True,
        )

        self.reward_net_optimizer = None
        self.rl_net_optimizer = None

        self.memory = EpisodeReplayMemory(100000, has_expert=True, expert_capacity=1000)
        self.seen_labels = []

    def get_planner(self):
        if self.classify_with_reward:
            planner = RLPlanner(env_name=self.env_name, env_args=self.env_args, net=self.irl, is_irl=True, gamma=self.gamma)
        else:
            planner = RLPlanner(env_name=self.env_name, env_args=self.env_args, net=self.irl.rl_net)
        return planner.to(self.device)

    def get_training_parameters(self):
        if self.train_by_classification:
            if self.rl_model == 'dqn':
                return self.irl.reward_net.parameters()
            elif self.rl_model == 'a2c':
                return self.irl.a2c.parameters()
        else:
            return None

    @property
    def qvalue_based(self):
        return False

    @property
    def online_optimizer(self):
        if self.rl_model == 'dqn':
            return self.reward_net_optimizer
        elif self.rl_model == 'a2c':
            return self.rl_net_optimizer

    def extra_state_dict(self):
        return {key: getattr(self, key) for key in self._EXTRA_DICT_KEY}

    def load_extra_state_dict(self, extra_dict):
        for key in self._EXTRA_DICT_KEY:
            setattr(self, key, extra_dict[key])

    def prepare_epoch(self, *args, lr=0.02, base_lr=None, progress=None, **kwargs):
        self.reward_net_optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.irl.reward_net.parameters()), lr=lr, weight_decay=1e-4
        )
        self.rl_net_optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.irl.rl_net.parameters()), lr=lr
        )
        self.train_by_classification = True

    def forward(self, trajs, labels=None, training=False, answer_labels=None, progress=None, save_dir=None, **kwargs):
        # xx = torch.tensor([[i, 1, 10] for i in range(1, 11)]).type(torch.float).to(self.device)
        # # qva = self.irl.rl_net(xx.unsqueeze(1), labels=['A'] * 10).tolist()
        # rwa = self.irl.reward_net(xx.unsqueeze(1), labels=['A'] * 10).tolist()
        # # qvb = self.irl.rl_net(xx.unsqueeze(1), labels=['B'] * 10).tolist()
        # rwb = self.irl.reward_net(xx.unsqueeze(1), labels=['B'] * 10).tolist()
        # for i in range(4, 6):
        #     print('ra=%s rb=%s' % (
        #         ','.join('%3.3f' % t for t in rwa[i][0]),
        #         ','.join('%3.3f' % t for t in rwb[i][0])
        #     ))
        # exit()

        # xx = torch.tensor([[i, 1, 10] for i in range(1, 11)]).type(torch.float).to(self.device)
        # qva = self.irl.rl_net(xx.unsqueeze(1), labels=['A'] * 10).tolist()
        # rwa = self.irl.reward_net(xx.unsqueeze(1), labels=['A'] * 10).tolist()
        # qvb = self.irl.rl_net(xx.unsqueeze(1), labels=['B'] * 10).tolist()
        # rwb = self.irl.reward_net(xx.unsqueeze(1), labels=['B'] * 10).tolist()
        # for i in range(10):
        #     print('qa=%s, qb=%s, ra=%s rb=%s' % (
        #         ','.join('%3.3f' % t for t in qva[i][0]),
        #         ','.join('%3.3f' % t for t in qvb[i][0]),
        #         ','.join('%3.3f' % t for t in rwa[i][0]),
        #         ','.join('%3.3f' % t for t in rwb[i][0])
        #     ))
        # self.analysis_on_trajs(trajs, labels, answer_labels)
        # exit()

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

        # if not training:
        #     self.show_model_from_data(trajs, save_dir=save_dir)
        #     exit()
        #     self.analysis_on_trajs(trajs, labels, answer_labels)
        #     pass
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
                detach_qvalue=True,
            ).view(len(labels), n, max_len)
            scores = []
            for i in range(n):
                scores.append(F.log_softmax(policy_logprobs[:, i, : actions_lens[i]].sum(1), dim=0))
            return torch.stack(scores, dim=0)

        elif self.rl_model == 'dqn':
            return DQNEvaluator.compute_label_scores(
                self.irl.dqn, trajs, states_lens, packed_states, labels=labels, use_lstm=self.use_lstm
            )
        elif self.rl_model == 'a2c':
            return A2CEvaluator.compute_label_scores(
                self.irl.a2c,
                trajs,
                states_lens,
                packed_states,
                actions_list,
                labels=labels,
                use_lstm=self.use_lstm,
                continuous=self.continuous,
            )

    def train(self, mode=True):
        super(IRLEvaluator, self).train(mode=mode)

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

    def sample_action(self, state, label, eps, hidden=None):
        if self.rl_model == 'dqn':
            return DQNEvaluator.dqn_sample_action(
                self.irl.dqn, state, label, eps, hidden=hidden, use_lstm=self.use_lstm
            )
        elif self.rl_model == 'a2c':
            return A2CEvaluator.a2c_sample_action(
                self.irl.a2c,
                state,
                label,
                eps,
                hidden=hidden,
                use_lstm=self.use_lstm,
                continuous=self.continuous,
                broadcast_env=self.broadcast_env,
            )

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

    def perform_action(self, state, action):
        new, valid = self.broadcast_env.action(state.unsqueeze(0), action, inplace=False)
        return new.squeeze(0), valid.item()

    def update_dqn(self, batch_size=128, rho=None, minimum_batch_size=16):
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
        packed_actions = torch.stack(
            [F.pad(episode.actions, (0, max_len - length)) for episode, length in zip(episodes, lens)], dim=0
        )
        labels = [episode.label if random.random() < .5 or len(self.seen_labels) == 0 else random.choice(self.seen_labels) for episode in episodes]
        mask = torch.stack([torch.arange(max_len, device=packed_states.device).lt(length) for length in lens], dim=0)

        state_action_values = self.irl.dqn(
            packed_states,
            labels=labels,
            hidden=self.irl.dqn.get_zero_lstm_state(packed_states) if self.use_lstm else None,
        )
        state_action_values = (
            (state_action_values[0] if self.use_lstm else state_action_values)
                .gather(2, packed_actions.unsqueeze(2))
                .squeeze(2)
        )

        next_state_values = self.irl.dqn(
            packed_states,
            labels=labels,
            hidden=self.irl.dqn.get_zero_lstm_state(packed_states) if self.use_lstm else None,
        )
        next_state_values = (next_state_values[0] if self.use_lstm else next_state_values).max(2)[0]
        next_state_values = torch.cat(
            [next_state_values[:, 1:], torch.zeros_like(next_state_values[:, :1])], dim=1
        ).detach()  # TODO check whether we should detach here

        state_rewards = self.irl(
            packed_states, labels=labels, hidden=self.irl.get_zero_lstm_state(packed_states) if self.use_lstm else None
        )
        state_rewards = (
            (state_rewards[0] if self.use_lstm else state_rewards)
                .gather(2, packed_actions.unsqueeze(2))
                .squeeze(2)
                .detach()
        )

        expected_state_action_values = next_state_values * self.gamma + state_rewards

        loss = F.l1_loss(expected_state_action_values[mask], state_action_values[mask])
        self._average_qvalue = state_action_values[mask].mean().detach().cpu().item()
        # Optimize the model
        self.rl_net_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.irl.dqn.parameters()), 10)
        self.rl_net_optimizer.step()

    def update_a2c(self, batch_size=128, rho=None, minimum_batch_size=16, optimize_reward=False):
        if rho is None:
            rho = self.rho
        episodes = growing_sampler(self.memory.sample_episodes, minimum_batch_size, batch_size, rho=rho)
        if episodes is None:
            return

        lens = [episode.actions.size(0) for episode in episodes]
        max_len = max(lens) + 1

        packed_labels = [episode.label for episode in episodes]
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
        packed_rewards = self.irl(
            packed_states,
            packed_actions,
            hidden=self.irl.reward_net.get_zero_lstm_state(packed_states) if self.use_lstm else None,
            labels=packed_labels
        )
        if self.use_lstm:
            packed_rewards = packed_rewards[0]
        labels = [episode.label for episode in episodes]
        mask = torch.stack([torch.arange(max_len, device=packed_states.device).lt(length) for length in lens], dim=0)

        outputs = self.irl.a2c(packed_states, labels=labels, hidden=self.irl.a2c.get_zero_lstm_state(packed_states) if self.use_lstm else None)

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

        if not optimize_reward:
            # Optimize the model
            self.rl_net_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.irl.a2c.parameters()), 10)
            self.rl_net_optimizer.step()
        else:
            # Optimize reward model
            self.reward_net_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.irl.reward_net.parameters()), 10)
            self.reward_net_optimizer.step()

    def update_rl_net(self, *args, **kwargs):
        if self.rl_model == 'dqn':
            self.update_dqn(*args, **kwargs)
        elif self.rl_model == 'a2c':
            self.update_a2c(*args, **kwargs)

    def compute_qvalues_and_rewards(self, packed_states, labels, detach_qvalue=True):
        max_len = packed_states.size(1)
        state_action_qvalues = []
        step_rewards = []
        dqn_hidden = self.irl.dqn.get_zero_lstm_state(packed_states[:, :1]) if self.use_lstm else None
        reward_hidden = self.irl.reward_net.get_zero_lstm_state(packed_states[:, :1]) if self.use_lstm else None
        for t in range(max_len):
            if self.use_lstm:
                _, dqn_hidden = self.irl.dqn(packed_states[:, t: t + 1], labels=labels, hidden=dqn_hidden)
            action_reward = self.irl.reward_net(packed_states[:, t: t + 1], labels=labels, hidden=reward_hidden)
            if self.use_lstm:
                action_reward, reward_hidden = action_reward
            action_qvalues = []
            for action in range(self.broadcast_env.get_action_dim()):
                packed_next_states, _ = self.broadcast_env.action(packed_states[:, t: t + 1], action, inplace=False)
                qvalues = self.irl.dqn(packed_next_states, labels=labels, hidden=dqn_hidden)
                qvalues = (qvalues[0] if self.use_lstm else qvalues).max(2)[0]
                action_qvalues.append(qvalues)
            action_qvalues = torch.stack(action_qvalues, dim=-1)
            if detach_qvalue:
                action_qvalues = action_qvalues.detach()
            state_action_qvalues.append(action_qvalues * self.gamma + action_reward)
            step_rewards.append(action_reward)
        return state_action_qvalues, step_rewards

    def compute_policy_logprobs(self, packed_states, packed_actions, lens, labels, with_mask=False, detach_qvalue=True):
        max_len = max(lens) + 1
        mask = torch.stack(
            [torch.arange(max_len, device=packed_states.device).lt(length) for length in lens], dim=0
        )
        if self.rl_model == 'dqn':
            state_action_qvalues, step_rewards = self.compute_qvalues_and_rewards(packed_states, labels, detach_qvalue=detach_qvalue)

            for t in range(max_len - 1, -1, -1):
                if t + 1 < max_len:
                    transit_qvalue = state_action_qvalues[t + 1].max(2)[0] * self.gamma + step_rewards[t].gather(2, packed_actions[:, t:t + 1].unsqueeze(2)).squeeze(2)
                    transit_qvalue = transit_qvalue * mask[:, t].unsqueeze(1).type(torch.long) - 1e9 * (~mask[:, t].unsqueeze(1)).type(torch.long)
                    scatter_max(transit_qvalue.unsqueeze(2), packed_actions[:, t:t + 1].unsqueeze(2), 2, state_action_qvalues[t])
            state_action_qvalues = torch.cat(state_action_qvalues, dim=1)
            state_action_logprob = F.log_softmax(state_action_qvalues, dim=2)
        elif self.rl_model == 'a2c':
            if not self.continuous:
                state_action_scores = self.irl.a2c(
                    packed_states,
                    labels=labels,
                    hidden = self.irl.a2c.get_zero_lstm_state(packed_states) if self.use_lstm else None
                )
                if self.use_lstm:
                    state_action_scores = state_action_scores[0]
                state_action_logprob = F.log_softmax(state_action_scores, dim=2)
            else:
                state_action_distribution = self.irl.a2c(
                    packed_states,
                    labels=labels,
                    hidden=self.irl.a2c.get_zero_lstm_state(packed_states) if self.use_lstm else None
                )
                if self.use_lstm:
                    (mu, var), value, _ = state_action_distribution
                else:
                    (mu, var), value = state_action_distribution
                state_action_logprob = self.compute_logprob(mu, var, packed_actions).sum(2)
        else:
            raise ValueError()
        if not self.continuous:
            policy_logprob = state_action_logprob.gather(2, packed_actions.unsqueeze(2)).squeeze(2)
        else:
            policy_logprob = state_action_logprob
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
            [F.pad(episode.actions, (0, max_len - length)) for episode, length in zip(exp_episodes, exp_lens)], dim=0
        )
        our_packed_states = torch.stack(
            [F.pad(episode.states, (0, 0, 0, max_len - length - 1)) for episode, length in zip(our_episodes, our_lens)], dim=0
        )
        our_packed_actions = torch.stack(
            [F.pad(episode.actions, (0, max_len - length)) for episode, length in zip(our_episodes, our_lens)], dim=0
        )
        exp_mask = torch.stack([torch.arange(max_len, device=exp_packed_states.device).lt(length) for length in exp_lens], dim=0)
        our_mask = torch.stack([torch.arange(max_len, device=our_packed_states.device).lt(length) for length in our_lens], dim=0)

        exp_qvalues, exp_rewards = self.compute_qvalues_and_rewards(exp_packed_states, labels, detach_qvalue=True)
        our_qvalues, our_rewards = self.compute_qvalues_and_rewards(our_packed_states, labels, detach_qvalue=True)

        t2lcpidx = {t: [k for k in range(len(lcps)) if lcps[k] - 1 == t] for t in range(max_len)}

        for t in range(max_len - 1, -1, -1):
            if t + 1 < max_len:
                exp_transit_qvalue = exp_qvalues[t + 1].max(2)[0] * self.gamma + exp_rewards[t].gather(2, exp_packed_actions[:, t:t + 1].unsqueeze(2)).squeeze(2)
                exp_transit_qvalue = exp_transit_qvalue * exp_mask[:, t].unsqueeze(1).type(torch.long) - 1e9 * (~ exp_mask[:, t].unsqueeze(1)).type(torch.long)
                our_transit_qvalue = our_qvalues[t + 1].max(2)[0] * self.gamma + our_rewards[t].gather(2, our_packed_actions[:, t:t + 1].unsqueeze(2)).squeeze(2)
                our_transit_qvalue = our_transit_qvalue * our_mask[:, t].unsqueeze(1).type(torch.long) - 1e9 * (~ our_mask[:, t].unsqueeze(1)).type(torch.long)

                scatter_max(exp_transit_qvalue.unsqueeze(2), exp_packed_actions[:, t:t + 1].unsqueeze(2), 2, exp_qvalues[t])
                scatter_max(our_transit_qvalue.unsqueeze(2), our_packed_actions[:, t:t + 1].unsqueeze(2), 2, our_qvalues[t])

            for k in t2lcpidx[t]:
                exp_qvalues[t] = torch.cat([
                    exp_qvalues[t][:k],
                    torch.max(exp_qvalues[t][k:k + 1], our_qvalues[t][k:k + 1]),
                    exp_qvalues[t][k + 1:]
                ], dim=0)

        state_action_qvalues = torch.cat(exp_qvalues, dim=1)
        state_action_logprob = F.log_softmax(state_action_qvalues, dim=2)
        policy_logprob = state_action_logprob.gather(2, exp_packed_actions.unsqueeze(2)).squeeze(2)
        loss = -policy_logprob[exp_mask].mean()
        # print(exp_packed_actions[exp_mask], torch.cat([state_action_qvalues[exp_mask], state_action_qvalues.gather(2, exp_packed_actions.unsqueeze(2)).squeeze(2)[exp_mask].unsqueeze(1)], dim=1), exp_lens)

        # print(exp_packed_actions[exp_mask], state_action_qvalues[exp_mask][:, 1:3].mean(), state_action_qvalues.gather(2, exp_packed_actions.unsqueeze(2)).squeeze(2)[exp_mask].unsqueeze(1).mean())
        #
        # input()
        print('Exp episode loss=', float(loss))
        # print(state_action_logprob[exp_mask])
        # input()

        # Optimize the model
        self.reward_net_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.irl.dqn.parameters()), 10)
        # for param in self.irl.reward_net.parameters():
        #     print(param.grad)
        # input()
        # print('grad to bias is', self.irl.dqn.decoder[4].bias.grad)
        self.reward_net_optimizer.step()

    def optimize_reward_net_a2c(self, batch_size=128, minimum_batch_size=16):
        self.update_a2c(rho=0, batch_size=batch_size, minimum_batch_size=minimum_batch_size, optimize_reward=True)

    def optimize_reward_net(self, batch_size=128, minimum_batch_size=16):
        if self.rl_model == 'dqn':
            self.optimize_reward_net_dqn(batch_size=batch_size, minimum_batch_size=minimum_batch_size)
        elif self.rl_model == 'a2c':
            self.optimize_reward_net_a2c(batch_size=batch_size, minimum_batch_size=minimum_batch_size)
        else:
            raise ValueError()

    def self_training(self, trajs, answer_labels, progress=0, **kwargs):
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
            if (self.episode_count + 1) % 250 == 0:
                print('episode', self.episode_count + 1)
                if hasattr(self, '_average_qvalue'):
                    print('_average_qvalue =', self._average_qvalue)
            # Get start state
            exp_episode = self.memory.sample_episodes(1, 0)[0]
            state, label = exp_episode.states[0], exp_episode.label
            lcp = 1

            hidden = self.irl.rl_net.get_zero_lstm_state(state.unsqueeze(0).unsqueeze(0)) if self.use_lstm else None
            explore_states = [state]
            explore_actions = []
            for t in range(max_steps):
                # print(t)
                # print('start sample')
                if self.use_lstm:
                    action, hidden = self.sample_action(state, label, eps, hidden=hidden)
                else:
                    action = self.sample_action(state, label, eps)

                # print('start perform')
                # print('action=', action)
                next_state, valid = self.perform_action(state, action)
                # print('end perform')
                done = not valid
                state = next_state
                explore_states.append(next_state)
                explore_actions.append(action)
                if lcp < len(exp_episode.states) and torch.equal(next_state, exp_episode.states[lcp]):
                    lcp += 1

                # Optimize the model
                if self.rl_model == 'dqn' or (self.rl_model == 'a2c' and t % 4 == 0):
                    # print('start update_rl_net')
                    self.update_rl_net()
                    # print('end update_rl_net')
                if done:
                    break

            explore_states = torch.stack(explore_states, dim=0)
            explore_actions = (
                torch.LongTensor(explore_actions).to(self.device) if not self.continuous
                else torch.stack(explore_actions, dim=0).to(self.device).type(torch.float)
            )
            self.memory.push(Episode(explore_states, explore_actions, None, label), record=(exp_episode, lcp))
            # print('episode end')
            self.episode_count += 1
            if i_episode == n_episodes or self.episode_count % self.update_per_episode == 0:
                self.optimize_reward_net()
        # print('self training end')

    def analysis_on_trajs(self, trajs, labels, answer_labels, **kwargs):
        states_list = [self.traj_encoder.traj_to_tensor(traj, **kwargs) for traj in trajs]
        actions_list = [self.traj_encoder.traj_to_action_tensor(traj, **kwargs) for traj in trajs]
        for traj, states, actions, answer_label in zip(trajs, states_list, actions_list, answer_labels):
            qmax_list = {}
            qsa_list = {}
            pp_list = {}
            rsa_list = {}
            for label in labels:
                q = self.irl.dqn(
                    states.unsqueeze(0),
                    labels=[label],
                    hidden=self.irl.dqn.get_zero_lstm_state(states.unsqueeze(0)) if self.use_lstm else None,
                )
                if self.use_lstm:
                    q = q[0]
                qmax = q.squeeze(0).max(1)[0]
                pp = torch.exp(F.log_softmax(q.squeeze(0), dim=1).gather(1, actions.unsqueeze(1)).squeeze(1))
                qsa = q.squeeze(0)[:-1].gather(1, actions.unsqueeze(1)).squeeze(1)
                r = self.irl.reward_net(
                    states.unsqueeze(0),
                    labels=[label],
                    hidden=self.irl.reward_net.get_zero_lstm_state(states.unsqueeze(0)) if self.use_lstm else None,
                )
                if self.use_lstm:
                    r = r[0]
                rsa = r.squeeze(0)[:-1].gather(1, actions.unsqueeze(1)).squeeze(1)
                qmax_list[label] = qmax
                qsa_list[label] = qsa
                pp_list[label] = pp
                rsa_list[label] = rsa
            print('#' * 80)
            print('True label=', answer_label)
            for i in range(actions.size(0)):
                print('s=', traj[0][i], 'a=', traj[1][i])
                print('for labels =', labels)
                print('V(s) =', ', '.join(['%.3f' % float(qmax_list[label][i].item()) for label in labels]), end=', ')
                print('Q(s,a) =', ', '.join(['%.3f' % float(qsa_list[label][i].item()) for label in labels]), end=', ')
                print('R(s,a) =', ', '.join(['%.3f' % float(rsa_list[label][i].item()) for label in labels]), end=', ')
                print(
                    'V(s\')*γ+R(s,a)-Q(s,a)=',
                    ', '.join(
                        [
                            '%.3f'
                            % float(
                                (qmax_list[label][i + 1] * self.gamma + rsa_list[label][i] - qsa_list[label][i]).item()
                            )
                            for label in labels
                        ]
                    ),
                    end=', ',
                )
                print('π(a|s) =', ', '.join(['%.3f' % float(pp_list[label][i].item()) for label in labels]))
                input()
            print('s=', traj[0][-1], end='\n\n')
            input()

    def get_value_and_policy_from_states(self, states, label):
        states = self.traj_encoder.traj_to_tensor((states, [0] * (len(states) - 1))).unsqueeze(1)

        qsa = self.irl.dqn(
            states,
            labels=[label] * states.size(0),
            hidden=self.irl.dqn.get_zero_lstm_state(states) if self.use_lstm else None,
        )
        if self.use_lstm:
            qsa = qsa[0]
        value = qsa.max(2)[0].squeeze(1)
        policy = (qsa[:, :, 1:5].max(2)[1] + 1).squeeze(1)
        return value.cpu(), policy.cpu()

    def visualize(self, map_size, marked_positions, labels, save_dir=None):
        from hacl.envs.gridworld.gridworld_2d.visualize.data_visualizer import GridWorldDataVisualizer
        from hacl.envs.gridworld.gridworld_2d.components import Position
        import os

        os.makedirs(save_dir, exist_ok=True)
        for label in labels:
            f = GridWorldDataVisualizer(map_size=map_size, n_diagram=1)
            f.make_grid(aid=0, marked_positions=marked_positions, title=label)
            states = []
            for x in range(1, map_size + 1):
                for y in range(1, map_size + 1):
                    if Position(x, y) not in marked_positions.values():
                        states.append(
                            (marked_positions['A'], marked_positions['B'], marked_positions['C'], Position(x, y), 0)
                        )
            values, policies = self.get_value_and_policy_from_states(states, label)
            values -= values.min()
            values /= max(1e-4, float(values.max()))
            for i in range(len(states)):
                x, y = states[i][-2].x, states[i][-2].y
                d = int(policies[i].item())
                v = values[i]
                f.add_dot(x, y, radius=0.4, color=f.grey(v))
                f.add_arrow(x, y, d, position='out', length=0.2, width=0.05, color='black')
            f.savefig(os.path.join(save_dir, 'irl-' + label + '.png'))
            f.close()

    def show_model_from_data(self, trajs, save_dir):
        pass