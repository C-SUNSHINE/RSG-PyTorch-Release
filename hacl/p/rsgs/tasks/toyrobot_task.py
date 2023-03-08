#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import torch

from hacl.algorithms.rrt.builder import build_rrt, build_rrt_graph
from hacl.algorithms.value_based_classification import QValueGraphBasedActionClassifier
from hacl.envs.simple_continuous.playroom_gdk.toyrobot_v20210423 import ToyRobotV20210423
from hacl.p.rsgs.tasks.task_pipeline import TaskPipeline
from hacl.p.rsgs.utils import average, dump_confusing_matrix, get_acc_dict, make_dict_default, medium


def build_rrt_graph_from_path(args):
    if len(args) == 3:
        path, env_args, init_symbolic_state = args
        start_state = path[0]
    else:
        path, env_args, init_symbolic_state, start_state = args
    env = ToyRobotV20210423(env_args)
    env.load_from_symbolic_state(init_symbolic_state)
    rrt = build_rrt(env.pspace, start_state=start_state, nr_iterations=200)
    graph = build_rrt_graph(env.pspace, rrt, path, tqdm=False, full_graph=False)
    return rrt, graph


class ToyRobotTask(TaskPipeline):
    def __init__(
        self, data_loader, label2state_machine, meters=None, save_dir=None, log=print, use_tb=False, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.logger = log
        self.save_dir = save_dir
        self.meters = meters
        self.use_tb = use_tb

        self.data_loader = data_loader
        self.labels = self.data_loader.labels
        self.n_labels = {k: len(v) for k, v in self.labels.items()}
        self.label2index = {k: {label: i for i, label in enumerate(self.labels[k])} for k in self.labels}
        self.label2state_machine = label2state_machine
        self.qvalue_classifier = {
            split: QValueGraphBasedActionClassifier(
                self.data_loader.labels[split],
                label2state_machine,
            )
            for split in ('train', 'val', 'test')
        }
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.last_epoch_answer = None
        self.last_epoch_stored_samples = None
        self.last_epoch_planning_results = None

    def init(self):
        pass

    def epoch_init(self, epoch):
        self.losses = {}
        self.accs = {}
        self.agg_accs = {}
        self.confusing_matrix_configs = [
            ('trueip', 'ours'),
        ]
        self.confusing_matrices = {}
        self.stored_data = []
        self.kept_groups = {}
        self.planning_results = {}

    def get_model_args(self, graphs, rrts):
        model_args = defaultdict(lambda: dict())
        model_args.update(
            ours=dict(
                graphs=graphs,
                rrts=rrts,
                n_iters=100,
            ),
            trueip=dict(
                graphs=graphs,
                n_iters=100,
            ),
        )
        return model_args

    def _get_zero_confusing_matrix(self, split, dtype=np.int):
        return np.zeros((self.n_labels[split], self.n_labels[split]), dtype=dtype)

    def plot_loss_landscape(self, traj, graph, rrt, evaluator, split, labels, obj='Ball', objs=['Ball', 'Bell', 'Monkey', 'MusicOn', 'MusicOff', 'LightOn']):
        siz = 20
        n = 10
        score_mat = np.zeros((n, n))
        label2index = {l: i for i, l in enumerate(labels)}
        n_labels = len(labels)
        import matplotlib.pyplot as plt
        for x in range(n):
            for y in range(n):
                print('Set (x,y)=(%d,%d)' % (x, y))
                for ob in objs:
                    if ('eff_' + ob) in evaluator.init_values.init_value.net:
                        evaluator.init_values.init_value.net['eff_' + ob][1].set_target(siz * (x + .5) / n, siz * (y + .5) / n, force_target=True)
                evaluation = evaluator([traj], [graph], [rrt], n_iters=100)
                raw_qvalues = evaluation
                # print(raw_qvalues[label2index[obj]][0])
                # input()
                qvalues = [raw_qvalues[l][0].to(self.device) for l in range(n_labels)]
                states = [x[0] for x in traj]
                states_tensor = torch.Tensor(states).to(qvalues[0].device) if not hasattr(evaluator, 'get_states_tensor') else evaluator.get_states_tensor(states, traj[0])
                scores, paths = self.qvalue_classifier[split].prob(
                    states_tensor, qvalues, evaluator.init_values, labels=labels, record_path=True, device=self.device
                )
                value = scores[label2index[obj]]
                # value = raw_qvalues[label2index[obj]][0][:, 0].sum()
                score_mat[n - 1 - y, x] = value
                plt.text(x, n - 1 - y, '%.3f' % value, ha='center', va='center')
        plt.imshow(score_mat)
        plt.show()
        plt.close()

    def batch_run(
        self, batch_id, split, batch, last_test=False, train_choice=False, train_choice_adjacent=False, progress=None, *args, **kwargs
    ):
        evaluators = self.evaluators
        alpha = (kwargs['alpha'] if 'alpha' in kwargs else None) or 1.0
        beta = (kwargs['beta'] if 'beta' in kwargs else None) or 0.0
        training = split == 'train'
        n_groups = len(batch)
        env_args = self.data_loader.get_env_args()
        env = self.data_loader.build_env_from_args(env_args)
        trajs = []
        answer_labels_for_training = []
        for (group, label) in batch:
            trajs.extend(group['trajs'])
            answer_labels_for_training.extend([label] * len(group['trajs']))
        paths = [[symbolic_state[0] for symbolic_state in traj[0]] for traj in trajs]
        init_symbolic_states = [traj[0][0] for traj in trajs]

        if train_choice is not None and split != 'test':
            import random
            assert train_choice <= len(self.labels[split])
            if not train_choice_adjacent:
                batch_label = batch[0][1]
                negative_labels = [label for label in self.labels[split] if label != batch_label]
                random.shuffle(negative_labels)
                assert train_choice <= len(negative_labels) + 1
                labels = [batch_label] + negative_labels[: train_choice - 1]
                random.shuffle(labels)
            else:
                batch_label = batch[0][1]
                pos = list(filter(lambda i: self.labels[split][i] == batch_label, range(len(self.labels[split]))))[0]
                start_pos = len(self.labels[split]) + pos - (train_choice - 1) // 2
                labels = (self.labels[split] * 3)[start_pos:start_pos + train_choice]
        else:
            labels = self.labels[split]
        label2index = {l: i for i, l in enumerate(labels)}
        n_labels = len(labels)

        if any(ename in evaluators for ename in ['ours', 'trueip']):
            print("Start building rrt")
            if len(trajs) < 5:  # Single processing
                graphs = []
                rrts = []
                for path, init_symbolic_state in zip(paths, init_symbolic_states):
                    env.load_from_symbolic_state(init_symbolic_state)
                    rrt = build_rrt(env.pspace, start_state=path[0], nr_iterations=200)
                    rrts.append(rrt)
                    graph = build_rrt_graph(env.pspace, rrt, path, tqdm=False, full_graph=False)
                    graphs.append(graph)
            else:  # Multi processing
                input_args = [(path, env_args, init_symbolic_state) for path, init_symbolic_state in zip(paths, init_symbolic_states)]
                with Pool(min(len(trajs), 25)) as pool:
                    rrt_graphs = list(pool.map(build_rrt_graph_from_path, input_args))
                rrts = [x[0] for x in rrt_graphs]
                graphs = [x[1] for x in rrt_graphs]

            for traj, path, init_symbolic_state, graph, rrt, answer_label in zip(trajs, paths, init_symbolic_states, graphs, rrts, answer_labels_for_training):
                env.load_from_symbolic_state(init_symbolic_state)
                # print(init_symbolic_state)
                # visualize_problem_and_solution(env.pspace, path=path, rrt=rrt, window=answer_label, play_step=False)
                # self.plot_loss_landscape(traj, graph, rrt, evaluators['ours'], split, labels)
                # input('next data?')
            print("Finish building rrt")
        else:
            rrts, graphs = None, None

        evaluations = {}

        model_args = self.get_model_args(graphs, rrts)

        for ename, evaluator in evaluators.items():
            evaluations[ename] = evaluator(
                trajs,
                **model_args[ename],
                labels=labels,
                training=training,
                progress=progress,
                answer_labels=answer_labels_for_training if training else None,
            )

        losses = {ename: [] for ename in evaluators}
        group_kept = {ename: [] for ename in evaluators}
        sample_kept = {ename: [] for ename in evaluators}
        correct = []

        start_index = 0
        start_indices = []
        end_indices = []
        for group_id, (group, label) in enumerate(batch):
            end_index = start_index + len(group['trajs'])
            tar = label2index[label]
            agg_scores = {
                ename: torch.zeros(n_labels, device=evaluator.device) for ename, evaluator in evaluators.items()
            }

            for traj_id, traj in enumerate(trajs[start_index:end_index]):
                i = start_index + traj_id
                for ename, evaluator in evaluators.items():
                    if evaluator.qvalue_based:
                        raw_qvalues = evaluations[ename]
                        qvalues = [raw_qvalues[l][i].to(self.device) for l in range(n_labels)]
                        states_tensor = evaluator.states_to_states_tensor(traj[0])
                        scores, paths = self.qvalue_classifier[split].prob(
                            states_tensor, qvalues, evaluator.init_values, labels=labels, record_path=True, device=self.device
                        )
                        log_probs = torch.log_softmax(scores, dim=0)
                        probs = torch.softmax(log_probs, dim=0)
                        output, acc_dict = get_acc_dict(scores, tar)
                        agg_scores[ename] += scores
                        losses[ename].append(- alpha * torch.log_softmax(scores, dim=0)[tar] - beta * scores[tar])
                        # print(alpha, beta)
                        # print(- alpha * torch.log_softmax(scores, dim=0)[tar], - beta * scores[tar])
                        # print(labels[tar], -beta * scores[tar])

                        sample_kept[ename].append(
                            dict(
                                scores=scores.detach().cpu(),
                                probs=probs.detach().cpu(),
                                paths=paths,
                                qvalues=[x.detach().cpu() for x in qvalues],
                                output=output,
                                output_label=labels[output],
                                target=tar,
                                target_label=labels[tar],
                                acc=acc_dict,
                            )
                        )
                    else:
                        scores = evaluations[ename][i]
                        log_probs = torch.log_softmax(scores, dim=0)
                        probs = torch.softmax(log_probs, dim=0)
                        output, acc_dict = get_acc_dict(scores, tar)
                        agg_scores[ename] += scores
                        losses[ename].append(- alpha * torch.log_softmax(scores, dim=0)[tar] - beta * scores[tar])

                        sample_kept[ename].append(
                            dict(
                                scores=scores.detach().cpu(),
                                probs=probs.detach().cpu(),
                                output=output,
                                output_label=labels[output],
                                target=tar,
                                target_label=labels[tar],
                                acc=acc_dict,
                            )
                        )

            for ename, evaluator in evaluators.items():
                output, acc_dict = get_acc_dict(agg_scores[ename], tar)
                output_label = labels[output]

                group_kept[ename].append(
                    dict(
                        agg_scores=agg_scores[ename].detach().cpu(),
                        agg_output=output,
                        agg_output_label=output_label,
                        target=tar,
                        target_label=labels[tar],
                        acc=acc_dict,
                    )
                )
            correct.append('ours' in group_kept and group_kept['ours'][-1]['agg_output'] == tar)

            # Update Confusing Matrix:
            for row, col in list(('@', ename) for ename in evaluators) + self.confusing_matrix_configs:
                if (row != '@' and row not in evaluators.keys()) or col not in evaluators.keys():
                    continue
                for cmmode in ('prob', 'single', 'agg'):
                    if cmmode == 'agg' and self.data_loader.group_size(split) == 1:
                        continue
                    cmname = ("Answer" if row == '@' else row) + " vs " + col + ' -' + cmmode + ' :  ' + split
                    if cmname not in self.confusing_matrices:
                        self.confusing_matrices[cmname] = self._get_zero_confusing_matrix(
                            split, dtype=np.float if cmmode == 'prob' else np.int
                        )
                    if cmmode == 'prob':
                        for j in range(start_index, end_index):
                            row_label = label if row == '@' else sample_kept[row][j]['output_label']
                            for col_label_id in range(n_labels):
                                col_label = labels[col_label_id]
                                pair_prob = sample_kept[col][j]['probs'][col_label_id]
                                self.confusing_matrices[cmname][
                                    self.label2index[split][row_label], self.label2index[split][col_label]
                                ] += pair_prob
                    elif cmmode == 'single':
                        for j in range(start_index, end_index):
                            row_label = label if row == '@' else sample_kept[row][j]['output_label']
                            col_label = sample_kept[col][j]['output_label']
                            self.confusing_matrices[cmname][
                                self.label2index[split][row_label], self.label2index[split][col_label]
                            ] += 1
                    elif cmmode == 'agg':
                        row_label = label if row == '@' else group_kept[row][-1]['agg_output_label']
                        col_label = group_kept[col][-1]['agg_output_label']
                        self.confusing_matrices[cmname][
                            self.label2index[split][row_label], self.label2index[split][col_label]
                        ] += 1

            if split == 'test':
                print(
                    "Label: %s, Output: %s, True: %s"
                    % (
                        label,
                        group_kept['ours'][-1]['agg_output_label'] if 'ours' in group_kept else 'none',
                        group_kept['trueip'][-1]['agg_output_label'] if 'trueip' in group_kept else 'none',
                    )
                )

            # Update start end index
            start_indices.append(start_index)
            end_indices.append(end_index)
            start_index = end_index

        for ename in group_kept:
            make_dict_default(self.kept_groups, ename, [])
            self.kept_groups[ename].extend(group_kept[ename])

        # Average loss over batches
        losses = {ename: sum(loss) / len(loss) for ename, loss in losses.items()}

        # Get statistic of loss and accuracy
        for ename in evaluators:
            if ename in losses:
                make_dict_default(self.losses, ename, dict())
                make_dict_default(self.losses[ename], split, [])
                self.losses[ename][split].append(float(losses[ename]))
            make_dict_default(self.accs, ename, dict())
            make_dict_default(self.accs[ename], split, [])
            for j in range(len(sample_kept[ename])):
                self.accs[ename][split].append(sample_kept[ename][j]['acc'])
            make_dict_default(self.agg_accs, ename, dict())
            make_dict_default(self.agg_accs[ename], split, [])
            for k in range(n_groups):
                self.agg_accs[ename][split].append(group_kept[ename][k]['acc'])

        if last_test:  # last test
            for k in range(n_groups):
                if 'ours' in evaluators and not correct[k]:  # 'ours' in evaluators and not correct[k]:
                    all_store = dict(
                        agg_output=group_kept['ours'][k]['agg_output'],
                        agg_output_label=group_kept['ours'][k]['agg_output_label'],
                        agg_scores=group_kept['ours'][k]['agg_scores'],
                        target=group_kept['ours'][k]['target'],
                        target_label=group_kept['ours'][k]['target_label'],
                        probs=[sample_kept['ours'][j]['probs'] for j in range(start_indices[k], end_indices[k])],
                        paths=[sample_kept['ours'][j]['paths'] for j in range(start_indices[k], end_indices[k])],
                        qvalues=[sample_kept['ours'][j]['qvalues'] for j in range(start_indices[k], end_indices[k])],
                    )
                    if 'trueip' in evaluators:
                        all_store.update(
                            true_probs=[
                                sample_kept['trueip'][j]['probs'] for j in range(start_indices[k], end_indices[k])
                            ],
                            true_paths=[
                                sample_kept['trueip'][j]['paths'] for j in range(start_indices[k], end_indices[k])
                            ],
                            true_qvalues=[
                                sample_kept['trueip'][j]['qvalues'] for j in range(start_indices[k], end_indices[k])
                            ],
                        )
                    self.collect_data_from_batch('wrong_sample', batch, k, batch_id=batch_id, sample_id=k, **all_store)
            self.dump_stored_data()

        # print('losses', losses)

        # self._debug_visualize('toyrobot', evaluators['ours'].env_args, evaluators['ours'], trajs[0][0])
        #
        # point_net = evaluators['ours'].init_values.init_value.net['eff_Ball']
        # point_net[0].bias.requires_grad_(False)
        # from torch.optim import SGD
        # from torch import nn
        # optm = SGD(filter(lambda p: p.requires_grad, point_net.parameters()), lr=3e-3)
        # optm.zero_grad()
        # losses['ours'].backward()
        # nn.utils.clip_grad_norm_(
        #     filter(lambda p: p.requires_grad, point_net.parameters()), 10
        # )
        # optm.step()
        # self._debug_visualize('toyrobot', evaluators['ours'].env_args, evaluators['ours'], trajs[0][0])
        # losses['ours'] = torch.zeros(1).to(self.device)
        return losses

    def _debug_visualize(self, env_name, env_args, qvalue_evaluator, init_symbolic_state):
        from hacl.envs.simple_continuous.playroom_gdk.broadcast_engine import ToyRobotBroadcastEngine
        from hacl.envs.simple_continuous.playroom_gdk.toyrobot_v20210423 import ToyRobotV20210423
        acts = ['eff_Ball']
        init_benv = ToyRobotBroadcastEngine(env_args)
        init_benv.env.load_from_symbolic_state(init_symbolic_state)
        qvalue_evaluator.init_values.visualize_acts(
            acts,
            env_args=ToyRobotV20210423.complete_env_args(env_args),
            save_dir=None,
            setup='toy_robot',
            init_benv=init_benv
        )

    def batch_plan(
        self, batch_id, split, batch, traj_checker, policy=None, *args, **kwargs
    ):
        evaluators = self.evaluators
        assert split == 'test'
        n_groups = len(batch)
        env_args = self.data_loader.get_env_args()
        env = self.data_loader.build_env_from_args(env_args)
        start_states = []
        batch_labels = []
        for (group, label) in batch:
            start_states.extend(traj[0][0] for traj in group['trajs'])
            batch_labels.extend([label] * len(group['trajs']))

        for ename, evaluator in evaluators.items():
            results = evaluator.plan(start_states, batch_labels, policy=policy, **kwargs)
            if ename == 'ours':
                trajs, graphs, rrts, target_configs_list = (
                    [res['traj'] for res in results],
                    [res['graph'] for res in results],
                    [res['rrt'] for res in results],
                    [res['target_configs'] for res in results],
                )
            else:
                trajs = [res['traj'] for res in results]
                target_configs_list = rrts = [None] * len(start_states)

            for start_state, label, traj, rrt, target_configs in zip(start_states, batch_labels, trajs, rrts, target_configs_list):
                done, progress = traj_checker(traj, label, start_state=start_state)
                if ename not in self.planning_results:
                    self.planning_results[ename] = []
                self.planning_results[ename].append(
                    dict(start_state=start_state, label=label, traj=traj, done=done, progress=progress)
                )

                # if not done:
                #     print(label)
                #     for state in traj[0]:
                #         print(state)
                print('Correct' if done else 'Fail')
                print('current acc is', average([int(result['done']) for result in self.planning_results[ename]]))
                print('current avg_prog is', average([(result['progress']) for result in self.planning_results[ename]]))
                # if ename == 'ours' and label == 'Ball':
                #     path = [state[0] for state in traj[0]]
                #
                #     env.load_from_symbolic_state(start_state)
                #     visualize_problem_and_solution(env.pspace, path=[target_configs['eff_' + label]], window=label + ('done' if done else 'fail'), play_step=False)
                #     visualize_problem_and_solution(env.pspace, path=path, rrt=rrt, window=label + ('done' if done else 'fail'), play_step=True)

    def epoch_end(self, epoch, plan=False, save_answer=None, debug=False):
        meter_dict = dict()
        epoch_answer = dict()
        for ename in self.accs:
            for split in self.accs[ename]:
                for metric in self.accs[ename][split][0].keys():
                    metric_list = list(ad[metric] for ad in self.accs[ename][split])
                    metric_full_name = ename + '_' + split + '_acc_' + metric
                    meter_dict[metric_full_name] = average(metric_list) if not metric.startswith('medium') else medium(metric_list)
        for ename in self.agg_accs:
            for split in self.agg_accs[ename]:
                for metric in self.agg_accs[ename][split][0].keys():
                    metric_list = list(ad[metric] for ad in self.agg_accs[ename][split])
                    metric_full_name = ename + '_' + split + '_agg_acc_' + metric
                    meter_dict[metric_full_name] = average(metric_list) if not metric.startswith('medium') else medium(metric_list)
                    if split == 'test':
                        epoch_answer[metric_full_name] = meter_dict[metric_full_name]
        for ename in self.losses:
            for split in self.losses[ename]:
                meter_dict[ename + '_' + split + '_' + 'loss'] = average(self.losses[ename][split])
        if plan:
            for ename in self.planning_results:
                meter_dict[ename + '_planning_acc'] = average(
                    [int(result['done']) for result in self.planning_results[ename]]
                )
                meter_dict[ename + '_planning_avg_prog'] = average(
                    [result['progress'] for result in self.planning_results[ename]]
                )
                epoch_answer[ename + '_planning_acc'] = meter_dict[ename + '_planning_acc']
        self.meters.update(**meter_dict)
        if self.use_tb and hasattr(self.meters, 'flush'):
            self.meters.flush()
        self.logger(self.meters.format_simple('Epoch {}'.format(epoch), values='val'))
        for cmname, matrix in self.confusing_matrices.items():
            split = cmname.split(' ')[-1]

            dump_confusing_matrix(
                matrix,
                'Confusing Matrix: ' + cmname,
                self.labels[split],
                self.logger,
                plot=True,
                folder=os.path.join(self.save_dir, 'plots', self.data_loader.data_version),
                filename=cmname + '.png',
            )
        self.last_epoch_answer = epoch_answer
        self.last_epoch_stored_samples = self.kept_groups
        self.last_epoch_planning_results = self.planning_results

        if save_answer is not None:
            import json
            import pickle
            print('Answer=', json.dumps(self.last_epoch_answer, indent=2))
            json.dump(self.last_epoch_answer, open(os.path.join(self.save_dir, save_answer + '.json'), 'w'), indent=2)
            dump_dict = dict(
                answer=self.last_epoch_answer,
                stored_samples=self.last_epoch_stored_samples,
                planning_results=self.last_epoch_planning_results
            )
            pickle.dump(dump_dict, open(os.path.join(self.save_dir, save_answer + '.pkl'), 'wb'), protocol=2)

    def summary(self, plot_wrong_sample=None, load_stored_data=False, local=True, **kwargs):
        if load_stored_data:
            self.load_stored_data()
        else:
            self.dump_stored_data()
        n_wrong_samples = (
            input("Please enter the number of wrong samples to plot")
            if plot_wrong_sample is None
            else plot_wrong_sample
        )

        for label in self.labels['test']:
            self.logger("%s:   %s" % (label, self.label2state_machine[label]))
        self.print_collected_data_by_index(
            'wrong_sample',
            slice(0, int(n_wrong_samples)),
            note_keys=[
                'batch_id',
                'sample_id',
                'label',
                'agg_output_label',
                'agg_scores',
            ],
            traj_note_keys=['probs', 'paths', 'true_probs', 'true_paths'],
            log=self.logger,
            local=local,
        )

    def dump_stored_data(self):
        stored_data_filename = os.path.join(self.save_dir, 'stored_data.pkl')
        pickle.dump(self.stored_data, open(stored_data_filename, 'wb'), protocol=2)

    def load_stored_data(self):
        stored_data_filename = os.path.join(self.save_dir, 'stored_data.pkl')
        self.stored_data = pickle.load(open(stored_data_filename, 'rb'))

    def collect_data_from_batch(self, data_type, batch, index_in_batch, **kwargs):
        new_sample = self.data_loader.collect_data(batch, index_in_batch, **kwargs)
        self.stored_data.append({'data': new_sample, 'data_type': data_type})

    def print_collected_data(self, sample, note_keys=None, traj_note_keys=None, log=print, local=True):
        self.data_loader.print_data(
            sample, note_keys, traj_note_keys,
            log=log,
            local=local,
            qvalue_evaluator=self.evaluators['ours'] if 'ours' in self.evaluators else None,
            save_dir=os.path.join(self.save_dir, 'visualize', 'samples')
        )

    def print_collected_data_by_index(
        self, data_type, indices, note_keys=None, traj_note_keys=None, log=print, local=True
    ):
        if isinstance(data_type, str):
            data_type = [data_type]
        filtered_data = [data['data'] for data in self.stored_data if data['data_type'] in data_type]
        if isinstance(indices, slice):
            for sample in filtered_data[indices]:
                self.print_collected_data(
                    sample, note_keys=note_keys, traj_note_keys=traj_note_keys, log=log, local=local
                )
        else:
            for k in indices:
                self.print_collected_data(
                    filtered_data[k], note_keys=note_keys, traj_note_keys=traj_note_keys, log=log, local=local
                )
