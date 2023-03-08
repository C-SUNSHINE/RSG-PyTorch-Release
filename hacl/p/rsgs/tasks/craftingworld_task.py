#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle

import numpy as np
import torch

from hacl.algorithms.value_based_classification import QValueBasedActionClassifier
from hacl.envs.gridworld.crafting_world.broadcast_engine import CraftingWorldBroadcastEngine
from hacl.envs.gridworld.crafting_world.engine.engine import CraftingWorldActions
from hacl.envs.gridworld.crafting_world.engine.rules import ALL_RULES
from hacl.envs.gridworld.crafting_world.configs import PRIMITIVES
from hacl.algorithms.planning.astar_planning_config import SKILL_V_RANGE
from hacl.p.rsgs.utils import average, dump_confusing_matrix, get_acc_dict, make_dict_default, medium
from .task_pipeline import TaskPipeline

Actions = CraftingWorldActions


class CraftingWorldTask(TaskPipeline):
    def __init__(
        self, data_loader, label2state_machine, meters=None, save_dir=None, log=print, use_tb=False, ptrajonly=False, *args, **kwargs
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
            split: QValueBasedActionClassifier(
                self.data_loader.labels[split],
                label2state_machine,
                broadcast_engine=CraftingWorldBroadcastEngine(object_limit=30),
                ptrajonly=ptrajonly,
            )
            for split in ('train', 'val', 'test')
        }
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.last_epoch_answer = None
        self.last_epoch_stored_samples = None
        self.last_epoch_groups = None
        self.last_epoch_planning_results = None

    def init(self):
        pass

    def epoch_init(self, epoch):
        self.losses = {}
        self.sub_losses = {}
        self.accs = {}
        self.agg_accs = {}
        self.confusing_matrix_configs = [
            ('trueip', 'ours'),
        ]
        self.confusing_matrices = {}
        self.stored_data = []
        self.stored_samples = {}
        self.planning_results = {}
        self.goal_classifier_tests = []
        self.dependency_score = {l: {l2: 0. for l2 in PRIMITIVES} for l in PRIMITIVES}
        self.final_value = {l: 0 for l in PRIMITIVES}
        self.avg_value = {l: 0 for l in PRIMITIVES}

    def get_model_args(self, evaluators, split, action_set):
        from collections import defaultdict

        model_args = defaultdict(lambda: dict())
        if 'ours' in evaluators:
            model_args.update(
                ours=dict(
                    action_set=action_set,
                    n_iters=1 if split == 'test' else 1,
                    n_epochs=4,
                    brute_depth=(4 if split == 'test' else 3),
                    explore_depth=25 if split == 'test' else 15,
                    shallow_filter=1,
                    flatten_tree=False,
                    max_branches=10,
                )
            )
        if 'trueip' in evaluators:
            model_args.update(
                trueip=dict(
                    action_set=action_set,
                )
            )
        return model_args

    def _get_zero_confusing_matrix(self, split, dtype=np.int):
        return np.zeros((self.n_labels[split], self.n_labels[split]), dtype=dtype)

    def batch_run(
        self, batch_id, split, batch, last_test=False, train_choice=False, train_choice_adjacent=False, train_binomial=False, progress=None, *args, **kwargs
    ):
        evaluators = self.evaluators
        alpha = kwargs['alpha'] if 'alpha' in kwargs else 1.
        beta = kwargs['beta'] if 'beta' in kwargs else 0.
        training = split == 'train'
        n = len(batch)
        trajs = []
        answer_labels_for_training = []
        for sample in batch:
            trajs.append(sample['traj'])
            answer_labels_for_training.append(sample['label'])

        if train_binomial and split != 'test':
            batch_label_cnt = {label: sum((1 if sample['label'] == label else 0) for sample in batch) for label in self.labels[split]}
            batch_label = max(self.labels[split], key=lambda x: batch_label_cnt[x])
            labels = [batch_label]
        elif train_choice is not None and split != 'test':
            import random
            assert train_choice <= len(self.labels[split])
            batch_label_set = set(sample['label'] for sample in batch)
            if len(batch_label_set) == 1:
                if not train_choice_adjacent:
                    batch_label = batch[0]['label']
                    negative_labels = [label for label in self.labels[split] if label != batch_label]
                    random.shuffle(negative_labels)
                    assert train_choice <= len(negative_labels) + 1
                    labels = [batch_label] + negative_labels[: train_choice - 1]
                    random.shuffle(labels)
                else:
                    batch_label = batch[0]['label']
                    pos = list(filter(lambda i: self.labels[split][i] == batch_label, range(len(self.labels[split]))))[0]
                    start_pos = len(self.labels[split]) + pos - (train_choice - 1) // 2
                    labels = (self.labels[split] * 3)[start_pos:start_pos + train_choice]
                    print(labels, batch_label)
            else:
                labels = list(sorted(batch_label_set))
                label_cnt = {label: sum((1 if sample['label'] == label else 0) for sample in batch) for label in labels}
                print(label_cnt)
        else:
            labels = self.labels[split]
        label2index = {l: i for i, l in enumerate(labels)}
        n_labels = len(labels)

        action_set = (Actions.Up, Actions.Down, Actions.Left, Actions.Right, Actions.Toggle)

        evaluations = {}

        model_args = self.get_model_args(evaluators, split, action_set)

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
        correctness = {ename: [] for ename in evaluators}
        sub_losses = {ename: {} for ename in evaluators}
        kept = {ename: [] for ename in evaluators}
        correct_ours = []

        for i, sample in enumerate(batch):
            states, actions = sample['traj']
            label = sample['label']
            tar = label2index[label] if label in label2index else None
            for ename, evaluator in evaluators.items():
                if evaluator.qvalue_based:
                    raw_qvalues = evaluations[ename]
                    qvalues = [raw_qvalues[l][i].to(self.device) for l in range(n_labels)]
                    scores, paths, step_values = self.qvalue_classifier[split].prob(
                        states,
                        actions,
                        action_set,
                        qvalues,
                        evaluator.init_values,
                        labels=labels,
                        record_path=True,
                        record_step_value=True,
                        device=self.device,
                    )
                    log_probs = torch.log_softmax(scores, dim=0)
                    probs = torch.softmax(log_probs, dim=0)
                    output = max(range(n_labels), key=lambda x: scores[x])

                    kept[ename].append(
                        dict(
                            scores=scores.detach().cpu(),
                            probs=probs.detach().cpu(),
                            paths=paths,
                            step_values=step_values,
                            qvalues=[x.detach().cpu() for x in qvalues],
                            output=output,
                            output_label=labels[output],
                            target=tar,
                            target_label=label,
                            ungroup=(train_choice or train_binomial) and split != 'test',
                        )
                    )
                else:
                    scores = evaluations[ename][i]
                    log_probs = torch.log_softmax(scores, dim=0)
                    probs = torch.softmax(log_probs, dim=0)
                    output = max(range(n_labels), key=lambda x: scores[x])

                    kept[ename].append(
                        dict(
                            scores=scores.detach().cpu(),
                            probs=probs.detach().cpu(),
                            output=output,
                            output_label=labels[output],
                            target=tar,
                            target_label=label,
                            ungroup=(train_choice or train_binomial) and split != 'test',
                        )
                    )

                if train_binomial and split != 'test':
                    if label == batch_label:
                        make_dict_default(sub_losses[ename], 'positive', [])
                        sub_losses[ename]['positive'].append(-scores[0])
                        losses[ename].append(-scores[0])
                    else:
                        make_dict_default(sub_losses[ename], 'negative', [])
                        sub_losses[ename]['negative'].append(scores[0])
                        losses[ename].append(scores[0])
                    correctness[ename].append(float(scores[0]))
                else:
                    make_dict_default(sub_losses[ename], 'alpha', [])
                    make_dict_default(sub_losses[ename], 'beta', [])
                    sub_losses[ename]['alpha'].append(- alpha * torch.log_softmax(scores, dim=0)[tar])
                    sub_losses[ename]['beta'].append(- beta * scores[tar])
                    losses[ename].append(sub_losses[ename]['alpha'][-1] + sub_losses[ename]['beta'][-1])
                    correctness[ename].append(labels[output] == label)

            correct_ours.append('ours' in kept and kept['ours'][-1]['output'] == tar)
            # Update Confusing Matrix:
            for row, col in list(('@', ename) for ename in evaluators) + self.confusing_matrix_configs:
                if (row != '@' and row not in evaluators.keys()) or col not in evaluators.keys():
                    continue
                for cmmode in ('prob', 'single'):
                    cmname = ("Answer" if row == '@' else row) + " vs " + col + ' -' + cmmode + ' :  ' + split
                    if cmname not in self.confusing_matrices:
                        self.confusing_matrices[cmname] = self._get_zero_confusing_matrix(
                            split, dtype=np.float if cmmode == 'prob' else np.int
                        )
                    if cmmode == 'prob':
                        row_label = label if row == '@' else kept[row][-1]['output_label']
                        for col_label_id in range(n_labels):
                            col_label = labels[col_label_id]
                            pair_prob = kept[col][-1]['probs'][col_label_id]
                            self.confusing_matrices[cmname][
                                self.label2index[split][row_label], self.label2index[split][col_label]
                            ] += pair_prob
                    elif cmmode == 'single':
                        row_label = label if row == '@' else kept[row][-1]['output_label']
                        col_label = kept[col][-1]['output_label']
                        self.confusing_matrices[cmname][
                            self.label2index[split][row_label], self.label2index[split][col_label]
                        ] += 1

            # Store sample
            for ename in evaluators:
                if ename not in self.stored_samples:
                    self.stored_samples[ename] = {}
                self.stored_samples[ename][sample['index']] = self.data_loader.collect_data(batch, i, **kept[ename][-1])
            if split == 'test' and 'ours' in kept:
                if correct_ours[-1]:
                    print("Correct!")
                else:
                    print("Wrong! Expected %s, got %s" % (kept['ours'][-1]['target_label'], kept['ours'][-1]['output_label']))
                    print(kept['ours'][-1]['scores'])

        # Average statistic over batches
        losses = {ename: average(loss) for ename, loss in losses.items()}
        sub_losses = {ename: {name: average(loss) for name, loss in sub_loss.items()} for ename, sub_loss in sub_losses.items()}
        if train_binomial and split != 'test':
            for ename in correctness:
                ranklist = list(sorted(range(n), key=lambda i: correctness[ename][i], reverse=True))
                for rk, i in enumerate(ranklist):
                    correctness[ename][i] = (rk < n // 2)
                for i, sample in enumerate(batch):
                    correctness[ename][i] = (correctness[ename][i] == (sample['label'] == batch_label))
        # Get statistic of loss and accuracy
        for ename in evaluators:
            if ename in losses:
                if ename not in self.losses:
                    self.losses[ename] = dict()
                    self.sub_losses[ename] = dict()
                if split not in self.losses[ename]:
                    self.losses[ename][split] = []
                    self.sub_losses[ename][split] = dict()
                self.losses[ename][split].append(float(losses[ename]))
                for sub_name, sub_loss in sub_losses[ename].items():
                    if sub_name not in self.sub_losses[ename][split]:
                        self.sub_losses[ename][split][sub_name] = []
                    self.sub_losses[ename][split][sub_name].append(float(sub_loss))
            if ename not in self.accs:
                self.accs[ename] = dict()
            if split not in self.accs[ename]:
                self.accs[ename][split] = []
            for k in range(n):
                self.accs[ename][split].append(1 if correctness[ename][k] else 0)
        return losses

    def batch_test_goal_classifier(self, batch_id, split, batch, traj_checker):
        assert 'ours' in self.evaluators
        evaluator = self.evaluators['ours']
        res = []
        for sample in batch:
            for state in sample['traj'][0]:
                for skill in PRIMITIVES:
                    out = evaluator.test_goal_classifier([state], skill).cpu().item()
                    ans = traj_checker.state_goal_predicate(state, skill)
                    if out is not None:
                        res.append((skill, float(out), float(ans)))
        self.goal_classifier_tests.extend(res)

    def batch_get_dependencies(self, batch_id, split, batch):
        import math
        assert 'ours' in self.evaluators
        evaluator = self.evaluators['ours']
        for sample in batch:
            states = sample['traj'][0]
            evals = {}
            progs = {}
            for skill in PRIMITIVES:
                out = evaluator.test_goal_classifier(sample['traj'][0], skill).cpu()
                # if math.exp(out[-1])-math.exp(out[0]) > 0.1:
                #     print(skill, [math.exp(out[i]) for i in range(len(states))])
                # evals[skill] = [min(1.0, math.exp(out[i] - SKILL_V_RANGE[skill][1])) for i in range(len(states))]
                evals[skill] = [1 if out[i] >= (SKILL_V_RANGE[skill][0] + SKILL_V_RANGE[skill][1] / 2) else 0 for i in
                                range(len(states))]
                progs[skill] = []
                mx = 0
                for i in range(len(evals[skill])):
                    nmx = max(mx, evals[skill][i])
                    progs[skill].append(nmx - mx)
                    mx = nmx
                self.final_value[skill] += mx
                self.avg_value[skill] += sum(evals[skill])

            for skill1 in PRIMITIVES:
                for skill2 in PRIMITIVES:
                    if skill1 not in sample['label'] or skill2 not in sample['label']:
                        continue
                    version_now = 'v1'
                    if version_now == 'v1':
                        for i1 in range(len(progs[skill1])):
                            for i2 in range(i1 + 1, len(progs[skill2])):
                                self.dependency_score[skill2][skill1] += progs[skill1][i1] * progs[skill2][i2]
                    elif version_now == 'v2':
                        for p1, p2 in zip(evals[skill1][:-1], progs[skill2][1:]):
                            self.dependency_score[skill2][skill1] += p1 * p2

    def batch_plan(self, batch_id, split, batch, traj_checker, plan_search=False, search_iter=None, *args, **kwargs):
        evaluators = self.evaluators
        assert split == 'test'

        action_set = traj_checker.action_set
        constraints = []
        batch_labels = []
        for sample in batch:
            constraints.append(sample['traj'][0][0] if not plan_search else (sample['traj'][0][0], traj_checker.get_terminate_checker(sample['label'])))
            batch_labels.append(sample['label'])
            assert traj_checker(sample['traj'], sample['label'], sample['traj'][0][0])[0]

        results = {}

        for ename, evaluator in evaluators.items():
            results[ename] = evaluator.plan(
                constraints,
                batch_labels,
                action_set=action_set,
                prune=False,
                n_iters=search_iter or 5000,
                plan_search=plan_search,
                **kwargs
            )
            for constraint, label, result in zip(constraints, batch_labels, results[ename]):
                if plan_search:
                    start_state, terminate_checker = constraint
                else:
                    start_state, end_state = constraint, None
                traj = result['traj']
                done, progress = traj_checker(traj, label, start_state=start_state)
                if ename not in self.planning_results:
                    self.planning_results[ename] = []
                planning_result = dict(start_state=start_state, label=label, traj=traj, done=done, progress=progress)
                for key in ['search_count']:
                    if key in result:
                        planning_result[key] = result[key]
                self.planning_results[ename].append(planning_result)
                if not done:
                    print(label)
                    for i in range(len(traj[1])):
                        print(traj[0][i], traj[1][i])
                    print(traj[0][-1])
                print('Label=', label)
                print('Correct' if done else 'Fail',
                      'progress=', round(progress, 2),
                      'search_count=', result['search_count'] if 'search_count' in result else None
                      )
                print('current acc is', average([int(res['done']) for res in self.planning_results[ename]]))
                print('current avg_prog is', average([(res['progress']) for res in self.planning_results[ename]]))
                if 'search_count' in result:
                    print('current avg_search_count is', average([(res['search_count']) for res in self.planning_results[ename]]))
                if not done:
                    pass
                    # KeysDoorsV1.render_data_human(
                    #     [{'label': label, 'start_state': start_state, 'traj': traj}],
                    #     env_args=self.data_loader.get_env_args(),
                    # )

    def compute_groups(self, split, groups):
        enames = list(self.stored_samples.keys())
        for group, label in groups:
            incomplete = False
            target_label = label
            target = self.label2index[split][label]
            outputs = {}
            output_labels = {}
            for ename in enames:
                agg_scores = torch.zeros(self.n_labels[split])
                for index in group:
                    if index not in self.stored_samples[ename] or self.stored_samples[ename][index]['ungroup']:
                        incomplete = True
                        break
                    sample = self.stored_samples[ename][index]
                    agg_scores += sample['scores']
                if incomplete:
                    break
                output, acc_dict = get_acc_dict(agg_scores, target)
                outputs[ename] = output
                output_labels[ename] = self.labels[split][outputs[ename]]

                # Updade accuracy:
                if ename not in self.agg_accs:
                    self.agg_accs[ename] = dict()
                if split not in self.agg_accs[ename]:
                    self.agg_accs[ename][split] = []
                self.agg_accs[ename][split].append(acc_dict)
            if incomplete:
                continue

            # Update Confusing Matrix:
            for row, col in list(('@', ename) for ename in enames) + self.confusing_matrix_configs:
                if (row != '@' and row not in enames) or col not in enames:
                    continue
                for cmmode in ('agg',):
                    if cmmode == 'agg' and self.data_loader.group_size(split) == 1:
                        continue
                    cmname = ("Answer" if row == '@' else row) + " vs " + col + ' -' + cmmode + ' :  ' + split
                    if cmname not in self.confusing_matrices:
                        self.confusing_matrices[cmname] = self._get_zero_confusing_matrix(
                            split, dtype=np.float if cmmode == 'prob' else np.int
                        )
                    if cmmode == 'agg':
                        row_label = target_label if row == '@' else output_labels[row]
                        col_label = output_labels[col]
                        self.confusing_matrices[cmname][
                            self.label2index[split][row_label], self.label2index[split][col_label]
                        ] += 1

    def epoch_end(self, epoch, plan=False, debug=False, save_answer=None, test_goal_classifier=False, get_dependencies=False, **kwargs):

        if test_goal_classifier:
            import math, json
            res_by_skill = {}
            for e in self.goal_classifier_tests:
                s = e[0]
                if s not in res_by_skill:
                    res_by_skill[s] = []
                res_by_skill[s].append((e[1], e[2]))
            tot_acc = []
            skill_v_range = {}
            for skill, cases in res_by_skill.items():
                cases = list(sorted(cases, key=lambda x: x[0]))
                tot = len(cases)
                tot1 = sum(1 if x[1] else 0 for x in cases)
                cnt1 = 0
                correct = 0
                splitting = None
                for i in range(len(cases) + 1):
                    cor_neg = i - cnt1
                    cor_pos = tot1 - cnt1
                    cor_all = (cor_neg / (tot - tot1) + cor_pos / tot1) / 2
                    if cor_all > correct:
                        correct = cor_all
                        splitting = cases[i][0] if i == 0 else (cases[i - 1][0] if i == len(cases) else (cases[i - 1][0] + cases[i][0]) / 2)
                    if i < len(cases) and cases[i][1]:
                        cnt1 += 1

                # correct = sum([(math.exp(c[0]) >= 1) == c[1] for c in cases])

                acc = correct
                print("Skill = %s, acc = %.2f%%, spliting = %.20lf%%, value_range = (%lf, %lf)" % (skill, acc * 100, splitting, cases[0][0], cases[-1][0]))
                skill_v_range[skill] = (cases[0][0], cases[-1][0], splitting)
                tot_acc.append(acc)
            print('Avg_Acc = %.2lf%%' % (sum(tot_acc) / len(tot_acc) * 100, ))
            print(json.dumps(skill_v_range))
            exit()

        if get_dependencies:
            fout = open(os.path.join(self.save_dir, 'dependencies_score.txt'), 'w')
            fout.write('DATA = '+str(self.dependency_score))
            fout.write('\n')
            fout.write('SUM = '+str(self.final_value))
            fout.write('\n')
            fout.write('AVG = '+str(self.avg_value))
            fout.close()
            exit()

        for split in ('train', 'val', 'test'):
            self.last_epoch_groups = list(self.data_loader.groups(split=split))
            self.compute_groups(split, groups=self.last_epoch_groups)
        meter_dict = dict()
        epoch_answer = dict()
        for ename in self.agg_accs:
            for split in self.agg_accs[ename]:
                for metric in self.agg_accs[ename][split][0].keys():
                    metric_list = list(ad[metric] for ad in self.agg_accs[ename][split])
                    metric_full_name = ename + '_' + split + '_agg_acc_' + metric
                    meter_dict[metric_full_name] = average(metric_list) if not metric.startswith('medium') else medium(metric_list)
                    if split == 'test':
                        epoch_answer[metric_full_name] = meter_dict[metric_full_name]
        for ename in self.accs:
            for split in self.accs[ename]:
                meter_dict[ename + '_' + split + '_' + 'acc'] = average(self.accs[ename][split])
        for ename in self.losses:
            for split in self.losses[ename]:
                meter_dict[ename + '_' + split + '_' + 'loss'] = average(self.losses[ename][split])
        for ename in self.sub_losses:
            for split in self.sub_losses[ename]:
                for sub_name in self.sub_losses[ename][split]:
                    meter_dict[ename + '_' + split + '_' + 'loss-' + sub_name] = average(self.sub_losses[ename][split][sub_name])
        if plan:
            for ename in self.planning_results:
                meter_dict[ename + '_planning_acc'] = average(
                    [int(result['done']) for result in self.planning_results[ename]]
                )
                meter_dict[ename + '_planning_avg_prog'] = average(
                    [result['progress'] for result in self.planning_results[ename]]
                )
                epoch_answer[ename + '_planning_acc'] = meter_dict[ename + '_planning_acc']
                if any('search_count' in result for result in self.planning_results[ename]):
                    meter_dict[ename + '_planning_search_count'] = average(
                        [int(result['search_count']) for result in self.planning_results[ename] if 'search_count' in result]
                    )
                    epoch_answer[ename + '_planning_search_count'] = meter_dict[ename + '_planning_search_count']

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
        self.last_epoch_stored_samples = self.stored_samples
        self.last_epoch_planning_results = self.planning_results


        if save_answer is not None:
            import json
            import pickle
            print('Answer=', json.dumps(self.last_epoch_answer, indent=2))
            json.dump(self.last_epoch_answer, open(os.path.join(self.save_dir, save_answer + '.json'), 'w'), indent=2)
            dump_dict = dict(
                answer=self.last_epoch_answer,
                stored_samples=self.last_epoch_stored_samples,
                groups=self.last_epoch_groups,
                planning_results=self.last_epoch_planning_results
            )
            pickle.dump(dump_dict, open(os.path.join(self.save_dir, save_answer + '.pkl'), 'wb'), protocol=2)

    def dump_stored_data(self):
        stored_data_filename = os.path.join(self.save_dir, 'stored_data.pkl')
        pickle.dump(self.stored_data, open(stored_data_filename, 'wb'), protocol=2)

    def load_stored_data(self):
        stored_data_filename = os.path.join(self.save_dir, 'stored_data.pkl')
        self.stored_data = pickle.load(open(stored_data_filename, 'rb'))

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
        try:
            for label in self.labels['test']:
                self.logger("%s:   %s" % (label, self.label2state_machine[label]))
            self.print_collected_data_by_index(
                'wrong_sample',
                slice(0, int(n_wrong_samples)),
                note_keys=[
                    'batch_id',
                    'sample_id',
                    'label',
                    'output',
                    'scores',
                ],
                traj_note_keys=['probs', 'paths', 'true_probs', 'true_paths'],
                log=self.logger,
                local=local,
            )
        except Exception:
            print('Print Wrong Example Failed')
            pass
        finally:
            pass

    def collect_data_from_batch(self, data_type, batch, index_in_batch, **kwargs):
        new_sample = self.data_loader.collect_data(batch, index_in_batch, **kwargs)
        self.stored_data.append({'data': new_sample, 'data_type': data_type})

    def print_collected_data(self, sample, note_keys=None, traj_note_keys=None, log=print, visualize=True, local=True):
        self.data_loader.print_data(sample, note_keys, traj_note_keys, log=log, local=local)
        if visualize:
            self.visualize(sample, log=log)

    def print_collected_data_by_index(
        self, data_type, indices, note_keys=None, traj_note_keys=None, log=print, visualize=True, local=True
    ):
        if isinstance(data_type, str):
            data_type = [data_type]
        filtered_data = [data['data'] for data in self.stored_data if data['data_type'] in data_type]
        if isinstance(indices, slice):
            for sample in filtered_data[indices]:
                self.print_collected_data(
                    sample,
                    note_keys=note_keys,
                    traj_note_keys=traj_note_keys,
                    log=log,
                    visualize=visualize,
                    local=local,
                )
        else:
            try:
                for k in indices:
                    self.print_collected_data(
                        filtered_data[k],
                        note_keys=note_keys,
                        traj_note_keys=traj_note_keys,
                        log=log,
                        visualize=visualize,
                        local=local,
                    )
            except:
                self.print_collected_data(
                    filtered_data[indices],
                    note_keys=note_keys,
                    traj_note_keys=traj_note_keys,
                    log=log,
                    visualize=visualize,
                    local=local,
                )
