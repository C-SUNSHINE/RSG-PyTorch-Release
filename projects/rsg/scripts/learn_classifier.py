#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import time

import jactorch
import torch
import torch.nn as nn
from jaclearn.mldash import MLDashClient
from tqdm import tqdm

from hacl.models.rsgs.evaluators import *
from hacl.models.rsgs.state_machine.builders import (get_compact_state_machine_from_label, get_incompact_state_machine_from_label)
from hacl.p.rsgs.representation_learner import RepresentationLearner
from hacl.p.rsgs.tasks.craftingworld_task import CraftingWorldTask
from hacl.p.rsgs.tasks.toyrobot_task import ToyRobotTask
from hacl.utils.logger import FileLogger


def average(x):
    if len(x) == 0:
        return -1
    else:
        return sum(x) / len(x)


def default_lr_schedule(epoch, n_epochs=None, **kwargs):
    if n_epochs is None:
        if epoch <= 6:
            return 1.0
        elif epoch <= 15:
            return 0.1
        return 0.01
    else:
        if epoch * 10 <= n_epochs * 5:
            return 1.0
        elif epoch * 10 <= n_epochs * 8:
            return 0.1
        else:
            return 0.01


def load_model_state(save_dir, name):
    filename = os.path.join(save_dir, name + '.pth')
    return torch.load(filename)


def save_model_state(model, save_dir, name):
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, name + '.pth')
    torch.save(model.state_dict(), filename)


class NeuralAstarClassifierLearner(RepresentationLearner):
    def __init__(self):
        super().__init__()

    def set_data_loader(self, data_loader):
        self.data_loader = data_loader
        self.labels = self.data_loader.labels

    def log(self, log_fout, message, end='\n'):
        print(message, end=end)
        log_fout.write(message + end)
        log_fout.flush()

    @classmethod
    def get_label2state_machine(self, labels, state_machine_type):
        label2state_machine = {}
        for label in labels:
            if state_machine_type == 'compact':
                label2state_machine[label] = get_compact_state_machine_from_label(label)
            elif state_machine_type == 'incompact':
                label2state_machine[label] = get_incompact_state_machine_from_label(label)
        return label2state_machine

    @classmethod
    def get_pipeline(cls, env_name):
        if env_name == 'toyrobot':
            return ToyRobotTask
        elif env_name == 'craftingworld':
            return CraftingWorldTask
        else:
            raise ValueError()

    def get_evaluators(
        self,
        env_name,
        dataset_name,
        env_args,
        ptrajonly=False,
        use_not_goal=False,
        use_true_value=None,
        deep_search_tree=False,
        state_machine_type=None,
        action_cost=None,
        toy_robot_net=None,
        search_unique=None,
        all_labels=None,
        skip_our_model=False,
        trueip=False,
        lstm=False,
        irl_classify_by_reward=False,
        dqn=False,
        lstmdqn=False,
        a2c=False,
        lstma2c=False,
        irldqn=False,
        irllstmdqn=False,
        irla2c=False,
        irllstma2c=False,
        seq2seq=False,
        bc=False,
        irlcont=False,
        **kwargs
    ):
        evaluators = {}
        if env_name == 'toyrobot':
            if not skip_our_model:
                evaluators['ours'] = GraphEvaluator(
                    env_args,
                    use_true_init=use_true_value,
                    state_machine_type=state_machine_type,
                    action_cost=action_cost,
                    net_type=toy_robot_net,
                    add_labels=all_labels,
                )
            if trueip:
                evaluators['trueip'] = GraphTrueEvaluator(
                    env_args,
                    state_machine_type=state_machine_type,
                    action_cost=action_cost,
                    add_labels=all_labels,
                )

        elif env_name == 'craftingworld':
            if not skip_our_model:
                evaluators['ours'] = AstarEvaluator(
                    env_args,
                    ptrajonly=ptrajonly,
                    use_not_goal=use_not_goal,
                    use_true_init=use_true_value,
                    state_machine_type=state_machine_type,
                    action_cost=action_cost,
                    setting='craftingworld',
                    add_labels=all_labels,
                    search_unique=search_unique,
                    flatten_tree=not deep_search_tree,
                )

        if lstm:
            evaluators['lstm'] = LSTMEvaluator(env_name, env_args, add_labels=all_labels)
        if dqn:
            evaluators['dqn'] = DQNEvaluator(env_name, env_args, add_labels=all_labels, use_lstm=False)
        if lstmdqn:
            evaluators['lstmdqn'] = DQNEvaluator(env_name, env_args, add_labels=all_labels, use_lstm=True)
        if a2c:
            evaluators['a2c'] = A2CEvaluator(env_name, env_args, add_labels=all_labels, use_lstm=False)
        if lstma2c:
            evaluators['lstma2c'] = A2CEvaluator(env_name, env_args, add_labels=all_labels, use_lstm=True)
        if irldqn:
            evaluators['irldqn'] = IRLEvaluator(
                env_name, env_args, add_labels=all_labels, use_lstm=False, rl_model='dqn', classify_with_reward=irl_classify_by_reward
            )
        if irllstmdqn:
            evaluators['irllstmdqn'] = IRLEvaluator(
                env_name, env_args, add_labels=all_labels, use_lstm=True, rl_model='dqn', classify_with_reward=irl_classify_by_reward
            )
        if irla2c:
            evaluators['irla2c'] = IRLEvaluator(
                env_name, env_args, add_labels=all_labels, use_lstm=False, rl_model='a2c', classify_with_reward=irl_classify_by_reward
            )
        if irllstma2c:
            evaluators['irllstma2c'] = IRLEvaluator(
                env_name, env_args, add_labels=all_labels, use_lstm=True, rl_model='a2c', classify_with_reward=irl_classify_by_reward
            )
        if seq2seq:
            evaluators['seq2seq'] = Seq2Seq_Evaluator(
                env_name, env_args, add_labels=all_labels, use_lstm=False,
            )
        if bc:
            evaluators['bc'] = BC_Evaluator(
                env_name, env_args, add_labels=all_labels, use_lstm=True
            )
        if irlcont:
            evaluators['irlcont'] = IRLContEvaluator(
                env_name, env_args, add_labels=all_labels, use_lstm=True, rl_model='dqn', classify_with_reward=irl_classify_by_reward
            )
        return evaluators

    def get_optimizer(self, optim, parameters, lr):
        if parameters is not None:
            if optim == 'adam':
                optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, parameters), lr=lr, weight_decay=1e-4)
            elif optim == 'sgd':
                optimizer = torch.optim.SGD(
                    filter(lambda p: p.requires_grad, parameters), lr=lr, momentum=0.9, weight_decay=1e-4
                )
            elif optim == 'RMSprop':
                optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, parameters), lr=lr)
            else:
                raise ValueError()
            return optimizer
        return None

    def save_evaluators(self, evaluators, best=False, save_dir=None):
        os.makedirs(os.path.join(save_dir, 'model_states'), exist_ok=True)
        for ename, evaluator in evaluators.items():
            state_dict_fullname = os.path.join(
                save_dir, 'model_states', '%s.pth' % ename if not best else '%s-best.pth' % ename
            )
            torch.save(jactorch.io.state_dict(evaluator), state_dict_fullname)

    def load_evaluators(self, evaluators, best=False, save_dir=None, force=False):
        for ename in evaluators:
            state_dict_fullname = os.path.join(
                save_dir, 'model_states', '%s.pth' % ename if not best else '%s-best.pth' % ename
            )
            try:
                state_dict = torch.load(state_dict_fullname, map_location=evaluators[ename].device)
                jactorch.io.load_state_dict(evaluators[ename], state_dict)
            except Exception as e:
                print(e)
                if best:
                    while True:
                        if force:
                            print('Load %s from %s failed! Continue testing...' % (ename, state_dict_fullname))
                            break
                        response = input(
                            'Load %s from %s failed! Continue testing? yes/no ' % (ename, state_dict_fullname)
                        )
                        if response.lower() in ['yes', 'no']:
                            if response.lower() == 'yes':
                                break
                            else:
                                exit()

                else:
                    print('Load %s from %s failed! Use initialized evaluator.' % (ename, state_dict_fullname))

    def learn_classifier(
        self,
        trial=None,
        env_name='gridworld',
        dataset_name='default',
        group=None,
        cont=False,
        save_every=1,
        save_answer=None,
        grad_clip=10,
        n_epochs=20,
        start_epoch=1,
        lr=0.03,
        batch_size=2,
        n_train_batches=None,
        n_val_batches=None,
        test_batch_size=None,
        mode='train',
        load_model=None,
        force=False,
        plan=False,
        plan_search=False,
        test_goal_classifier=False,
        get_dependencies=False,
        search_iter=False,
        policy=None,
        optim='adam',
        lr_schedule=default_lr_schedule,
        use_true_value=False,
        deep_search_tree=False,
        state_machine_type='incompact',
        alpha=None,
        beta=None,
        add_baseline_trueip=False,
        plot_wrong_sample=None,
        parsed_args=None,
        use_mldash=True,
        use_tb=False,
        use_tbx=False,
        debug=False,
        local=False,
        toy_robot_net=None,
        action_cost=None,
        train_choice=False,
        train_choice_adjacent=False,
        train_choice_uniform=False,
        train_binomial=False,
        search_unique=False,
        add_baseline_lstm=False,
        irl_classify_by_reward=False,
        skip_our_model=False,
        add_baseline_dqn=False,
        add_baseline_lstmdqn=False,
        add_baseline_a2c=False,
        add_baseline_lstma2c=False,
        add_baseline_irldqn=False,
        add_baseline_irllstmdqn=False,
        add_baseline_irla2c=False,
        add_baseline_irllstma2c=False,
        add_baseline_seq2seq=False,
        add_baseline_bc=False,
        add_baseline_irlcont=False,
        ptrajonly=False,
        test_part=None,
        test_part_label=False,
        test_part_start=None,
        test_part_end=None,
        use_not_goal=False,
    ):
        # torch.autograd.set_detect_anomaly(True)
        assert trial is not None
        assert self.data_loader is not None
        assert not plan or mode == 'test'
        save_dir = os.path.join('dumps', trial)
        training_mode = mode in ['fast-train', 'slow-train']
        os.makedirs(save_dir, exist_ok=True)
        log_file = os.path.join(save_dir, 'log.txt') if training_mode else os.path.join(save_dir, 'log_test.txt')
        tb_dir = os.path.join(save_dir, 'tb') if training_mode else None
        meter_file = os.path.join(save_dir, 'meter.txt' if training_mode else 'meter_test.txt')
        logger = FileLogger(open(log_file, 'a'), display=True)
        run_name = str(trial) + ':' + mode + ':' + time.strftime('%Y-%m-%d-%H-%M-%S')
        if use_mldash and not debug:
            try:
                mldash = MLDashClient('dumps')
                group_suffix = '' if group is None else '-' + group
                mldash.init(
                    desc_name=env_name + group_suffix + '.' + dataset_name,
                    expr_name=str(trial),
                    run_name=run_name,
                    args=parsed_args,
                )
                mldash.update(log_file=os.path.join(save_dir, 'log.txt'), meter_file=meter_file, tb_dir=tb_dir)
            except Exception:
                print('MLDash failed')
                mldash = None
        if training_mode and not debug and use_tb:
            os.makedirs(tb_dir, exist_ok=True)
            if training_mode and not cont:
                for filename in os.listdir(tb_dir):
                    os.remove(os.path.join(tb_dir, filename))
            from jactorch.train.tb import TBLogger, TBGroupMeters

            tb_logger = TBLogger(tb_dir)
            meters = TBGroupMeters(tb_logger)
        else:
            from jacinle.utils.meter import GroupMeters
            meters = GroupMeters()

        if use_tbx:
            from tensorboardX import SummaryWriter
            tbx_dir = os.path.join(save_dir, 'tbx')
            os.makedirs(tbx_dir, exist_ok=True)
            tbx_writer = SummaryWriter(log_dir=tbx_dir)
            global_iteration = 0

        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')

        label2state_machine = self.get_label2state_machine(self.data_loader.get_all_labels(), state_machine_type)

        env_args = self.data_loader.get_env_args()
        evaluators = self.get_evaluators(
            env_name,
            dataset_name,
            env_args,
            ptrajonly=ptrajonly,
            use_not_goal=use_not_goal,
            use_true_value=use_true_value,
            state_machine_type=state_machine_type,
            deep_search_tree=deep_search_tree,
            action_cost=action_cost,
            toy_robot_net=toy_robot_net,
            search_unique=search_unique,
            all_labels=self.data_loader.get_all_labels(),
            skip_our_model=skip_our_model,
            trueip=add_baseline_trueip,
            lstm=add_baseline_lstm,
            irl_classify_by_reward=irl_classify_by_reward,
            dqn=add_baseline_dqn,
            lstmdqn=add_baseline_lstmdqn,
            a2c=add_baseline_a2c,
            lstma2c=add_baseline_lstma2c,
            irldqn=add_baseline_irldqn,
            irllstmdqn=add_baseline_irllstmdqn,
            irla2c=add_baseline_irla2c,
            irllstma2c=add_baseline_irllstma2c,
            seq2seq=add_baseline_seq2seq,
            bc=add_baseline_bc,
            irlcont=add_baseline_irlcont,
        )

        if mode in ['test', 'show', 'show_stored_data']:
            if mode == 'test':
                if 'ours' in evaluators and isinstance(evaluators['ours'], AstarEvaluator):
                    evaluators['ours'].set_zero_parameter()
            self.load_evaluators(evaluators, best=True if load_model is None else (load_model == 'best'), save_dir=save_dir, force=force)
        elif cont:
            self.load_evaluators(evaluators, best=False if load_model is None else (load_model == 'best'), save_dir=save_dir, force=force)

        task = self.get_pipeline(env_name)(
            self.data_loader,
            label2state_machine,
            meters=meters,
            save_dir=save_dir,
            log=logger,
            use_tb=use_tb,
            ptrajonly=ptrajonly,
            device=device,
        )

        for ename in evaluators:
            evaluators[ename] = evaluators[ename].to(device)
        best_loss = {ename: None for ename in evaluators}
        best_state_dicts = {ename: None for ename in evaluators}

        task.init()
        task.register_evaluators(**evaluators)

        if mode == 'show':
            self.show(
                evaluators, env_name, env_args, state_machine_type=state_machine_type, local=local, save_dir=save_dir
            )
            return

        if test_part is not None or test_part_label:
            assert not (test_part is not None and test_part_label)
            assert save_answer is not None
            if test_part_start is None:
                test_part_start = 1

        print("Start %s !" % mode)

        if mode == 'show_stored_data':
            task.summary(plot_wrong_sample=plot_wrong_sample, load_stored_data=True, local=local)
            return

        if mode == 'test':
            # torch.set_grad_enabled(False)
            start_epoch, n_epochs = 1, 1
            if test_part is not None:
                start_epoch, n_epochs = test_part_start or 1, test_part
                if test_part_end is not None:
                    n_epochs = test_part_end
            elif test_part_label:
                start_epoch, n_epochs = test_part_start or 1, len(self.data_loader.labels['test'])
                if test_part_end is not None:
                    n_epochs = test_part_end

        for epoch in range(start_epoch, n_epochs + 1 + (1 if mode == 'fast-train' else 0)):
            meters.update(epoch=epoch)
            # Get mode train/eval
            if epoch > n_epochs or mode == 'test':
                training_epoch = False
                if epoch > n_epochs:
                    for ename in evaluators:
                        jactorch.io.load_state_dict(evaluators[ename], best_state_dicts[ename])
                for evaluator in evaluators.values():
                    evaluator.eval()
            else:
                training_epoch = True

            # Get learning rate
            cur_lr = lr * lr_schedule(epoch, n_epochs=n_epochs) if callable(lr_schedule) else lr
            progress = min(1.0, start_epoch / n_epochs)
            # Evaluators prepare for the incoming epoch
            for evaluator in evaluators.values():
                evaluator.prepare_epoch(lr=cur_lr, base_lr=lr, progress=progress)
            # Get optimizer
            optimizers = {}
            if training_epoch:
                for ename, evaluator in evaluators.items():
                    training_parameters = evaluators[ename].get_training_parameters()
                    if training_parameters is not None:
                        optimizers[ename] = self.get_optimizer(optim, parameters=training_parameters, lr=cur_lr)
            # Init epoch, prepare for batches
            task.epoch_init(epoch)

            if mode == 'fast-train':
                splits = 'test' if epoch > n_epochs else 'trainval'
            elif mode == 'slow-train':
                splits = 'trainvaltest'
            elif mode == 'val':
                splits = 'val'
            elif mode == 'test':
                splits = 'test'
            else:
                raise ValueError()
            assert training_epoch == ('train' in splits)
            batches = tqdm(
                self.data_loader.batches(
                    splits,
                    batch_size=batch_size,
                    n_batches=n_train_batches,
                    val_n_batches=n_val_batches,
                    test_batch_size=test_batch_size,
                    by_label=train_choice is not None and not train_choice_uniform,
                    label_choice=train_choice if train_choice_uniform else None,
                    pos_neg_sample=train_binomial,
                    test_part=(epoch, test_part) if test_part is not None else None,
                    test_part_label=epoch if test_part_label else None,
                )
            )

            batches.set_description('Epoch %d' % epoch)

            # Run batches, train model if needed.
            for batch_id, (split, batch) in enumerate(batches):
                # if batch_id != 2:
                #     print('drop batch_id=', batch_id)
                #     continue
                # else:
                #     print('batch_id=', batch_id)
                training = split == 'train'
                last_test = (
                    (mode == 'fast-train' and splits == 'test')
                    or (mode == 'slow-train' and epoch == n_epochs)
                    or (mode == 'test')
                )
                if plan:
                    task.batch_plan(
                        batch_id,
                        split,
                        batch,
                        self.data_loader.get_traj_checker(plan_search=plan_search),
                        save_dir=save_dir,
                        policy=policy,
                        plan_search=plan_search,
                        search_iter=search_iter,
                    )
                    continue

                if test_goal_classifier:
                    task.batch_test_goal_classifier(
                        batch_id,
                        split,
                        batch,
                        self.data_loader.get_traj_checker(plan_search=plan_search)
                    )
                    continue

                if get_dependencies:
                    task.batch_get_dependencies(
                        batch_id,
                        split,
                        batch,
                    )
                    continue

                batch_losses = task.batch_run(
                    batch_id,
                    split,
                    batch,
                    last_test=last_test,
                    train_choice=train_choice,
                    train_choice_adjacent=train_choice_adjacent,
                    train_binomial=train_binomial,
                    progress=progress,
                    save_dir=save_dir,
                    alph=alpha,
                    beta=beta,
                )

                if use_tbx:
                    for ename in batch_losses:
                        tbx_writer.add_scalar(run_name + '-' + ename + '_' + split + '_loss', float(batch_losses[ename]), global_iteration)
                    global_iteration += 1

                if training:
                    for ename in optimizers:
                        if evaluators[ename].online_optimizer is not None:
                            optimizer = evaluators[ename].online_optimizer
                        else:
                            optimizer = optimizers[ename]
                        loss = batch_losses[ename]
                        optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(
                            filter(lambda p: p.requires_grad, evaluators[ename].get_training_parameters()), grad_clip
                        )
                        optimizer.step()
                else:
                    for loss in batch_losses.values():
                        del loss

            task.epoch_end(epoch,
                           plan=plan,
                           test_goal_classifier=test_goal_classifier,
                           get_dependencies=get_dependencies,
                           save_answer=save_answer if (test_part is None and not test_part_label) else save_answer + '_' + str(epoch)
                           )
            if not debug:
                meters.dump(meter_file, values='val')
                if mldash is not None:
                    try:
                        mldash.log_metric('epoch', epoch, desc=False, expr=False)
                    except Exception:
                        pass
                    finally:
                        pass
                    for key, value in meters.items():
                        if key in ('ours_train_loss', 'ours_val_loss', 'ours_test_loss'):
                            mldash.log_metric_min(key, value.val)
                        if key in ('ours_train_acc', 'ours_val_acc', 'ours_test_acc'):
                            mldash.log_metric_max(key, value.val)
            if epoch % save_every == 0 and mode in ('slow-train', 'fast-train') and epoch <= n_epochs:
                self.save_evaluators(evaluators, best=False, save_dir=save_dir)
            best_evaluators = {}
            for ename, evaluator in evaluators.items():
                val_loss = meters[ename + '_val_loss'].val
                if best_loss[ename] is None or val_loss < best_loss[ename]:
                    best_state_dicts[ename] = jactorch.io.state_dict(evaluator)
                    best_loss[ename] = val_loss
                    best_evaluators[ename] = evaluator
            if mode in ['fast-train', 'slow-train']:
                self.save_evaluators(best_evaluators, best=True, save_dir=save_dir)

        task.summary(
            plot_wrong_sample=plot_wrong_sample,
            local=local,
        )


def parse_arguments(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t', '--trial', type=str, default=None, help="trial of experiment, save dir will be at dumps/<trial>/"
    )
    parser.add_argument(
        '--env', type=str, default=None, choices=('toyrobot', 'craftingworld'), help="Select the environment"
    )
    parser.add_argument('--dataset', type=str, default='default', help="Select the dataset, can be default")
    parser.add_argument(
        '--problem',
        type=str,
        default='classification',
        choices=('classification', 'planning'),
        help="Select the problem",
    )
    parser.add_argument('--data_version', type=str, default=None, help="Select the dataset")
    parser.add_argument('--force_regen', action='store_true', default=False, help="Regen the dataset.")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size.")
    parser.add_argument('--n_train_batches', type=int, default=None, help="Number of batches per training epoch.")
    parser.add_argument('--n_val_batches', type=int, default=None, help="Number of batches per validation epoch.")
    parser.add_argument(
        '--test_batch_size', type=int, default=None, help="Batch size for testing(if different from batch_size)."
    )
    parser.add_argument('--n_epochs', type=int, default=10, help="Number of epochs.")
    parser.add_argument('--lr', type=float, default=0.03, help="Default learning rate.")
    parser.add_argument('--start_epoch', type=int, default=1, help="start at which epoch.")
    parser.add_argument('--cont', action='store_true', default=False, help="Continue to train.")
    parser.add_argument('--save_every', type=int, default=1, help="Save every # epoch.")
    parser.add_argument('--save_answer', type=str, default=None, help="Filename to save a single answer to.")
    parser.add_argument('--grad_clip', type=float, default=10, help="Clip grad with max norm.")
    parser.add_argument('--debug', action='store_true', default=False, help="Debug mode.")
    parser.add_argument('--test_part', type=int, default=None, help="Test by part.")
    parser.add_argument('--test_part_label', action='store_true', default=False, help="Test by label index.")
    parser.add_argument('--test_part_start', type=int, default=None, help="Test start at part.")
    parser.add_argument('--test_part_end', type=int, default=None, help="Test end at part.")
    parser.add_argument('--group', type=str, default=None, help="Group of experiments a suffix.")
    parser.add_argument(
        '--mode',
        type=str,
        default='fast-train',
        choices=('fast-train', 'slow-train', 'test', 'val', 'show', 'show_stored_data'),
        help="Mode of run.",
    )
    parser.add_argument(
        '--load_model',
        type=str,
        default=None,
        choices=('last', 'best'),
        help="Which stored model to load for testing/training.",
    )
    parser.add_argument('--force', action='store_true', default=False, help="Force loading/testing/show.")
    parser.add_argument('--plan', action='store_true', default=False, help="Test planning.")
    parser.add_argument('--plan_search', type=str, default=False,
                        choices=('brute', 'hierarchical', 'dependency', 'dependency_base'),
                        help="Test hierarchical search for planning and search method.")
    parser.add_argument('--test_goal_classifier', action='store_true', default=False, help="Test goal classifier.")
    parser.add_argument('--get_dependencies', action='store_true', default=False,
                        help="Get Dependencies based on learned model.")
    parser.add_argument('--search_iter', type=int, default=None, help="Number of iterations in search/RRT nodes/...")
    parser.add_argument(
        '--policy',
        type=str,
        default='optimal',
        help="Method for selecting action when planning."
    )
    parser.add_argument(
        '--optim', type=str, default='adam', choices=('adam', 'sgd', 'RMSprop'), help="Optimizer, adam of sgd"
    )
    parser.add_argument('--train_init', action='store_true', default=False, help="use trained init value")
    parser.add_argument(
        '--search_unique',
        action='store_true',
        default=False,
        help="search each state only once when astar (may cause it to be slower)",
    )
    parser.add_argument('--alpha', type=float, default=None, help="Coefficient of P(o|T).")
    parser.add_argument('--beta', type=float, default=None, help="Coefficient of P(T|o).")
    parser.add_argument(
        '--add_baseline_trueip', action='store_true', default=False, help="Add True evaluator for comparason"
    )
    parser.add_argument(
        '--state_machine_type',
        type=str,
        default=None,
        choices=('incompact', 'compact'),
        help="Type of state machine we use",
    )
    parser.add_argument(
        '--deep_search_tree',
        action='store_true',
        default=False,
        help="Set make the astar search tree along the trajectory.",
    )
    parser.add_argument('--plot_wrong_sample', type=int, default=None, help="Number of wrong samples to plot")
    parser.add_argument('--use_tb', action='store_true', default=False, help="Use tensorboard i.e. TBLogger and TBGroupMeter.")
    parser.add_argument('--use_tbx', action='store_true', default=False, help="Use tensorboardX only.")
    parser.add_argument('--local', action='store_true', default=False, help="Running locally.")
    parser.add_argument(
        '--toy_robot_net',
        type=str,
        default='point',
        choices=('point', 'mlp', 'multi-point'),
        help="Type of network in use for toy robot",
    )
    parser.add_argument('--action_cost', type=float, default=0.1, help="Cost per action on Grid datasets_v1, or per unit movement in continuous datasets_v1")
    parser.add_argument('--train_choice', type=int, default=None, help="Train on multiple-choice style")
    parser.add_argument('--train_choice_adjacent', action='store_true', default=False, help="Multiple choice are adjacent to the label.")
    parser.add_argument('--train_choice_uniform', action='store_true', default=False, help="Multiple choice uniform distribution.")
    parser.add_argument('--train_binomial', action='store_true', default=False, help="Train by positive and negative samples of the same label.")
    parser.add_argument('--add_baseline_lstm', action='store_true', default=False, help="Add LSTM baseline.")
    parser.add_argument('--add_baseline_dqn', action='store_true', default=False, help="Add DQN baseline.")
    parser.add_argument('--add_baseline_lstmdqn', action='store_true', default=False, help="Add LSTM-DQN baseline.")
    parser.add_argument('--add_baseline_a2c', action='store_true', default=False, help="Add A2C baseline.")
    parser.add_argument('--add_baseline_lstma2c', action='store_true', default=False, help="Add LSTM-A2C baseline.")
    parser.add_argument('--irl_classify_by_reward', action='store_true', default=False, help="IRL models use reward in addition to Q value on classification.")
    parser.add_argument('--add_baseline_irldqn', action='store_true', default=False, help="Add IRL-DQN baseline.")
    parser.add_argument(
        '--add_baseline_irllstmdqn', action='store_true', default=False, help="Add IRL-LSTM-DQN baseline."
    )
    parser.add_argument('--add_baseline_irla2c', action='store_true', default=False, help="Add IRL-A2C baseline.")
    parser.add_argument(
        '--add_baseline_irllstma2c', action='store_true', default=False, help="Add IRL-LSTM-A2C baseline."
    )
    parser.add_argument(
        '--add_baseline_seq2seq', action='store_true', default=False, help="Add Seq2Seq(Behavior cloning with state machine) baseline."
    )
    parser.add_argument(
        '--add_baseline_bc', action='store_true', default=False, help="Add Behavior cloning baseline."
    )
    parser.add_argument(
        '--add_baseline_irlcont', action='store_true', default=False, help="Add IRL for continuous domain."
    )
    parser.add_argument('--skip_our_model', action='store_true', default=False, help="Skip our model.")
    parser.add_argument('--ptrajonly', action='store_true', default=False, help="Only model P(traj|o).")
    parser.add_argument('--use_not_goal', action='store_true', default=False, help="Use not GOAL as INIT.")

    args, argv = parser.parse_known_args(args)
    return args, argv


def parse_dataset(env, dataset, data_version, force_regen):
    from hacl.envs.simple_continuous.playroom_gdk.datasets.visit_regions_data_loader import VisitRegionsDataLoader
    from hacl.envs.simple_continuous.playroom_gdk.datasets.playroom_data_loader import PlayroomDataLoader
    from hacl.envs.gridworld.crafting_world.datasets.crafting_data_loader import CraftingDataLoader
    dataset_name, dataset_cls = None, None
    if env == 'toyrobot':
        if dataset == 'default' or dataset == 'regions':
            dataset_name, dataset_cls = 'regions', VisitRegionsDataLoader
        elif dataset == 'playroom':
            dataset_name, dataset_cls = 'playroom', PlayroomDataLoader
    elif env == 'craftingworld':
        if dataset == 'default' or dataset == 'crafting':
            dataset_name, dataset_cls = 'crafting', CraftingDataLoader
    if dataset_name is None:
        raise ValueError("Invalid env name %s or dataset name %s." % (env, dataset))
    return dataset_name, dataset_cls(data_version, force_regen)


def main(raw_args=None):
    args, argv = parse_arguments(raw_args)
    dataset_name, data_loader = parse_dataset(args.env, args.dataset, args.data_version, args.force_regen)

    learner = NeuralAstarClassifierLearner()
    print('Started!!!')
    learner.set_data_loader(data_loader)
    learner.learn_classifier(
        args.trial,
        env_name=args.env,
        dataset_name=dataset_name,
        group=args.group,
        cont=args.cont,
        save_every=args.save_every,
        save_answer=args.save_answer,
        grad_clip=args.grad_clip,
        start_epoch=args.start_epoch,
        n_epochs=args.n_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        n_train_batches=args.n_train_batches,
        n_val_batches=args.n_val_batches,
        test_batch_size=args.test_batch_size,
        mode=args.mode,
        load_model=args.load_model,
        force=args.force,
        plan=args.plan,
        plan_search=args.plan_search,
        test_goal_classifier=args.test_goal_classifier,
        get_dependencies=args.get_dependencies,
        search_iter=args.search_iter,
        policy=args.policy,
        optim=args.optim,
        use_true_value=not args.train_init,
        state_machine_type=args.state_machine_type,
        alpha=args.alpha,
        beta=args.beta,
        deep_search_tree=args.deep_search_tree,
        add_baseline_trueip=args.add_baseline_trueip,
        plot_wrong_sample=args.plot_wrong_sample,
        parsed_args=args,
        use_mldash=True,
        use_tb=args.use_tb,
        use_tbx=args.use_tbx,
        debug=args.debug,
        local=args.local,
        toy_robot_net=args.toy_robot_net,
        action_cost=args.action_cost,
        train_choice=args.train_choice,
        train_choice_adjacent=args.train_choice_adjacent,
        train_choice_uniform=args.train_choice_uniform,
        train_binomial=args.train_binomial,
        search_unique=args.search_unique,
        irl_classify_by_reward=args.irl_classify_by_reward,
        skip_our_model=args.skip_our_model,
        add_baseline_lstm=args.add_baseline_lstm,
        add_baseline_dqn=args.add_baseline_dqn,
        add_baseline_lstmdqn=args.add_baseline_lstmdqn,
        add_baseline_a2c=args.add_baseline_a2c,
        add_baseline_lstma2c=args.add_baseline_lstma2c,
        add_baseline_irldqn=args.add_baseline_irldqn,
        add_baseline_irllstmdqn=args.add_baseline_irllstmdqn,
        add_baseline_irla2c=args.add_baseline_irla2c,
        add_baseline_irllstma2c=args.add_baseline_irllstma2c,
        add_baseline_seq2seq=args.add_baseline_seq2seq,
        add_baseline_bc=args.add_baseline_bc,
        add_baseline_irlcont=args.add_baseline_irlcont,
        ptrajonly=args.ptrajonly,
        use_not_goal=args.use_not_goal,
        test_part=args.test_part,
        test_part_label=args.test_part_label,
        test_part_start=args.test_part_start,
        test_part_end=args.test_part_end,
    )

    # cls.inject(env)
    # GridWorldCLIEmulator(env).mainloop()

    # TODO
    # 1. Learning
    # 2. Object-centric representations


if __name__ == '__main__':
    import sys

    sys_args = sys.argv[1:]
    args_list = []
    last = -1
    for i in range(len(sys_args) + 1):
        if i == len(sys_args) or sys_args[i] == '++':
            args_list.append(sys_args[last + 1:i])
            last = i
    for args in args_list:
        if len(args) > 0:
            main(args)
