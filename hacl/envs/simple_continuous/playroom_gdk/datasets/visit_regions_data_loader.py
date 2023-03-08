#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random

from hacl.envs.simple_continuous.playroom_gdk.configs import DEFAULT_ENV_ARGS_V1
from hacl.p.rsgs.data_loader import DataLoader
from hacl.utils.logger import BufferPrinter
from .visit_regions_v1 import VisitRegionsV1

DATA_CONFIG = {
    'v1.0': {
        'train': {
            'labels': ('A', 'B', 'C'),
            'args': dict(n_data=4, env_args='regions_empty'),
            'group_size': 1,
            'drop_tail': True,
        },
        'val': {
            'labels': ('A', 'B', 'C'),
            'args': dict(n_data=2, env_args='regions_empty'),
            'group_size': 1,
            'drop_tail': True,
        },
        'test': {
            'labels': ('A', 'B', 'C'),
            'args': dict(n_data=3, env_args='regions_empty'),
            'group_size': 3,
            'drop_tail': True,
        },
    },
    'v1.1': {
        'train': {
            'labels': ('A', 'B', 'C', 'A>B', 'B>A', 'A>C', 'C>A', 'B>C', 'C>B'),
            'args': dict(n_data=15, env_args='regions_empty'),
            'group_size': 1,
            'drop_tail': True,
        },
        'val': {
            'labels': ('A', 'B', 'C', 'A>B', 'B>A', 'A>C', 'C>A', 'B>C', 'C>B'),
            'args': dict(n_data=10, env_args='regions_empty'),
            'group_size': 1,
            'drop_tail': True,
        },
        'test': {
            'labels': ('A>B>C', 'B|C', 'C>A|B', 'A>C', 'A&C'),
            'args': dict(n_data=10, env_args='regions_empty'),
            'group_size': 5,
            'drop_tail': True,
        },
    },
    'v1.2': {
        'train': {
            'labels': ('A', 'B', 'C', 'D', 'A>B', 'B>C', 'C>D', 'D>A', 'A>D', 'D>C', 'C>B', 'B>A'),
            'args': dict(n_data=15, env_args='regions_Xshape'),
            'group_size': 1,
            'drop_tail': True,
        },
        'val': {
            'labels': ('A', 'B', 'C', 'D', 'A>B', 'B>C', 'C>D', 'D>A', 'A>D', 'D>C', 'C>B', 'B>A'),
            'args': dict(n_data=10, env_args='regions_Xshape'),
            'group_size': 1,
            'drop_tail': True,
        },
        'test': {
            'labels': ('A>B>C', 'B|C', 'C>A|B', 'A&C', 'B>A&C', 'A|C>B', 'A>C|D', 'C|D'),
            'args': dict(n_data=10, env_args='regions_Xshape'),
            'group_size': 5,
            'drop_tail': True,
        },
    },
    'v1.3': {
        'train': {
            'labels': ('A', 'B', 'C', 'A>B', 'B>C', 'C>A', 'A>C', 'C>B', 'B>A'),
            'args': dict(n_data=15, env_args='regions_maze1'),
            'group_size': 1,
            'drop_tail': True,
        },
        'val': {
            'labels': ('A', 'B', 'C', 'A>B', 'B>C', 'C>A', 'A>C', 'C>B', 'B>A'),
            'args': dict(n_data=10, env_args='regions_maze1'),
            'group_size': 1,
            'drop_tail': True,
        },
        'test': {
            'labels': ('A>B>C', 'B|C', 'C>A|B', 'A&C', 'B>A&C', 'A|C>B'),
            'args': dict(n_data=10, env_args='regions_maze1'),
            'group_size': 5,
            'drop_tail': True,
        },
    },
    # test model
    'v2.0': {
        'train': {
            'labels': ('A', 'B', 'C'),
            'args': dict(n_data=4, env_args='regions_empty'),
            'group_size': 2,
            'drop_tail': True,
        },
        'val': {
            'labels': ('A', 'B', 'C'),
            'args': dict(n_data=2, env_args='regions_empty'),
            'group_size': 1,
            'drop_tail': True,
        },
        'test': {
            'labels': ('A', 'B', 'C'),
            'args': dict(n_data=3, env_args='regions_empty'),
            'group_size': 3,
            'drop_tail': True,
        },
    },
    'v2.01': {
        'train': {
            'labels': ('A', 'B', 'C', 'D'),
            'args': dict(n_data=40, env_args='regions_empty_corner'),
            'group_size': 1,
            'drop_tail': True,
        },
        'val': {
            'labels': ('A', 'B', 'C', 'D'),
            'args': dict(n_data=15, env_args='regions_empty_corner'),
            'group_size': 1,
            'drop_tail': True,
        },
        'test': {
            'labels': ('A', 'B', 'C', 'D'),
            'args': dict(n_data=100, env_args='regions_empty_corner'),
            'group_size': 5,
            'drop_tail': True,
        },
    },
    # Empty map, train ABC test ABC
    'v2.1': {
        'train': {
            'labels': ('A', 'B', 'C'),
            'args': dict(n_data=40, env_args='regions_empty'),
            'group_size': 1,
            'drop_tail': True,
        },
        'val': {
            'labels': ('A', 'B', 'C'),
            'args': dict(n_data=15, env_args='regions_empty'),
            'group_size': 1,
            'drop_tail': True,
        },
        'test': {
            'labels': ('A', 'B', 'C'),
            'args': dict(n_data=25, env_args='regions_empty'),
            'group_size': 5,
            'drop_tail': True,
        },
    },
    # empty0, train simple, test complex
    'v2.2': {
        'train': {
            'labels': ('A', 'B', 'C', 'A>B', 'B>A', 'A>C', 'C>A', 'B>C', 'C>B'),
            'args': dict(n_data=50, env_args='regions_empty'),
            'group_size': 1,
            'drop_tail': True,
        },
        'val': {
            'labels': ('A', 'B', 'C', 'A>B', 'B>A', 'A>C', 'C>A', 'B>C', 'C>B'),
            'args': dict(n_data=15, env_args='regions_empty'),
            'group_size': 1,
            'drop_tail': True,
        },
        'test': {
            'labels': ('A>B>C', 'B|C', 'C>A|B', 'A&C', 'B>A&C', 'A|C>B'),
            'args': dict(n_data=25, env_args='regions_empty'),
            'group_size': 5,
            'drop_tail': True,
        },
    },
    # X_shape1, train simple, test complex
    'v2.3': {
        'train': {
            'labels': ('A', 'B', 'C', 'D', 'A>B', 'B>C', 'C>D', 'D>A', 'A>D', 'D>C', 'C>B', 'B>A'),
            'args': dict(n_data=40, env_args='regions_Xshape'),
            'group_size': 1,
            'drop_tail': True,
        },
        'val': {
            'labels': ('A', 'B', 'C', 'D', 'A>B', 'B>C', 'C>D', 'D>A', 'A>D', 'D>C', 'C>B', 'B>A'),
            'args': dict(n_data=15, env_args='regions_Xshape'),
            'group_size': 1,
            'drop_tail': True,
        },
        'test': {
            'labels': ('A>B>C', 'B|C', 'C>A|B', 'A&C', 'B>A&C', 'A|C>B', 'A>C|D', 'C|D'),
            'args': dict(n_data=25, env_args='regions_Xshape'),
            'group_size': 5,
            'drop_tail': True,
        },
    },
    # Maze1, train simple, test complex
    'v2.4': {
        'train': {
            'labels': ('A', 'B', 'C', 'A>B', 'B>C', 'C>A', 'A>C', 'C>B', 'B>A'),
            'args': dict(n_data=40, env_args='regions_maze1'),
            'group_size': 1,
            'drop_tail': True,
        },
        'val': {
            'labels': ('A', 'B', 'C', 'A>B', 'B>C', 'C>A', 'A>C', 'C>B', 'B>A'),
            'args': dict(n_data=15, env_args='regions_maze1'),
            'group_size': 1,
            'drop_tail': True,
        },
        'test': {
            'labels': ('A>B>C', 'B|C', 'C>A|B', 'A&C', 'B>A&C', 'A|C>B'),
            'args': dict(n_data=25, env_args='regions_maze1'),
            'group_size': 5,
            'drop_tail': True,
        },
    },
    # Maze1, train complex, test ABC
    'v2.5': {
        'train': {
            'labels': ('A>B', 'A|B', 'B>C', 'C', 'A>B|C', 'C>B>A'),
            'args': dict(n_data=40, env_args='regions_maze1'),
            'group_size': 1,
            'drop_tail': True,
        },
        'val': {
            'labels': ('A>B', 'A|B', 'B>C', 'C', 'A>B|C', 'C>B>A'),
            'args': dict(n_data=15, env_args='regions_maze1'),
            'group_size': 1,
            'drop_tail': True,
        },
        'test': {
            'labels': ('A', 'B', 'C'),
            'args': dict(n_data=25, env_args='regions_maze1'),
            'group_size': 5,
            'drop_tail': True,
        },
    },
}

DATA_GENERATOR = {
    '1.': VisitRegionsV1,
    '2.': VisitRegionsV1,
}


class VisitRegionsDataLoader(DataLoader):
    NAME = "ToyRobotVisitRegions"

    def __init__(self, data_version=None, force_regen=False, *args, **kwargs):
        super().__init__()
        self.splits = ('train', 'val', 'test')
        self.data_version = data_version
        self.config = DATA_CONFIG[data_version]
        gen_class = VisitRegionsV1
        for lv in DATA_GENERATOR:
            if lv in data_version:
                gen_class = DATA_GENERATOR[lv]

        train_data_generator = gen_class(2333, labels=self.config['train']['labels'])
        train_pack_id = (
            self.config['train']['pack_id'] if 'pack_id' in self.config['train'] else data_version + '_train'
        )
        train_data_pack = train_data_generator.generate(
            **self.config['train']['args'],
            split='train',
            data_pack_name=self.NAME + train_pack_id,
            force_regen=force_regen
        )
        val_data_generator = gen_class(2333, labels=self.config['val']['labels'])
        val_pack_id = self.config['val']['pack_id'] if 'pack_id' in self.config['val'] else data_version + '_val'
        val_data_pack = val_data_generator.generate(
            **self.config['val']['args'], split='val', data_pack_name=self.NAME + val_pack_id, force_regen=force_regen
        )
        test_data_generator = gen_class(23333, labels=self.config['test']['labels'])
        test_pack_id = self.config['test']['pack_id'] if 'pack_id' in self.config['test'] else data_version + '_test'
        test_data_pack = test_data_generator.generate(
            **self.config['test']['args'],
            split='test',
            data_pack_name=self.NAME + test_pack_id,
            force_regen=force_regen
        )
        self.labels = {
            'train': self.config['train']['labels'][:],
            'val': self.config['val']['labels'][:],
            'test': self.config['test']['labels'][:],
        }
        self._load_data_pack(train_data_pack, val_data_pack, test_data_pack)

    def _load_data_pack(self, train_data_pack, val_data_pack, test_data_pack):
        self.env_args = train_data_pack['env_args']
        self.data = train_data_pack['data'] + val_data_pack['data'] + test_data_pack['data']
        self.group_indices, self.label_group_indices = self.split_data()

    @classmethod
    def build_env_from_args(cls, env_args):
        return VisitRegionsV1.build_env_from_args(env_args)

    def get_all_labels(self):
        return set(self.labels['train']).union(set(self.labels['test'])).union(set(self.labels['val']))

    def get_env_args(self):
        if isinstance(self.env_args, str):
            return DEFAULT_ENV_ARGS_V1[self.env_args]
        else:
            return self.env_args

    def split_data(self):
        indices = {}
        left = []
        for i, d in enumerate(self.data):
            if 'split' not in d or d['split'] is None:
                left.append(i)
            else:
                if d['split'] not in indices:
                    indices[d['split']] = []
                indices[d['split']].append(i)
        for split in self.splits:
            if split not in indices:
                indices[split] = []
        indices['train'].extend(left[: len(left) * 8 // 16])
        indices['val'].extend(left[len(left) * 8 // 16: len(left) * 13 // 16])
        indices['test'].extend(left[len(left) * 13 // 16:])
        label_indices = {split: {} for split in self.splits}
        group_indices = {split: [] for split in self.splits}
        label_group_indices = {split: {} for split in self.splits}
        for split in self.splits:
            for i in indices[split]:
                label = self.data[i]['label']
                if label not in label_indices[split]:
                    label_indices[split][label] = []
                label_indices[split][label].append(i)
        rng = random.Random(123)
        for split in self.splits:
            for label in label_indices[split]:
                rng.shuffle(label_indices[split][label])
                m = len(label_indices[split][label])
                group_size = self.config[split]['group_size']
                for i in range(0, m, group_size):
                    if self.config[split]['drop_tail'] and i + group_size > m:
                        continue
                    index_group = tuple(label_indices[split][label][i: i + group_size])
                    group_indices[split].append(index_group)
                    if label not in label_group_indices[split]:
                        label_group_indices[split][label] = []
                    label_group_indices[split][label].append(index_group)
            rng.shuffle(group_indices[split])
        return group_indices, label_group_indices

    def group_size(self, split):
        return self.config[split]['group_size']

    def _get_indices_from_states(self, graph, predicate):
        for s in graph.states:
            if predicate(s):
                yield graph.state2index[s]

    def _make_batch_indices(self, split, batch_size, n_batches=None, by_label=False, seed=None):
        rng = random if seed is None else random.Random(seed)
        if not by_label:
            indices_list = [self.group_indices[split][:]]
        else:
            indices_list = [self.label_group_indices[split][label][:] for label in self.label_group_indices[split]]
        if split == 'train':
            for indices in indices_list:
                rng.shuffle(indices)

        actual_batch_size = (batch_size - 1) // self.config[split]['group_size'] + 1
        indices_batches = []

        for indices in indices_list:
            for k in range((len(indices) + actual_batch_size - 1) // actual_batch_size):
                indices_batches.append(indices[k * actual_batch_size: min((k + 1) * actual_batch_size, len(indices))])

        if split == 'train' or split == 'val':
            rng.shuffle(indices_batches)
            if n_batches is not None:
                indices_batches = indices_batches[:n_batches]
        return indices_batches

    def batches(self, split, batch_size=10, n_batches=None, val_n_batches=None, test_batch_size=None, by_label=False, seed=None):
        if split == 'trainvaltest':
            for e in self.batches('train', batch_size, n_batches=n_batches, val_n_batches=val_n_batches, by_label=by_label, seed=seed):
                yield e
            for e in self.batches('val', batch_size, n_batches=n_batches, val_n_batches=val_n_batches, by_label=by_label, seed=seed):
                yield e
            for e in self.batches('test', batch_size, n_batches=n_batches, val_n_batches=val_n_batches, test_batch_size=test_batch_size, seed=seed):
                yield e
            return
        elif split == 'trainval':
            for e in self.batches('train', batch_size, n_batches=n_batches, val_n_batches=val_n_batches, by_label=by_label, seed=seed):
                yield e
            for e in self.batches('val', batch_size, n_batches=n_batches, val_n_batches=val_n_batches, by_label=by_label, seed=seed):
                yield e
            return
        else:
            assert split in self.group_indices

        if split == 'test' and test_batch_size is not None:
            batch_size = test_batch_size
        if split == 'val' and val_n_batches is not None:
            n_batches = val_n_batches

        indices_batches = self._make_batch_indices(split, batch_size, n_batches=n_batches, by_label=by_label and split != 'test', seed=seed)

        for indices_batch in indices_batches:
            batch_data = []
            for indices in indices_batch:
                label = self.data[indices[0]]['label']

                labels = []
                start_configs = []
                trajs = []
                init_symbolic_states = []
                for i in indices:
                    labels.append(self.data[i]['label'])
                    start_configs.append(self.data[i]['start_config'])
                    trajs.append(self.data[i]['traj'])
                group_data = dict(
                    labels=labels,
                    start_configs=start_configs,
                    trajs=trajs,
                )
                batch_data.append((group_data, label))
            yield (split, batch_data)

    def collect_data(self, batch, index, *args, **kwargs):
        new_sample = {
            'labels': batch[index][0]['labels'],
            'start_configs': batch[index][0]['start_configs'],
            'trajs': batch[index][0]['trajs'],
            'label': batch[index][1],
        }
        for k in kwargs:
            assert k not in new_sample
            new_sample[k] = kwargs[k]
        print('Data collected')
        # print(new_sample)
        return new_sample

    def print_data(self, sample, note_keys=None, traj_note_keys=None, log=print, local=True, qvalue_evaluator=None, save_dir=None):
        log = BufferPrinter(log)
        n = len(sample['trajs'])
        if note_keys is None:
            note_keys = set()
        if traj_note_keys is None:
            traj_note_keys = set()
        log('#' * 80)
        # log('label:', sample['label'])
        for key in note_keys:
            if key in sample:
                log("%s:" % key, sample[key])
            else:
                log("%s:" % key, "Not Found.")
        from hacl.envs.simple_continuous.playroom_gdk.visualize import visualize_problem_and_solution

        env_args = self.get_env_args()
        pspace = self.build_env_from_args(env_args)
        for k in range(n):
            log('+' * 60)
            print("Traj %d/%d" % (k + 1, n))
            pspace.start_state, pspace.goal_state = None, None
            if local:
                visualize_problem_and_solution(pspace, path=sample['trajs'][k])

        if qvalue_evaluator is not None:
            from hacl.envs.simple_continuous.playroom_gdk.broadcast_engine import ToyRobotBroadcastEngine
            from hacl.envs.simple_continuous.playroom_gdk.toyrobot_v20210423 import ToyRobotV20210423
            for k in range(n):
                sample_full_id = str(sample['batch_id']) + '_' + str(sample['sample_id']) + '_' + str(k)
                acts = ['eff_A', 'eff_B', 'eff_C', 'eff_D']
                init_benv = ToyRobotBroadcastEngine(env_args)
                init_benv.env.load_from_symbolic_state(sample['trajs'][k][0][0])
                qvalue_evaluator.init_values.visualize_acts(
                    acts,
                    env_args=ToyRobotV20210423.complete_env_args(env_args),
                    save_dir=os.path.join(save_dir, sample_full_id),
                    setup='toy_robot',
                    init_benv=init_benv
                )

        return log.clear()
