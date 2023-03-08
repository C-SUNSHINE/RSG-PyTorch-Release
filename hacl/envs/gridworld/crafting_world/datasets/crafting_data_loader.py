#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import random

import numpy as np

from hacl.envs.gridworld.crafting_world.configs import COMPLEX_INSTRUCTIONS, NOVEL_INSTRUCTIONS, PLAN_SEARCH_INSTRUCTIONS, PRIMITIVES, PRIMITIVES_AND_INTEGRATED
from hacl.envs.gridworld.crafting_world.datasets.crafting_v1 import CraftingV1
from hacl.p.rsgs.data_loader import DataLoader
from hacl.utils.logger import BufferPrinter

DATA_PACK_CONFIG = {
    'v1.trial1.train_large': {
        'labels': (
            'grab_pickaxe',
            'mine_coal',
            'mine_iron_ore',
            'craft_iron_ingot'
        ),
        'args': dict(n_data=400, max_steps=50, env_args='plains', map_ids=[0, 1, 2, 3]),
        'group_size': 1,
        'drop_tail': True,
    },
    'v1.trial1.train_small': {
        'labels': (
            'grab_pickaxe',
            'mine_coal',
            'mine_iron_ore',
            'craft_iron_ingot'
        ),
        'args': dict(n_data=40, max_steps=50, env_args='plains', map_ids=[0, 1, 2, 3]),
        'group_size': 1,
        'drop_tail': True,
    },
    'v1.trial1.val': {
        'labels': (
            'grab_pickaxe',
            'mine_coal',
            'mine_iron_ore',
            'craft_iron_ingot'
        ),
        'args': dict(n_data=20, max_steps=50, env_args='plains', map_ids=[0, 1, 2, 3]),
        'group_size': 1,
        'drop_tail': True,
    },
    'v1.trial1.test': {
        'labels': (
            'grab_pickaxe',
            'mine_coal',
            'mine_iron_ore',
            'craft_iron_ingot'
        ),
        'args': dict(n_data=100, max_steps=50, env_args='plains', map_ids=[0, 1, 2, 3]),
        'group_size': 5,
        'drop_tail': True,
    },

    'v2.primitives.train_large': {
        'labels': tuple(PRIMITIVES),
        'args': dict(n_data=400, max_steps=50, env_args='primitives', map_ids=None),
        'group_size': 1,
        'drop_tail': True,
    },

    'v2.primitives.train_small': {
        'labels': tuple(PRIMITIVES),
        'args': dict(n_data=40, max_steps=50, env_args='primitives', map_ids=None),
        'group_size': 1,
        'drop_tail': True,
    },

    'v2.primitives.val': {
        'labels': tuple(PRIMITIVES),
        'args': dict(n_data=40, max_steps=50, env_args='primitives', map_ids=None),
        'group_size': 1,
        'drop_tail': True,
    },

    'v2.primitives.test': {
        'labels': tuple(PRIMITIVES),
        'args': dict(n_data=150, max_steps=50, env_args='primitives', map_ids=None),
        'group_size': 5,
        'drop_tail': True,
    },

    'v2.integrated.train_large': {
        'labels': tuple(x[0] for x in COMPLEX_INSTRUCTIONS),
        'args': dict(n_data=400, max_steps=50, env_args='integrated', map_ids=None, prerequisites=tuple(x[1] for x in COMPLEX_INSTRUCTIONS)),
        'group_size': 1,
        'drop_tail': True,
    },

    'v2.integrated.train_small': {
        'labels': tuple(x[0] for x in COMPLEX_INSTRUCTIONS),
        'args': dict(n_data=40, max_steps=50, env_args='integrated', map_ids=None, prerequisites=tuple(x[1] for x in COMPLEX_INSTRUCTIONS)),
        'group_size': 1,
        'drop_tail': True,
    },

    'v2.integrated.val': {
        'labels': tuple(x[0] for x in COMPLEX_INSTRUCTIONS),
        'args': dict(n_data=40, max_steps=50, env_args='integrated', map_ids=None, prerequisites=tuple(x[1] for x in COMPLEX_INSTRUCTIONS)),
        'group_size': 1,
        'drop_tail': True,
    },

    'v2.integrated.test': {
        'labels': tuple(x[0] for x in COMPLEX_INSTRUCTIONS),
        'args': dict(n_data=150, max_steps=50, env_args='integrated', map_ids=None, prerequisites=tuple(x[1] for x in COMPLEX_INSTRUCTIONS)),
        'group_size': 5,
        'drop_tail': True,
    },

    'v2.novel.no_train': {
        'labels': tuple(NOVEL_INSTRUCTIONS),
        'args': dict(n_data=4, max_steps=50, env_args='novels', map_ids=None),
        'group_size': 1,
        'drop_tail': True,
    },

    'v2.novel.no_val': {
        'labels': tuple(NOVEL_INSTRUCTIONS),
        'args': dict(n_data=2, max_steps=50, env_args='novels', map_ids=None),
        'group_size': 1,
        'drop_tail': True,
    },

    'v2.novel.test': {
        'labels': tuple(NOVEL_INSTRUCTIONS),
        'args': dict(n_data=100, max_steps=50, env_args='novels', map_ids=None),
        'group_size': 5,
        'drop_tail': True,
    },

    'v2.plan_search.no_train': {
        'labels': tuple(PLAN_SEARCH_INSTRUCTIONS),
        'args': dict(n_data=4, max_steps=50, env_args='plan_search', map_ids=None),
        'group_size': 1,
        'drop_tail': True,
    },

    'v2.plan_search.no_val': {
        'labels': tuple(PLAN_SEARCH_INSTRUCTIONS),
        'args': dict(n_data=2, max_steps=50, env_args='plan_search', map_ids=None),
        'group_size': 1,
        'drop_tail': True,
    },

    'v2.plan_search.test': {
        'labels': tuple(PLAN_SEARCH_INSTRUCTIONS),
        'args': dict(n_data=100, max_steps=50, env_args='plan_search', map_ids=None),
        'group_size': 1,
        'drop_tail': True,
    },
}

for PACK_NAME in DATA_PACK_CONFIG:
    DATA_PACK_CONFIG[PACK_NAME]['pack_id'] = PACK_NAME

DATA_CONFIG = {
    'v1.trial1.large': {
        'train': DATA_PACK_CONFIG['v1.trial1.train_large'],
        'val': DATA_PACK_CONFIG['v1.trial1.val'],
        'test': DATA_PACK_CONFIG['v1.trial1.test']
    },
    'v1.trial1.small': {
        'train': DATA_PACK_CONFIG['v1.trial1.train_small'],
        'val': DATA_PACK_CONFIG['v1.trial1.val'],
        'test': DATA_PACK_CONFIG['v1.trial1.test']
    },
    'v2.primitives.large': {
        'train': DATA_PACK_CONFIG['v2.primitives.train_large'],
        'val': DATA_PACK_CONFIG['v2.primitives.val'],
        'test': DATA_PACK_CONFIG['v2.primitives.test'],
    },
    'v2.primitives.small': {
        'train': DATA_PACK_CONFIG['v2.primitives.train_small'],
        'val': DATA_PACK_CONFIG['v2.primitives.val'],
        'test': DATA_PACK_CONFIG['v2.primitives.test'],
    },
    'v2.integrated.large': {
        'train': DATA_PACK_CONFIG['v2.integrated.train_large'],
        'val': DATA_PACK_CONFIG['v2.integrated.val'],
        'test': DATA_PACK_CONFIG['v2.integrated.test'],
    },
    'v2.integrated.small': {
        'train': DATA_PACK_CONFIG['v2.integrated.train_small'],
        'val': DATA_PACK_CONFIG['v2.integrated.val'],
        'test': DATA_PACK_CONFIG['v2.integrated.test'],
    },
    'v2.novel': {
        'train': DATA_PACK_CONFIG['v2.novel.no_train'],
        'val': DATA_PACK_CONFIG['v2.novel.no_val'],
        'test': DATA_PACK_CONFIG['v2.novel.test'],
    },
    'v2.plan_search': {
        'train': DATA_PACK_CONFIG['v2.plan_search.no_train'],
        'val': DATA_PACK_CONFIG['v2.plan_search.no_val'],
        'test': DATA_PACK_CONFIG['v2.plan_search.test'],
    },
    'v2.all.small': {
        'labels': PRIMITIVES_AND_INTEGRATED,
        'train': [DATA_PACK_CONFIG['v2.primitives.train_small'], DATA_PACK_CONFIG['v2.integrated.train_small']],
        'val': [DATA_PACK_CONFIG['v2.primitives.val'], DATA_PACK_CONFIG['v2.integrated.val']],
        'test': [DATA_PACK_CONFIG['v2.primitives.test'], DATA_PACK_CONFIG['v2.integrated.test']],
    },
    'v2.all.large': {
        'labels': PRIMITIVES_AND_INTEGRATED,
        'train': [DATA_PACK_CONFIG['v2.primitives.train_large'], DATA_PACK_CONFIG['v2.integrated.train_large']],
        'val': [DATA_PACK_CONFIG['v2.primitives.val'], DATA_PACK_CONFIG['v2.integrated.val']],
        'test': [DATA_PACK_CONFIG['v2.primitives.test'], DATA_PACK_CONFIG['v2.integrated.test']],
    }
}


class CraftingDataLoader(DataLoader):
    NAME = "CraftingGroup"

    def __init__(self, data_version=None, force_regen=False, *args, **kwargs):
        super().__init__()
        self.splits = ('train', 'val', 'test')
        self.data_version = data_version
        self.config = DATA_CONFIG[data_version]
        gen_class = CraftingV1
        self.rng = random.Random(233)
        train_data_packs = []
        val_data_packs = []
        test_data_packs = []
        for train_config in ([self.config['train']] if not isinstance(self.config['train'], list) else self.config['train']):
            train_data_generator = gen_class(2333, labels=train_config['labels'])
            train_pack_id = (
                train_config['pack_id'] if 'pack_id' in train_config else data_version + '_train'
            )
            train_data_pack = train_data_generator.generate(
                **train_config['args'],
                split='train',
                data_pack_name=self.NAME + train_pack_id,
                force_regen=force_regen
            )
            train_data_packs.append(train_data_pack)
        for val_config in ([self.config['val']] if not isinstance(self.config['val'], list) else self.config['val']):
            val_data_generator = gen_class(2333, labels=val_config['labels'])
            val_pack_id = val_config['pack_id'] if 'pack_id' in val_config else data_version + '_val'
            val_data_pack = val_data_generator.generate(
                **val_config['args'], split='val', data_pack_name=self.NAME + val_pack_id, force_regen=force_regen
            )
            val_data_packs.append(val_data_pack)
        for test_config in ([self.config['test']] if not isinstance(self.config['test'], list) else self.config['test']):
            test_data_generator = gen_class(23333, labels=test_config['labels'])
            test_pack_id = test_config['pack_id'] if 'pack_id' in test_config else data_version + '_test'
            test_data_pack = test_data_generator.generate(
                **test_config['args'],
                split='test',
                data_pack_name=self.NAME + test_pack_id,
                force_regen=force_regen
            )
            test_data_packs.append(test_data_pack)
        if 'labels' not in self.config:
            self.labels = {
                'train': self.config['train']['labels'][:],
                'val': self.config['val']['labels'][:],
                'test': self.config['test']['labels'][:],
            }
        else:
            self.labels = {
                'train': self.config['labels'],
                'val': self.config['labels'],
                'test': self.config['labels'],
            }
        self._load_data_pack(train_data_packs, val_data_packs, test_data_packs)

    def _load_data_pack(self, train_data_packs, val_data_packs, test_data_packs):
        self.env_args = train_data_packs[0]['env_args']
        self.data = []
        for pack in train_data_packs + val_data_packs + test_data_packs:
            self.data.extend(pack['data'])
        random.shuffle(self.data, random.Random(1).random)
        self.indices, self.label_indices, self.group_indices, self.label_group_indices = self.split_data()

    def get_all_labels(self):
        return set(PRIMITIVES).union(self.labels['test'])

    def get_env_args(self):
        return self.env_args

    def get_traj_checker(self, plan_search=False):
        from .crafting_checker import CraftingChecker

        return CraftingChecker(self.env_args, plan_search=plan_search)

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
        for split in indices:
            self.rng.shuffle(indices[split])
        label_indices = {split: {} for split in self.splits}
        label_group_indices = {split: {} for split in self.splits}
        group_indices = {split: [] for split in self.splits}
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
                group_size = self.config[split]['group_size'] if not isinstance(self.config[split], list) else self.config[split][0]['group_size']
                drop_tail = self.config[split]['drop_tail'] if not isinstance(self.config[split], list) else self.config[split][0]['drop_tail']
                for i in range(0, m, group_size):
                    if drop_tail and i + group_size > m:
                        continue
                    index_group = tuple(label_indices[split][label][i: i + group_size])
                    group_indices[split].append(index_group)
                    if label not in label_group_indices[split]:
                        label_group_indices[split][label] = []
                    label_group_indices[split][label].append(index_group)

        return indices, label_indices, group_indices, label_group_indices

    def _get_indices_from_states(self, graph, predicate):
        for s in graph.states:
            if predicate(s):
                yield graph.state2index[s]

    def _make_batch_indices(
        self, split,
        batch_size,
        n_batches=None,
        by_label=False,
        label_choice=False,
        pos_neg_sample=False,
        test_part=None,
        test_part_label=None,
        seed=None
    ):
        rng = random if seed is None else random.Random(seed)
        if not by_label:
            indices_list = [self.indices[split][:]]
        else:
            indices_list = []
            for label in self.label_indices[split]:
                indices_list.append(self.label_indices[split][label][:])
        if split == 'train':
            for indices in indices_list:
                rng.shuffle(indices)

        if test_part is not None or test_part_label is not None:
            assert not (test_part is not None and test_part_label is not None)
            assert split == 'test'
            part_indices = []
            if test_part is not None:
                assert len(test_part) == 2
                cur, tot = test_part
                assert 1 <= cur <= tot
                for label in self.label_group_indices[split]:
                    index_groups = self.label_group_indices[split][label]
                    assert len(index_groups) % tot == 0
                    per_part = len(index_groups) // tot
                    for gi in range(per_part * (cur - 1), per_part * cur):
                        part_indices.extend(index_groups[gi])
                print('Part %d/%d' % (cur, tot))
            else:
                cur = test_part_label
                tot = len(self.labels[split])
                index_groups = self.label_group_indices[split][self.labels[split][cur - 1]]
                for ig in index_groups:
                    part_indices.extend(ig)
                print('Label %d/%d: %s' % (cur, tot, self.labels[split][cur - 1]))
            part_indices = list(sorted(part_indices, key=lambda idx: len(self.data[idx]['traj'][0]), reverse=True))
            n_batches = (len(part_indices) + batch_size - 1) // batch_size
            by_batch_id = [[] for i in range(n_batches)]
            ptr = 0
            for bid in range(batch_size):
                for j in range(len(part_indices) // batch_size + (1 if bid < len(part_indices) % batch_size else 0)):
                    by_batch_id[j].append(part_indices[ptr])
                    ptr += 1
            assert ptr == len(part_indices)
            part_indices = []
            for batch_id_indices in by_batch_id:
                part_indices.extend(batch_id_indices)
            indices_list = [part_indices]

        if label_choice:
            assert not pos_neg_sample
            n_batches = n_batches or (len(self.indices[split]) + batch_size - 1) // batch_size
            indices_batches = []
            for k in range(n_batches):
                nprng = np.random.default_rng(rng.randint(0, 10 ** 9))
                while True:
                    label_ids = nprng.choice(len(self.labels[split]), label_choice, replace=False)
                    concat_label = ''.join(self.labels[split][label_id] for label_id in label_ids)
                    if label_choice + sum((1 if c in '>&|' else 0) for c in concat_label) > 10 or len(concat_label) > 150:
                        continue
                    break
                assert batch_size >= label_ids.shape[0]
                indices_batch = []
                for j in range(label_choice):
                    label = self.labels[split][label_ids[j]]
                    n_samples = batch_size // label_choice + (1 if j < batch_size % label_choice else 0)
                    sample_ids = nprng.choice(len(self.label_indices[split][label]), n_samples, replace=False)
                    indices_batch.extend(self.label_indices[split][label][id] for id in sample_ids)
                rng.shuffle(indices_batch)
                indices_batches.append(indices_batch)
            return indices_batches

        if pos_neg_sample:
            n_batches = n_batches or (len(self.indices[split]) + batch_size - 1) // batch_size
            assert batch_size % 2 == 0
            indices_batches = []
            nprng = np.random.default_rng(rng.randint(0, 10 ** 9))
            for k in range(n_batches):
                indices_batch = []
                label = rng.choice(self.labels[split])
                other_labels = list(x for x in self.labels[split] if x != label)
                sample_ids = nprng.choice(len(self.label_indices[split][label]), batch_size // 2, replace=False)
                indices_batch.extend(self.label_indices[split][label][id] for id in sample_ids)
                for j in range(batch_size - batch_size // 2):
                    neg_label = rng.choice(other_labels)
                    sample_id = rng.randint(0, len(self.label_indices[split][neg_label]) - 1)
                    indices_batch.append(self.label_indices[split][neg_label][sample_id])
                rng.shuffle(indices_batch)
                indices_batches.append(indices_batch)
            return indices_batches

        indices_batches = []
        if by_label:
            label_indices_batches = {}

        for indices in indices_list:
            for k in range((len(indices) + batch_size - 1) // batch_size):
                indices_batches.append(indices[k * batch_size: min((k + 1) * batch_size, len(indices))])
                if by_label:
                    label = self.data[indices_batches[-1][0]]['label']
                    if label not in label_indices_batches:
                        label_indices_batches[label] = []
                    label_indices_batches[label].append(indices_batches[-1])

        if split == 'train' or split == 'val':
            rng.shuffle(indices_batches)
            if n_batches is not None:
                if by_label:
                    indices_batches = []
                    for i in range(n_batches):
                        label = rng.choice(list(label_indices_batches.keys()))
                        indices_batches.append(rng.choice(label_indices_batches[label]))
                else:
                    indices_batches = indices_batches[:n_batches]
        # for batch_id, indices_batch in enumerate(indices_batches):
        #     print('total_length %d = ' % batch_id, sum(len(self.data[idx]['traj'][0]) for idx in indices_batch))
        return indices_batches

    def batches(self, split,
                batch_size=10,
                n_batches=None,
                val_n_batches=None,
                test_batch_size=None,
                by_label=False,
                label_choice=False,
                pos_neg_sample=False,
                test_part=None,
                test_part_label=None,
                seed=None,
                **kwargs):
        if split == 'trainvaltest':
            for e in self.batches('train', batch_size, n_batches=n_batches, val_n_batches=val_n_batches, by_label=by_label, label_choice=label_choice, pos_neg_sample=pos_neg_sample, seed=seed):
                yield e
            for e in self.batches('val', batch_size, n_batches=n_batches, val_n_batches=val_n_batches, by_label=by_label, label_choice=label_choice, pos_neg_sample=pos_neg_sample, seed=seed):
                yield e
            for e in self.batches('test', batch_size, n_batches=n_batches, val_n_batches=val_n_batches, test_batch_size=test_batch_size, seed=seed):
                yield e
            return
        elif split == 'trainval':
            for e in self.batches('train', batch_size, n_batches=n_batches, val_n_batches=val_n_batches, by_label=by_label, label_choice=label_choice, pos_neg_sample=pos_neg_sample, seed=seed):
                yield e
            for e in self.batches('val', batch_size, n_batches=n_batches, val_n_batches=val_n_batches, by_label=by_label, label_choice=label_choice, pos_neg_sample=pos_neg_sample, seed=seed):
                yield e
            return
        else:
            assert split in self.group_indices

        if split == 'test' and test_batch_size is not None:
            batch_size = test_batch_size
        if split == 'val' and val_n_batches is not None:
            n_batches = val_n_batches

        indices_batches = self._make_batch_indices(
            split, batch_size,
            n_batches=n_batches,
            by_label=by_label and split != 'test',
            label_choice=label_choice if split != 'test' else False,
            pos_neg_sample=pos_neg_sample and split != 'test',
            test_part=test_part if split == 'test' else None,
            test_part_label=test_part_label if split == 'test' else None,
            seed=seed)
        print('#batches', len(indices_batches))
        for indices in indices_batches:
            batch_data = []
            label = None
            for i in indices:
                sample = self.data[i]
                sample = {k: v for k, v in sample.items()}
                sample['index'] = i
                if by_label and split != 'test':
                    assert label is None or sample['label'] == label
                label = sample['label']
                batch_data.append(sample)
            yield (split, batch_data)

    def group_size(self, split):
        return self.config[split]['group_size'] if not isinstance(self.config[split], list) else self.config[split][0]['group_size']

    def groups(self, split):
        for group in self.group_indices[split]:
            yield group, self.data[group[0]]['label']

    def collect_data(self, batch, index, *args, **kwargs):
        new_sample = {key: batch[index][key] for key in ('start_state', 'traj', 'label', 'index')}
        for k in kwargs:
            assert k not in new_sample
            new_sample[k] = kwargs[k]
        # print('Data collected, data_index=%d' % new_sample['index'])
        return new_sample

    def print_data(self, sample, note_keys=None, log=print, local=True, **kwargs):
        log = BufferPrinter(log)
        if note_keys is None:
            note_keys = set()
        for key in note_keys:
            if key in sample:
                log("%s:" % key, sample[key])
            else:
                log("%s:" % key, "Not Found.")
        log('#' * 80)
        if local:
            CraftingV1.render_data_human([sample], env_args=self.env_args)

        return log.clear()


if __name__ == '__main__':
    cnt = {}
    data_loader = CraftingDataLoader(data_version='v1.trial1.small', force_regen=False)
    for group, label in data_loader.groups('test'):
        if label not in cnt:
            cnt[label] = 0
        cnt[label] += 1
    print(cnt)
    for (split, batch) in data_loader.batches('train'):
        for sample in batch:
            data_loader.print_data(sample, local=True)
