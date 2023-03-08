#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random

from hacl.envs.simple_continuous.playroom_gdk.configs import DEFAULT_ENV_ARGS_V1
from hacl.p.rsgs.data_loader import DataLoader
from hacl.utils.logger import BufferPrinter
from .playroom_v1 import PlayRoomV1

COMPLEX_TRAINING_LABELS = [
    'Ball',  # Play the ball in the dark
    'LightOn>Bell',  # Ring the bell, this requires the light to be on
    'MusicOn&Ball>Monkey',  # Turn on the music and the play the ball, this will make the monkey cry, then go check the monkey.
    'Ball>LightOn',  # Play the ball in the dark and turn on the light to check.
    'MusicOn&Ball>MusicOff',  # Make the monkey cry and then stop it.
    'MusicOn|Ball',  # Make some sound in the dark (only the music_on_button and ball can be seen in the dark)
    'MusicOff>Ball>MusicOn',  # Safely play the ball in the dark (without making the monkey cry)
    'MusicOn&Ball&(LightOn>Bell)',  # Make all noise.
]

DATA_PACK_CONFIG = {

    'v2.debug.train': {
        'labels': ('Ball', 'Bell', 'LightOn', 'Monkey', 'MusicOff', 'MusicOn'),
        'args': dict(n_data=4, env_args='playroom_fourrooms'),
        'group_size': 1,
        'drop_tail': True,
    },
    'v2.debug.val': {
        'labels': ('Ball', 'Bell', 'LightOn', 'Monkey', 'MusicOff', 'MusicOn'),
        'args': dict(n_data=2, env_args='playroom_fourrooms'),
        'group_size': 1,
        'drop_tail': True,
    },
    'v2.debug.test': {
        'labels': ('Ball>LightOn', 'LightOn>Ball|Bell', 'MusicOn>LightOn', 'Ball>Monkey', 'MusicOn>MusicOff', 'MusicOn&Bell>Monkey', 'Bell>Monkey>LightOn', 'LightOn&(Bell|MusicOn)>Ball'),
        'args': dict(n_data=3, env_args='playroom_fourrooms'),
        'group_size': 3,
        'drop_tail': True,
    },

    'v2.single.train_large': {
        'labels': ('Ball', 'Bell', 'LightOn', 'Monkey', 'MusicOff', 'MusicOn'),
        'args': dict(n_data=400, env_args='playroom_fourrooms'),
        'group_size': 1,
        'drop_tail': True,
    },
    'v2.single.train_small': {
        'labels': ('Ball', 'Bell', 'LightOn', 'Monkey', 'MusicOff', 'MusicOn'),
        'args': dict(n_data=40, env_args='playroom_fourrooms'),
        'group_size': 1,
        'drop_tail': True,
    },
    'v2.single.val': {
        'labels': ('Ball', 'Bell', 'LightOn', 'Monkey', 'MusicOff', 'MusicOn'),
        'args': dict(n_data=20, env_args='playroom_fourrooms'),
        'group_size': 1,
        'drop_tail': True,
    },
    'v2.single.test': {
        'labels': ('Ball', 'Bell', 'LightOn', 'Monkey', 'MusicOff', 'MusicOn'),
        'args': dict(n_data=100, env_args='playroom_fourrooms'),
        'group_size': 5,
        'drop_tail': True,
    },
    'v2.pair.train_large': {
        'labels': ('Ball', 'Bell', 'LightOn', 'Monkey', 'MusicOff', 'MusicOn', 'Ball>Bell', 'Bell>LightOn', 'LightOn>Monkey', 'Monkey>MusicOff', 'MusicOff>MusicOn', 'MusicOn>Ball'),
        'args': dict(n_data=400, env_args='playroom_fourrooms'),
        'group_size': 1,
        'drop_tail': True,
    },
    'v2.pair.train_small': {
        'labels': ('Ball', 'Bell', 'LightOn', 'Monkey', 'MusicOff', 'MusicOn', 'Ball>Bell', 'Bell>LightOn', 'LightOn>Monkey', 'Monkey>MusicOff', 'MusicOff>MusicOn', 'MusicOn>Ball'),
        'args': dict(n_data=40, env_args='playroom_fourrooms'),
        'group_size': 1,
        'drop_tail': True,
    },
    'v2.pair.val': {
        'labels': ('Ball', 'Bell', 'LightOn', 'Monkey', 'MusicOff', 'MusicOn', 'Ball>Bell', 'Bell>LightOn', 'LightOn>Monkey', 'Monkey>MusicOff', 'MusicOff>MusicOn', 'MusicOn>Ball'),
        'args': dict(n_data=20, env_args='playroom_fourrooms'),
        'group_size': 1,
        'drop_tail': True,
    },
    'v2.pair.test': {
        'labels': ('Ball', 'Bell', 'LightOn', 'Monkey', 'MusicOff', 'MusicOn', 'Ball>Bell', 'Bell>LightOn', 'LightOn>Monkey', 'Monkey>MusicOff', 'MusicOff>MusicOn', 'MusicOn>Ball'),
        'args': dict(n_data=100, env_args='playroom_fourrooms'),
        'group_size': 5,
        'drop_tail': True,
    },
    'v2.complex.train_large': {
        'labels': ('Ball>LightOn', 'LightOn>Ball|Bell', 'MusicOn>LightOn', 'Ball>Monkey', 'MusicOn>MusicOff', 'MusicOn&Bell>Monkey', 'Bell>Monkey>LightOn', 'LightOn&(Bell|MusicOn)>Ball'),
        'args': dict(n_data=400, env_args='playroom_fourrooms'),
        'group_size': 1,
        'drop_tail': True,
    },
    'v2.complex.train_small': {
        'labels': ('Ball>LightOn', 'LightOn>Ball|Bell', 'MusicOn>LightOn', 'Ball>Monkey', 'MusicOn>MusicOff', 'MusicOn&Bell>Monkey', 'Bell>Monkey>LightOn', 'LightOn&(Bell|MusicOn)>Ball'),
        'args': dict(n_data=40, env_args='playroom_fourrooms'),
        'group_size': 1,
        'drop_tail': True,
    },
    'v2.complex.val': {
        'labels': ('Ball>LightOn', 'LightOn>Ball|Bell', 'MusicOn>LightOn', 'Ball>Monkey', 'MusicOn>MusicOff', 'MusicOn&Bell>Monkey', 'Bell>Monkey>LightOn', 'LightOn&(Bell|MusicOn)>Ball'),
        'args': dict(n_data=20, env_args='playroom_fourrooms'),
        'group_size': 1,
        'drop_tail': True,
    },
    'v2.complex.test': {
        'labels': ('Ball>LightOn', 'LightOn>Ball|Bell', 'MusicOn>LightOn', 'Ball>Monkey', 'MusicOn>MusicOff', 'MusicOn&Bell>Monkey', 'Bell>Monkey>LightOn', 'LightOn&(Bell|MusicOn)>Ball'),
        'args': dict(n_data=100, env_args='playroom_fourrooms'),
        'group_size': 5,
        'drop_tail': True,
    },
    'v2.c2s.train_large': {
        'labels': COMPLEX_TRAINING_LABELS,
        'args': dict(n_data=400, env_args='playroom_fourrooms'),
        'group_size': 1,
        'drop_tail': True,
    },
    'v2.c2s.train_small': {
        'labels': COMPLEX_TRAINING_LABELS,
        'args': dict(n_data=40, env_args='playroom_fourrooms'),
        'group_size': 1,
        'drop_tail': True,
    },
    'v2.c2s.val': {
        'labels': COMPLEX_TRAINING_LABELS,
        'args': dict(n_data=20, env_args='playroom_fourrooms'),
        'group_size': 1,
        'drop_tail': True,
    },
    'v2.c2s.test': {
        'labels': COMPLEX_TRAINING_LABELS,
        'args': dict(n_data=100, env_args='playroom_fourrooms'),
        'group_size': 5,
        'drop_tail': True,
    },
    'v2.mission1.test': {
        'labels': ('MusicOn>Ball>LightOn>Monkey>MusicOff>Bell', '(MusicOn&LightOn)>(Ball|Bell)>Monkey', 'MusicOn>(Ball&Bell)>Monkey'),
        'args': dict(n_data=100, env_args='playroom_fourrooms'),
        'group_size': 1,
        'drop_tail': True,
    },
    'v2.single.playroom_maze1.train': {
        'labels': ('Ball', 'Bell', 'LightOn', 'Monkey', 'MusicOff', 'MusicOn'),
        'args': dict(n_data=4, env_args='playroom_maze1'),
        'group_size': 1,
        'drop_tail': True,
    },
    'v2.single.playroom_maze1.val': {
        'labels': ('Ball', 'Bell', 'LightOn', 'Monkey', 'MusicOff', 'MusicOn'),
        'args': dict(n_data=2, env_args='playroom_maze1'),
        'group_size': 1,
        'drop_tail': True,
    },
    'v2.mission2.test': {
        'labels': ('MusicOn>Ball>LightOn>Monkey>MusicOff>Bell', '(MusicOn&LightOn)>(Ball|Bell)>Monkey', 'MusicOn>(Ball&Bell)>Monkey'),
        'args': dict(n_data=100, env_args='playroom_maze1'),
        'group_size': 1,
        'drop_tail': True,
    },
}

for PACK_NAME in DATA_PACK_CONFIG:
    DATA_PACK_CONFIG[PACK_NAME]['pack_id'] = PACK_NAME

DATA_CONFIG = {
    'v1.0': {
        'train': {
            'labels': ('MusicOn', 'LightOn', 'Bell'),
            'args': dict(n_data=4, env_args='playroom_default'),
            'group_size': 2,
            'drop_tail': True,
        },
        'val': {
            'labels': ('MusicOn', 'LightOn', 'Bell'),
            'args': dict(n_data=2, env_args='playroom_default'),
            'group_size': 1,
            'drop_tail': True,
        },
        'test': {
            'labels': ('MusicOn', 'LightOn', 'Bell'),
            'args': dict(n_data=3, env_args='playroom_default'),
            'group_size': 3,
            'drop_tail': True,
        },
    },

    'v1.01': {
        'train': {
            'labels': ('MusicOn', 'LightOn', 'Bell'),
            'args': dict(n_data=40, env_args='playroom_default'),
            'group_size': 1,
            'drop_tail': True,
        },
        'val': {
            'labels': ('MusicOn', 'LightOn', 'Bell'),
            'args': dict(n_data=15, env_args='playroom_default'),
            'group_size': 1,
            'drop_tail': True,
        },
        'test': {
            'labels': ('MusicOn', 'LightOn', 'Bell'),
            'args': dict(n_data=100, env_args='playroom_default'),
            'group_size': 5,
            'drop_tail': True,
        },
    },

    'v1.02': {
        'train': {
            'labels': ('Ball', 'Bell', 'LightOn', 'Monkey', 'MusicOff', 'MusicOn'),
            'args': dict(n_data=40, env_args='playroom_default'),
            'group_size': 1,
            'drop_tail': True,
        },
        'val': {
            'labels': ('Ball', 'Bell', 'LightOn', 'Monkey', 'MusicOff', 'MusicOn'),
            'args': dict(n_data=15, env_args='playroom_default'),
            'group_size': 1,
            'drop_tail': True,
        },
        'test': {
            'labels': ('Ball', 'Bell', 'LightOn', 'Monkey', 'MusicOff', 'MusicOn'),
            'args': dict(n_data=100, env_args='playroom_default'),
            'group_size': 5,
            'drop_tail': True,
        },
    },

    'v1.03': {
        'train': {
            'labels': ('Ball',),
            'args': dict(n_data=40, env_args='playroom_default'),
            'group_size': 1,
            'drop_tail': True,
        },
        'val': {
            'labels': ('Ball',),
            'args': dict(n_data=15, env_args='playroom_default'),
            'group_size': 1,
            'drop_tail': True,
        },
        'test': {
            'labels': ('Ball',),
            'args': dict(n_data=100, env_args='playroom_default'),
            'group_size': 5,
            'drop_tail': True,
        },
    },

    'v1.04': {
        'train': {
            'labels': ('Ball', 'Monkey'),
            'args': dict(n_data=40, env_args='playroom_fourrooms'),
            'group_size': 1,
            'drop_tail': True,
        },
        'val': {
            'labels': ('Ball', 'Monkey'),
            'args': dict(n_data=15, env_args='playroom_fourrooms'),
            'group_size': 1,
            'drop_tail': True,
        },
        'test': {
            'labels': ('Ball', 'Monkey'),
            'args': dict(n_data=100, env_args='playroom_fourrooms'),
            'group_size': 5,
            'drop_tail': True,
        },
    },

    'v1.05': {
        'train': {
            'labels': ('Ball', 'Monkey'),
            'args': dict(n_data=40, env_args='playroom_default'),
            'group_size': 1,
            'drop_tail': True,
        },
        'val': {
            'labels': ('Ball', 'Monkey'),
            'args': dict(n_data=15, env_args='playroom_default'),
            'group_size': 1,
            'drop_tail': True,
        },
        'test': {
            'labels': ('Ball', 'Monkey'),
            'args': dict(n_data=100, env_args='playroom_default'),
            'group_size': 5,
            'drop_tail': True,
        },
    },

    'v1.06': {
        'train': {
            'labels': ('Ball',),
            'args': dict(n_data=40, env_args='playroom_fourrooms'),
            'group_size': 1,
            'drop_tail': True,
        },
        'val': {
            'labels': ('Ball',),
            'args': dict(n_data=15, env_args='playroom_fourrooms'),
            'group_size': 1,
            'drop_tail': True,
        },
        'test': {
            'labels': ('Ball',),
            'args': dict(n_data=100, env_args='playroom_fourrooms'),
            'group_size': 5,
            'drop_tail': True,
        },
    },

    'v1.1': {
        'train': {
            'labels': ('Ball', 'Bell', 'LightOn', 'Monkey', 'MusicOff', 'MusicOn'),
            'args': dict(n_data=40, env_args='playroom_fourrooms'),
            'group_size': 1,
            'drop_tail': True,
        },
        'val': {
            'labels': ('Ball', 'Bell', 'LightOn', 'Monkey', 'MusicOff', 'MusicOn'),
            'args': dict(n_data=15, env_args='playroom_fourrooms'),
            'group_size': 1,
            'drop_tail': True,
        },
        'test': {
            'labels': ('Ball', 'Bell', 'LightOn', 'Monkey', 'MusicOff', 'MusicOn'),
            'args': dict(n_data=100, env_args='playroom_fourrooms'),
            'group_size': 5,
            'drop_tail': True,
        },
    },

    'v1.2': {
        'train': {
            'labels': ('Ball', 'Bell', 'LightOn', 'Monkey', 'MusicOff', 'MusicOn'),
            'args': dict(n_data=40, env_args='playroom_fourrooms'),
            'group_size': 1,
            'drop_tail': True,
        },
        'val': {
            'labels': ('Ball', 'Bell', 'LightOn', 'Monkey', 'MusicOff', 'MusicOn'),
            'args': dict(n_data=15, env_args='playroom_fourrooms'),
            'group_size': 1,
            'drop_tail': True,
        },
        'test': {
            'labels': ('MusicOn>Ball', 'MusicOn>MusicOff', 'MusicOn>Ball>LightOn', 'LightOn>Bell', 'LightOn>Ball>MusicOn>Monkey', 'Monkey>LightOn'),
            'args': dict(n_data=100, env_args='playroom_fourrooms'),
            'group_size': 5,
            'drop_tail': True,
        },
    },

    'v1.3': {
        'train': {
            'labels': ('Ball', 'Bell', 'LightOn', 'Monkey', 'MusicOff', 'MusicOn', 'Ball>Bell', 'Bell>LightOn', 'LightOn>Monkey', 'Monkey>MusicOff', 'MusicOff>MusicOn', 'MusicOn>Ball'),
            'args': dict(n_data=40, env_args='playroom_fourrooms'),
            'group_size': 1,
            'drop_tail': True,
        },
        'val': {
            'labels': ('Ball', 'Bell', 'LightOn', 'Monkey', 'MusicOff', 'MusicOn', 'Ball>Bell', 'Bell>LightOn', 'LightOn>Monkey', 'Monkey>MusicOff', 'MusicOff>MusicOn', 'MusicOn>Ball'),
            'args': dict(n_data=15, env_args='playroom_fourrooms'),
            'group_size': 1,
            'drop_tail': True,
        },
        'test': {
            'labels': ('MusicOn>Ball>LightOn>Monkey>MusicOff>Bell', '(MusicOn&LightOn)>(Ball|Bell)>Monkey', 'MusicOn>(Ball&Bell)>Monkey'),
            'args': dict(n_data=100, env_args='playroom_fourrooms'),
            'group_size': 5,
            'drop_tail': True,
        },
    },
    'v2.single.large': {
        'train': DATA_PACK_CONFIG['v2.single.train_large'],
        'val': DATA_PACK_CONFIG['v2.single.val'],
        'test': DATA_PACK_CONFIG['v2.single.test'],
    },
    'v2.single.small': {
        'train': DATA_PACK_CONFIG['v2.single.train_small'],
        'val': DATA_PACK_CONFIG['v2.single.val'],
        'test': DATA_PACK_CONFIG['v2.single.test'],
    },

    'v2.pair.large': {
        'train': DATA_PACK_CONFIG['v2.pair.train_large'],
        'val': DATA_PACK_CONFIG['v2.pair.val'],
        'test': DATA_PACK_CONFIG['v2.pair.test'],
    },

    'v2.pair.small': {
        'train': DATA_PACK_CONFIG['v2.pair.train_small'],
        'val': DATA_PACK_CONFIG['v2.pair.val'],
        'test': DATA_PACK_CONFIG['v2.pair.test'],
    },

    'v2.complex.large': {
        'train': DATA_PACK_CONFIG['v2.complex.train_large'],
        'val': DATA_PACK_CONFIG['v2.complex.val'],
        'test': DATA_PACK_CONFIG['v2.complex.test'],
    },

    'v2.complex.small': {
        'train': DATA_PACK_CONFIG['v2.complex.train_small'],
        'val': DATA_PACK_CONFIG['v2.complex.val'],
        'test': DATA_PACK_CONFIG['v2.complex.test'],
    },

    'v2.c2s.large': {
        'train': DATA_PACK_CONFIG['v2.c2s.train_large'],
        'val': DATA_PACK_CONFIG['v2.c2s.val'],
        'test': DATA_PACK_CONFIG['v2.c2s.test'],
    },

    'v2.c2s.small': {
        'train': DATA_PACK_CONFIG['v2.c2s.train_small'],
        'val': DATA_PACK_CONFIG['v2.c2s.val'],
        'test': DATA_PACK_CONFIG['v2.c2s.test'],
    },

    'v2.mission1': {
        'train': DATA_PACK_CONFIG['v2.single.train_large'],
        'val': DATA_PACK_CONFIG['v2.single.val'],
        'test': DATA_PACK_CONFIG['v2.mission1.test'],
    },

    'v2.mission2': {
        'train': DATA_PACK_CONFIG['v2.single.playroom_maze1.train'],
        'val': DATA_PACK_CONFIG['v2.single.playroom_maze1.val'],
        'test': DATA_PACK_CONFIG['v2.mission2.test'],
    },

    'v2.debug': {
        'train': DATA_PACK_CONFIG['v2.debug.train'],
        'val': DATA_PACK_CONFIG['v2.debug.val'],
        'test': DATA_PACK_CONFIG['v2.debug.test'],
    },
    'v2.all.large': {
        'labels': ['Ball', 'Bell', 'LightOn', 'Monkey', 'MusicOff', 'MusicOn'] + COMPLEX_TRAINING_LABELS[1:],
        'train': [DATA_PACK_CONFIG['v2.single.train_large'], DATA_PACK_CONFIG['v2.c2s.train_large']],
        'val': [DATA_PACK_CONFIG['v2.single.val'], DATA_PACK_CONFIG['v2.c2s.val']],
        'test': [DATA_PACK_CONFIG['v2.single.test'], DATA_PACK_CONFIG['v2.c2s.test']],
    },
    'v2.all.small': {
        'labels': ['Ball', 'Bell', 'LightOn', 'Monkey', 'MusicOff', 'MusicOn'] + COMPLEX_TRAINING_LABELS[1:],
        'train': [DATA_PACK_CONFIG['v2.single.train_small'], DATA_PACK_CONFIG['v2.c2s.train_small']],
        'val': [DATA_PACK_CONFIG['v2.single.val'], DATA_PACK_CONFIG['v2.c2s.val']],
        'test': [DATA_PACK_CONFIG['v2.single.test'], DATA_PACK_CONFIG['v2.c2s.test']],
    },
}

DATA_GENERATOR = {
    '1.': PlayRoomV1,
}


class PlayroomDataLoader(DataLoader):
    NAME = "ToyRobotPlayroom"

    def __init__(self, data_version=None, force_regen=False, *args, **kwargs):
        super().__init__()
        self.splits = ('train', 'val', 'test')
        self.data_version = data_version
        self.config = DATA_CONFIG[data_version]
        gen_class = PlayRoomV1
        for lv in DATA_GENERATOR:
            if lv in data_version:
                gen_class = DATA_GENERATOR[lv]

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
        self.group_indices, self.label_group_indices = self.split_data()

    @classmethod
    def build_env_from_args(cls, env_args):
        return PlayRoomV1.build_env_from_args(env_args)

    def get_traj_checker(self, plan_search=False):
        from .playroom_checker import PlayroomChecker

        return PlayroomChecker(self.env_args, plan_search=plan_search)

    def get_all_labels(self):
        return set(self.labels['train']).union(set(self.labels['test'])).union(set(self.labels['val']))

    def get_env_args(self):
        if isinstance(self.env_args, str):
            return DEFAULT_ENV_ARGS_V1[self.env_args]
        else:
            return self.env_args

    def split_data(self):
        # print(max(len(sample['traj'][0]) for sample in self.data))
        # exit()
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
            rng.shuffle(group_indices[split])
        return group_indices, label_group_indices

    def group_size(self, split):
        return self.config[split]['group_size'] if not isinstance(self.config[split], list) else self.config[split][0]['group_size']

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

        actual_batch_size = (batch_size - 1) // self.group_size(split) + 1
        indices_batches = []

        for indices in indices_list:
            for k in range((len(indices) + actual_batch_size - 1) // actual_batch_size):
                indices_batches.append(indices[k * actual_batch_size: min((k + 1) * actual_batch_size, len(indices))])

        if split == 'train' or split == 'val':
            rng.shuffle(indices_batches)
            if n_batches is not None:
                indices_batches = indices_batches[:n_batches]

        print('#batches=', len(indices_batches))
        return indices_batches

    def batches(self, split, batch_size=10, n_batches=None, val_n_batches=None, test_batch_size=None, by_label=False, seed=None, **kwargs):
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
        pspace = self.build_env_from_args(env_args).pspace
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
                acts = ['eff_MusicOn', 'eff_LightOn', 'eff_Bell', 'eff_Ball', 'eff_Monkey', 'eff_MusicOff']
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
