#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import mujoco_py
from .rotation_v2 import quat2mat


def as_rotation(r):
    if isinstance(r, np.ndarray) and r.shape == (3, 3):
        return r
    if isinstance(r, np.ndarray) and r.shape == (4,):
        return quat2mat(r)
    raise ValueError('Invalid rotation: {}.'.format(r))


def quat_create(axis, angle):
    quat = np.zeros([4], dtype='float')
    mujoco_py.functions.mju_axisAngle2Quat(quat, axis, angle)
    return quat
