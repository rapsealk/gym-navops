#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
from gym.envs.registration import register

register(
    id='Rimpac-v0',
    entry_point='gym_rimpac.envs:RimpacEnv',
    kwargs={
        'worker_id': 0,
        'base_port': None,
        'seed': 0,
        'no_graphics': False,
        'override_path': None,
        'mock': False,
        '_discrete': False,
        '_skip_frame': False
    }
)

register(
    id='RimpacDiscrete-v0',
    entry_point='gym_rimpac.envs:RimpacEnv',
    kwargs={
        'worker_id': 0,
        'base_port': None,
        'seed': 0,
        'no_graphics': False,
        'override_path': None,
        'mock': False,
        '_discrete': True,
        '_skip_frame': False
    }
)

register(
    id='RimpacDiscreteSkipFrame-v0',
    entry_point='gym_rimpac.envs:RimpacEnv',
    kwargs={
        'worker_id': 0,
        'base_port': None,
        'seed': 0,
        'no_graphics': False,
        'override_path': None,
        'mock': False,
        '_discrete': True,
        '_skip_frame': True
    }
)
