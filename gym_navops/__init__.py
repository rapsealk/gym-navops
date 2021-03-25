#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
from gym.envs.registration import register

register(
    id='NavOps-v0',
    entry_point='gym_navops.envs:NavOpsEnv',
    kwargs={
        'worker_id': 0,
        'base_port': None,
        'seed': 0,
        'no_graphics': False,
        'override_path': None,
        'mock': False,
        '_build': 'NavOps'
    }
)

register(
    id='NavOpsDiscrete-v0',
    entry_point='gym_navops.envs:NavOpsEnv',
    kwargs={
        'worker_id': 0,
        'base_port': None,
        'seed': 0,
        'no_graphics': False,
        'override_path': None,
        'mock': False,
        '_build': 'NavOpsDiscrete'
    }
)

register(
    id='NavOpsMultiDiscrete-v0',
    entry_point='gym_navops.envs:NavOpsEnv',
    kwargs={
        'worker_id': 0,
        'base_port': None,
        'seed': 0,
        'no_graphics': False,
        'override_path': None,
        'mock': False,
        '_build': 'NavOpsMultiDiscrete'
    }
)

"""
register(
    id='NavOpsDiscreteSkipFrame-v0',
    entry_point='gym_navops.envs:NavOpsEnv',
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
"""
