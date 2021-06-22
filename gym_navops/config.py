#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import json

with open(os.path.join(os.path.dirname(__file__), 'envs', 'config.json'), 'r') as f:
    config = ''.join(f.readlines())
    config = json.loads(config)

    EnvironmentConfig = config
