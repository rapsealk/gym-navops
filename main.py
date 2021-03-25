#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
from datetime import datetime
from itertools import count

import numpy as np

import gym
import gym_navops   # noqa: F401


def main():
    build_path = os.path.abspath(os.path.join('/', 'Users', 'rapsealk', 'Desktop', 'NavOpsMultihead', 'NavOps'))
    # build_path = os.path.join(os.path.dirname(__file__), 'NavOps')
    env = gym.make('NavOpsDiscrete-v0', override_path=build_path, no_graphics=True)

    for episode in count(1):
        observation = env.reset()
        for frame in count(1):
            actions = np.concatenate([
                np.random.randint(0, env.action_space.nvec[0], size=(2, 1)),
                np.random.randint(0, env.action_space.nvec[1], size=(2, 1))
            ], axis=1)
            # if np.any(actions == 5):
            #     print(f'[{datetime.now().isoformat()}] Episode#{episode} frame#{frame}: Attack {actions}')
            next_obs, reward, done, info = env.step(actions)
            # if np.any(reward != 0):
            #     print(f'[{datetime.now().isoformat()}] Episode#{episode} frame#{frame}: Reward {reward}')
            if done:
                break
        break


if __name__ == "__main__":
    main()
