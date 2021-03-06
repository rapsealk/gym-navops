#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
from datetime import datetime
from itertools import count

import numpy as np

import gym
import gym_rimpac   # noqa: F401


def main():
    env = gym.make('RimpacDiscrete-v0', override_path=os.path.join(os.path.dirname(__file__), 'Rimpac'))

    for episode in count(1):
        observation = env.reset()
        for frame in count(1):
            actions = np.random.randint(0, env.action_space.n, size=(2,))
            if np.any(actions == 5):
                print(f'[{datetime.now().isoformat()}] Episode#{episode} frame#{frame}: Attack {actions}')
            next_obs, reward, done, info = env.step(actions)
            if np.any(reward != 0):
                print(f'[{datetime.now().isoformat()}] Episode#{episode} frame#{frame}: Reward {reward}')
            if done:
                break
        break


if __name__ == "__main__":
    main()
