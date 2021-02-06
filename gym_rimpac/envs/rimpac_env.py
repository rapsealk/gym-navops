#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import os
import json
from collections import namedtuple

import numpy as np
import gym
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple

Observation = namedtuple('Observation',
                         ('decision_steps', 'terminal_steps'))

with open(os.path.join(os.path.dirname(__file__), '..', '..', 'config.json')) as f:
    config = ''.join(f.readlines())
    config = json.loads(config)

SKIP_FRAMES = 4


class RimpacEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        worker_id=0,
        base_port=None,
        seed=0,
        no_graphics=False
    ):
        try:
            file_name = os.environ['RIMPAC_PATH']
        except KeyError:
            raise Exception('Unable to find Rimpac.')

        self._env = UnityEnvironment(file_name, worker_id=worker_id, base_port=base_port, seed=seed, no_graphics=no_graphics)
        self._action_space = gym.spaces.Box(-1.0, 1.0, shape=tuple(config["action_space"]["shape"]))
        self._observation_space = gym.spaces.Box(-1.0, 1.0, shape=tuple(config["observation_space"]["shape"]))

        self.steps = []
        self.observation = []

    def step(self, action):
        done, info = False, {}
        for _ in range(SKIP_FRAMES):
            observation = self._update_environment_state()
            for team_id, (decision_steps, terminal_steps) in enumerate(self.steps):
                if terminal_steps.reward.shape[0] > 0:
                    info['win'] = team_id if terminal_steps.reward[0] > 0 else (1 - team_id)
                    done = True
                    break
                for i, behavior_name in enumerate(self.behavior_names):
                    if not done:
                        continuous_action = ActionTuple()
                        action_ = action[i][np.newaxis, :]
                        continuous_action.add_continuous(action_)
                    else:
                        continuous_action = ActionTuple()
                        continuous_action.add_continuous(np.zeros((0, 6)))
                    self._env.set_actions(behavior_name, continuous_action)
            if done:
                break
        if done:
            if 0 in observation.shape:
                observation = self.observation_cache
            reward = np.array([np.squeeze(obs.terminal_steps.reward) for obs in self.observation])
        else:
            self._env.step()
            observation = self._update_environment_state()
            if 0 in observation.shape:
                observation = self.observation_cache
            self.observation_cache = observation
            reward = np.array([np.squeeze(obs.decision_steps.reward) for obs in self.observation])
            if 0 in reward.shape:
                reward = np.array([np.squeeze(obs.terminal_steps.reward) for obs in self.observation])

        return observation, np.squeeze(reward), done, info

    def reset(self):
        self._env.reset()
        self.behavior_names = [name for name in self._env.behavior_specs.keys()]
        print('RimpacEnv.behavior_names:', self.behavior_names)

        observation = self._update_environment_state()

        return observation

    def render(self, mode='human', close=False):
        pass

    def close(self):
        self._env.close()

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    def _update_environment_state(self):
        self.steps = [self._env.get_steps(behavior_name=behavior) for behavior in self.behavior_names]
        self.observation = [Observation(*step) for step in self.steps]
        return np.array([obs.decision_steps.obs for obs in self.observation]).squeeze()


class MockRimpacEnv(gym.Env):

    def __init__(self, n=2):
        self._n = n
        self._action_space = gym.spaces.Box(-1.0, 1.0, shape=tuple(config["action_space"]["shape"]))
        self._observation_space = gym.spaces.Box(-1.0, 1.0, shape=tuple(config["observation_space"]["shape"]))

    def step(self, action):
        obs = np.random.normal(0, 1, (self._n,)+self.observation_space.shape)
        return obs, 0, False, {}

    def reset(self):
        return np.random.normal(0, 1, (self._n,)+self.observation_space.shape)

    def close(self):
        pass

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space
