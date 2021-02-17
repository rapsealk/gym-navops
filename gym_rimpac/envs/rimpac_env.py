#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import os
import sys
import json
import zipfile
from collections import namedtuple
from urllib import request

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


def get_build_dist(platform):
    dist = {
        "win32": "Windows",
        "linux": "Linux",
        "darwin": "MacOS"
    }
    return dist.get(platform, None)


def _build_download_hook(url: str):
    block_size = 0  # Enclosing

    def download_hook(blocknum, bs, size):
        nonlocal block_size
        block_size += bs
        print(f'\rDownload {url}: ({block_size / size * 100:.2f}%)', end='')
    return download_hook


class RimpacEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        worker_id=0,
        base_port=None,
        seed=0,
        no_graphics=False,
        mock=False,
        _discrete=False
    ):
        self._mock = mock
        self._discrete = _discrete

        self._observation_space = gym.spaces.Box(-1.0, 1.0, shape=tuple(config["Rimpac"]["observation_space"]["shape"]))
        if _discrete:
            self._action_space = gym.spaces.Discrete(config["RimpacDiscrete"]["action_space"]["n"])
        else:
            self._action_space = gym.spaces.Box(-1.0, 1.0, shape=tuple(config["Rimpac"]["action_space"]["shape"]))

        if mock:
            self._n = 2
            return

        build_dir_path = os.path.join(os.path.dirname(__file__), 'Rimpac')
        if not os.path.exists(build_dir_path):
            os.mkdir(build_dir_path)
        build_name = 'RimpacDiscrete' if _discrete else 'Rimpac'
        build_path = os.path.join(build_dir_path, f'{build_name}-v0')
        download_path = build_path + '.zip'
        if not os.path.exists(build_path):
            if not os.path.exists(download_path):
                dist = get_build_dist(sys.platform)
                url = f'https://github.com/rapsealk/gym-rimpac/releases/download/v0.1.0/{build_name}-{dist}-x86_64.2019.4.20f1.zip'
                request.urlretrieve(url, download_path, reporthook=_build_download_hook(url))
            with zipfile.ZipFile(download_path) as unzip:
                unzip.extractall(build_path)
        os.environ['RIMPAC_PATH'] = build_path

        try:
            file_name = os.environ['RIMPAC_PATH']
        except KeyError:
            raise Exception('Unable to find Rimpac.')

        self._env = UnityEnvironment(file_name, worker_id=worker_id, base_port=base_port, seed=seed, no_graphics=no_graphics)

        self.steps = []
        self.observation = []

    def step(self, action):
        if self._mock:
            obs = np.random.normal(0, 1, (self._n,)+self.observation_space.shape)
            reward = np.zeros((self._n,))
            done = np.random.normal(0, 1) > 0.8
            return obs, reward, done, {'win': np.random.randint(0, 2)}

        done, info = False, {}
        for _ in range(SKIP_FRAMES):
            observation = self._update_environment_state()
            for team_id, (decision_steps, terminal_steps) in enumerate(self.steps):
                if terminal_steps.reward.shape[0] > 0:
                    info['win'] = team_id if terminal_steps.reward[0] > 0 else (1 - team_id)
                    done = True
                    break
                for i, behavior_name in enumerate(self.behavior_names):
                    if self._discrete:
                        discrete_action = ActionTuple()
                        discrete_action.add_discrete(np.array([action[i]])[np.newaxis, :])
                        self._env.set_actions(behavior_name, discrete_action)
                    else:
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
        if self._mock:
            return np.random.normal(0, 1, (self._n,)+self.observation_space.shape)

        self._env.reset()
        self.behavior_names = [name for name in self._env.behavior_specs.keys()]
        print('RimpacEnv.behavior_names:', self.behavior_names)

        observation = self._update_environment_state()

        return observation

    def render(self, mode='human', close=False):
        pass

    def close(self):
        if self._mock:
            return

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
