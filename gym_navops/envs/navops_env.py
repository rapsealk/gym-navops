#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import os
import sys
import json
import zipfile
from collections import namedtuple
from datetime import datetime
from multiprocessing import Lock
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


def get_build_dist(platform):
    dist = {
        "win32": "Windows",
        "linux": "Linux",
        "darwin": "MacOS"
    }
    return dist.get(platform, None)


class NavOpsDownloader:

    def download(self, build_name: str, path: str, version='v0.1.0', unity_version='2020.3.1f1'):
        dist = get_build_dist(sys.platform)
        url = f'https://github.com/rapsealk/gym-navops/releases/download/{version}/{build_name}-{dist}-x86_64.{unity_version}.zip'
        try:
            request.urlretrieve(url, path, reporthook=self._build_download_hook(url))
        except KeyboardInterrupt:
            if os.path.exists(path):
                os.remove(path)

    def _build_download_hook(self, url: str):
        block_size = 0  # Enclosing

        def download_hook(blocknum, bs, size):
            nonlocal block_size
            block_size += bs
            sys.stdout.write(f'\rDownload {url}: ({block_size / size * 100:.2f}%) ')
            if block_size == size:
                sys.stdout.write('\n')
        return download_hook


class NavOpsEnv(gym.Env):

    metadata = {'render.modes': ['human']}
    __lock = Lock()

    def __init__(
        self,
        worker_id=0,
        base_port=None,
        seed=0,
        no_graphics=False,
        override_path=None,
        _build='NavOps',
        _n=2
    ):
        self._build = _build

        if _build == 'NavOps':
            self._observation_space = gym.spaces.Box(-1.0, 1.0, shape=tuple(config["NavOps"]["observation_space"]["shape"]))
            self._action_space = gym.spaces.Box(-1.0, 1.0, shape=tuple(config["NavOps"]["action_space"]["shape"]))
        elif _build == 'NavOpsDiscrete':
            self._observation_space = gym.spaces.Box(-1.0, 1.0, shape=tuple(config["NavOpsDiscrete"]["observation_space"]["shape"]))
            self._action_space = gym.spaces.Discrete(config["NavOpsDiscrete"]["action_space"]["n"])
        elif _build == 'NavOpsMultiDiscrete':
            self._observation_space = gym.spaces.Box(-1.0, 1.0, shape=tuple(config["NavOpsMultiDiscrete"]["observation_space"]["shape"]))
            self._action_space = gym.spaces.MultiDiscrete(config["NavOpsMultiDiscrete"]["action_space"]["nvec"])

        self._n = _n
        if _n == 1:
            _build += 'Single'

        if override_path:
            build_path = override_path
        else:
            build_dir_path = os.path.join(os.path.dirname(__file__), 'NavOps')
            if not os.path.exists(build_dir_path):
                os.mkdir(build_dir_path)
            build_path = os.path.join(build_dir_path, f'{self._build}-v0')
            with self.__lock:
                if not os.path.exists(build_path):
                    download_path = build_path + '.zip'
                    if not os.path.exists(download_path):
                        NavOpsDownloader().download(_build, download_path)
                    with zipfile.ZipFile(download_path) as unzip:
                        unzip.extractall(build_path)

        self._env = UnityEnvironment(build_path, worker_id=worker_id, base_port=base_port, seed=seed, no_graphics=no_graphics)

        self._skip_frames = 4

        self.steps = []
        self.observation = []

    def step(self, action):
        done, info = False, {'win': -1}

        skip_frame_step = self._skip_frame(action)
        for _ in range(self._skip_frames):
            observation, done, terminal_rewards = next(skip_frame_step)
            if done:
                break

        if done:
            if self._n == 1:
                if terminal_rewards[0] == 1.0: info['win'] = 0
                elif terminal_rewards[0] == -1.0: info['win'] = 1
                # elif terminal_rewards[0] == 0.0: info['win'] = -1
            else:
                for i, terminal_reward in enumerate(terminal_rewards):
                    if terminal_reward == 1.0:
                        info['win'] = i
                        break
            self._env.step()

        if done:
            if 0 in observation.shape:
                observation = self.observation_cache
            reward = np.array([np.squeeze(obs.terminal_steps.reward) for obs in self.observation])
            print(f'[gym-navops] TerminalRewards: {reward}')
        else:
            self._env.step()
            observation = self._update_environment_state()
            if 0 in observation.shape:
                observation = self.observation_cache
            self.observation_cache = observation
            reward = np.array([np.squeeze(obs.decision_steps.reward) for obs in self.observation])
            if 0 in reward.shape:
                reward = np.array([np.squeeze(obs.terminal_steps.reward) for obs in self.observation])

        if (reward := np.squeeze(reward)).ndim == 0:
            reward = np.expand_dims(reward, axis=0)

        return observation, reward, done, info

    def reset(self):
        self._env.reset()
        self.behavior_names = [name for name in self._env.behavior_specs.keys()]
        print(f'[{datetime.now().isoformat()}] NavOpsEnv.Reset() => behavior_names: {self.behavior_names}')

        return self._update_environment_state()

    def render(self, mode='human', close=False):
        pass

    def close(self):
        self._env.close()

    def _skip_frame(self, action):
        concat_actions = np.concatenate((
            np.expand_dims(action, axis=0),
            np.zeros((self._skip_frames-1,) + action.shape)
        ))

        for action in concat_actions:
            print('[gym-navops] skip_frame.action:', action)
            # 1 step 3 skips
            yield self._step(action)

    def _step(self, action):
        observation = self._update_environment_state()
        done = False
        terminal_rewards = np.zeros((self._n,))
        for team_id, (_, terminal_steps) in enumerate(self.steps):
            if terminal_steps.reward.shape[0] > 0:
                done = True
                terminal_rewards[team_id] = terminal_steps.reward[0]
                continue

            for i, behavior_name in enumerate(self.behavior_names):
                action_tuple = ActionTuple()
                if self._build == 'NavOps':
                    action_tuple.add_continuous(action[i][np.newaxis, :])
                elif self._build == 'NavOpsDiscrete':
                    action_tuple.add_discrete(np.expand_dims([action[i]], axis=0))
                elif self._build == 'NavOpsMultiDiscrete':
                    action_tuple.add_discrete(np.asarray([action[i]]))
                self._env.set_actions(behavior_name, action_tuple)

        self._env.step()

        return observation, done, terminal_rewards

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    def _update_environment_state(self):
        self.steps = [self._env.get_steps(behavior_name=behavior) for behavior in self.behavior_names]
        self.observation = [Observation(*step) for step in self.steps]
        obs = np.array([obs.decision_steps.obs for obs in self.observation]).squeeze(0).squeeze(0) # .squeeze()
        return obs
