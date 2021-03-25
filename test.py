#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import unittest

import numpy as np

import gym
import gym_navops   # noqa: F401


class TestNavOpsEnvironment(unittest.TestCase):

    def setUp(self):
        self._n = 2
        self._env = gym.make('NavOps-v0')
        self._env.reset()
        self._mock = gym.make('NavOps-v0', mock=True)

    def test_environment_space(self):
        self.assertEqual(
            self._env.observation_space.shape,
            self._mock.observation_space.shape
        )
        self.assertEqual(
            self._env.action_space.shape,
            self._mock.action_space.shape
        )
        action = np.random.normal(0, 1, (self._n,)+self._env.action_space.shape)
        obs, _, _, _ = self._env.step(action)
        mobs, _, _, _ = self._mock.step(action)
        self.assertEqual(obs.shape, mobs.shape)

    def tearDown(self):
        self._env.close()
        self._mock.close()


if __name__ == "__main__":
    unittest.main()