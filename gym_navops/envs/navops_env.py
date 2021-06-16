#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import os
import sys
import json
import time
from datetime import datetime
from typing import Optional
from multiprocessing import Lock
from subprocess import Popen
from threading import Thread

import numpy as np
import gym
import grpc

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import navops_service_pb2       # noqa: E402
import navops_service_pb2_grpc  # noqa: E402

with open(os.path.join(os.path.dirname(__file__), 'config.json')) as f:
    config = ''.join(f.readlines())
    config = json.loads(config)


class NavOpsEnv(gym.Env):

    def __init__(self, build_path=None, port=9090):
        self._process: Optional[Popen] = None
        self._grpc_client = NavOpsGrpcClient(env=self, port=port)

        self._observation_space = gym.spaces.Box(-1.0, 1.0, shape=tuple(config["NavOpsMultiDiscrete"]["observation_space"]["shape"]))
        self._action_space = gym.spaces.MultiDiscrete(config["NavOpsMultiDiscrete"]["action_space"]["nvec"])

        self._skip_frame = 4

        if build_path is not None:
            self._env_thread = Thread(target=self._run_env_subprocess, args=(build_path, port))
            self._env_thread.daemon = True
            self._env_thread.start()

        while not self._grpc_client.request_heartbeat():
            print(f'[{datetime.now().isoformat()}] [{self.__class__.__name__}] Request heartbeat..')
            time.sleep(1)

    def step(self, action):
        reward = 0.0
        for _ in range(self._skip_frame):
            time.sleep(0.03)
            response = self._grpc_client.call_environment_step(action)
            reward += response.reward
            if response.done:
                break
        info = {
            'obs': response.obs,
            'win': (response.reward == 1.0)
        }
        observation = self._decode_observation(response.obs)
        # reward = response.reward
        done = response.done
        return observation, reward, done, info

    def reset(self):
        # FIXME: EnvironmentResetRequest
        zero_action = [0, 0]
        response = self._grpc_client.call_environment_step(zero_action)
        return self._decode_observation(response.obs)

    def render(self):
        pass

    def close(self):
        if self._process is None:
            return
        return_code = self._process.poll()
        print('[GrpcEnvironment] close:', return_code)
        if not return_code:
            self._process.terminate()
        self._process.wait()

        # self._env_thread.interrupt()
        self._env_thread.join()

    def _run_env_subprocess(self, path: str, port: int = 9090):
        self._process = Popen([path, f'--port={port}'])
        # self._process.poll()
        self._process.wait()
        print(f'[{datetime.now().isoformat()}] [{self.__class__.__name__}] Subprocess is terminated.')

    def _decode_observation(self, obs: navops_service_pb2.Observation):
        buffer = []
        for fleet in obs.fleets:
            buffer.extend([
                fleet.team_id,
                fleet.hp,
                fleet.fuel,
                fleet.destroyed,
                fleet.detected,
                *(fleet.position.x, fleet.position.y),
                *(fleet.rotation.cos, fleet.rotation.sin),
                fleet.timestamp
            ])
        for location in obs.locations:
            buffer.extend([
                location.dominance,
                *(location.position.x, location.position.y)
            ])
        buffer.extend(obs.target_index_onehot)
        buffer.extend(obs.raycast_hits)
        for battery in obs.batteries:
            buffer.extend([
                *(battery.rotation.cos, battery.rotation.sin),
                battery.reloaded,
                battery.cooldown,
                battery.damaged,
                battery.repair_progress
            ])
        buffer.append(obs.ammo)
        buffer.extend(obs.speed_level_onehot)
        buffer.extend(obs.steer_level_onehot)
        for position in obs.obstacle_positions:
            buffer.extend(*[position.x, position.y])
        obs_np = np.array(buffer, dtype=np.float32)
        return np.expand_dims(obs_np, axis=0)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space


class NavOpsGrpcClient:

    def __init__(self, env=None, port=9090):
        self._env = env
        channel = grpc.insecure_channel(f'localhost:{port}')
        self._stub = navops_service_pb2_grpc.NavOpsGrpcServiceStub(channel)

    def request_heartbeat(self):
        request = navops_service_pb2.HeartbeatRequest()
        try:
            response = self._stub.RequestHeartbeat(request)
            sys.stdout.write(f'[{datetime.now().isoformat()}] [{self.__class__.__name__}] RequestHeartbeat: {response.succeed}\n')
            return response.succeed
        except Exception as e:
            sys.stderr.write(f'[{datetime.now().isoformat()}] [{self.__class__.__name__}] RequestHeartbeat: {e}\n')
        return False

    def call_environment_step(self, action):
        request = navops_service_pb2.EnvironmentStepRequest(
            actions=[
                navops_service_pb2.DiscreteActionSpace(
                    maneuver_action_id=action[0],
                    attack_action_id=action[1]
                )
            ]
        )
        response = self._stub.CallEnvironmentStep(request)

        return response

    @property
    def env(self):
        return self._env


if __name__ == "__main__":
    pass
