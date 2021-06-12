#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
from datetime import datetime
from subprocess import Popen
from threading import Thread
from typing import Optional

import gym
import grpc

with open(os.path.join(os.path.dirname(__file__), '..'))


class GrpcEnvironment:

    def __init__(self, path: str = None):
        self._process: Optional[Popen] = None
        self._grpc_client = NavOpsGrpcClient(env=self)
        # grpc_client.call_environment_step()

        # self._observation_space = gym.spaces.Box(-1.0, 1.0, shape=tuple(config["NavOpsMultiDiscrete"]["observation_space"]["shape"]))
        self._action_space = gym.spaces.MultiDiscrete(config["NavOpsMultiDiscrete"]["action_space"]["nvec"])

        if path is not None:
            self._env_thread = Thread(target=self._run_env_subprocess, args=(path,))
            self._env_thread.daemon = True
            self._env_thread.start()

        while not self._grpc_client.request_heartbeat():
            print(f'[{datetime.now().isoformat()}] [{self.__class__.__name__}] Request heartbeat..')
            time.sleep(1)

    """
    def run(self, path):
        self._process = Popen(path)
        print('process:', self._process)
        # self._process.poll()
        self._process.wait()
        print('clear')
    """

    def step(self):
        response = self._grpc_client.call_environment_step()
        observation = None
        reward = response.reward
        done = response.done
        info = {}
        return observation, reward, done, info

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

    def _run_env_subprocess(self, path: str):
        self._process = Popen(path)
        # self._process.poll()
        self._process.wait()
        print(f'[{datetime.now().isoformat()}] [{self.__class__.__name__}] Subprocess is terminated.')

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

    def call_environment_step(self):
        request = navops_service_pb2.EnvironmentStepRequest(
            actions=[
                navops_service_pb2.DiscreteActionSpace(
                    maneuver_action_id=np.random.randint(self.env.action_space.nvec[0]),
                    attack_action_id=np.random.randint(self.env.action_space.nvec[1])
                )
            ]
        )
        # print('request.actions:', request.actions)
        response = self._stub.CallEnvironmentStep(request)
        # print(f'[{datetime.now().isoformat()}] [{self.__class__.__name__}] CallEnvironmentStep(response={response})')

        return response

    @property
    def env(self):
        return self._env


if __name__ == "__main__":
    pass
