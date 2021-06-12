#/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import os
import sys
import time
import json
from concurrent import futures
from datetime import datetime
from typing import Optional
from itertools import count
from subprocess import Popen
from threading import Thread

import gym
import numpy as np
import grpc
# from google.protobuf import empty_pb2

import navops_service_pb2
import navops_service_pb2_grpc

with open(os.path.join(os.path.dirname(__file__), '..', 'config.json')) as f:
    config = ''.join(f.readlines())
    config = json.loads(config)

parser = argparse.ArgumentParser()
parser.add_argument('--client', action='store_true', default=False)
parser.add_argument('-p', '--port', default=9090, type=int)
args = parser.parse_args()


class GrpcEnvironment:

    def __init__(self, path: str = None):
        self._process: Optional[Popen] = None
        self._grpc_client = NavOpsGrpcClient(env=self)

        self._observation_space = gym.spaces.Box(-1.0, 1.0, shape=tuple(config["NavOpsMultiDiscrete"]["observation_space"]["shape"]))
        self._action_space = gym.spaces.MultiDiscrete(config["NavOpsMultiDiscrete"]["action_space"]["nvec"])

        if path is not None:
            self._env_thread = Thread(target=self._run_env_subprocess, args=(path,))
            self._env_thread.daemon = True
            self._env_thread.start()

        while not self._grpc_client.request_heartbeat():
            print(f'[{datetime.now().isoformat()}] [{self.__class__.__name__}] Request heartbeat..')
            time.sleep(1)

    def step(self):
        response = self._grpc_client.call_environment_step()
        observation = response.obs
        reward = response.reward
        done = response.done
        info = {'numpy': self._decode_observation(observation)}
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

    def _decode_observation(self, obs: navops_service_pb2.Observation):
        """
        message Observation {
            float ammo = 5;
            repeated float speed_level_onehot = 6;
            repeated float steer_level_onehot = 7;
        }
        """
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
        return np.array(buffer, dtype=np.float32)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space


"""
class NavOpsGrpcServer(navops_service_pb2_grpc.NavOpsGrpcServiceServicer):

    def __init__(self):
        pass

    def CallEnvironmentStep(self, request, context):
        print(f'[{datetime.now().isoformat()}] [{self.__class__.__name__}] CallEnvironmentStep(request={request})')
        response = navops_service_pb2.EnvironmentStepResponse(
            obs=navops_service_pb2.Observation(),
            reward=np.random.uniform(-1.0, 1.0, (1,)).tolist()[0],
            done=True
        )
        return response
"""


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


def main():
    if args.client:
        client = NavOpsGrpcClient(port=args.port)
        while input('x: ') != 'q':
            t = time.time()
            for _ in range(1000):
                client.call_environment_step()
            print(f'Time: {time.time() - t}s')
    else:
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
        navops_service_pb2_grpc.add_NavOpsGrpcServiceServicer_to_server(NavOpsGrpcServer(), server)
        server.add_insecure_port(f'[::]:{args.port}')
        print('Server is running..')
        server.start()
        server.wait_for_termination()


if __name__ == "__main__":
    # main()

    # path = os.path.join('C:\\', 'Users', 'rapsealk', 'Desktop', 'NavOpsGrpc', 'NavOps.exe')
    path = None
    env = GrpcEnvironment(path)

    for episode in count(1):
        t = time.time()
        # for _ in range(128):
        done = False
        rewards = []
        steps = 0
        while not done:
            steps += 1
            obs, reward, done, info = env.step()
            rewards.append(reward)
            time.sleep(0.1)
        print(f'Episode: {episode} ({steps} steps, {time.time() - t}s)')
        print(obs)
        print(info['numpy'].shape)
        print(rewards[-5:])
        if episode == 2:
            break
    env.close()
