#/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import time
from concurrent import futures
from datetime import datetime

import numpy as np
import grpc
# from google.protobuf import empty_pb2

import navops_service_pb2
import navops_service_pb2_grpc

parser = argparse.ArgumentParser()
parser.add_argument('--client', action='store_true', default=False)
parser.add_argument('-p', '--port', default=9090, type=int)
args = parser.parse_args()


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


class NavOpsGrpcClient:

    def __init__(self, port=9090):
        channel = grpc.insecure_channel(f'localhost:{port}')
        self._stub = navops_service_pb2_grpc.NavOpsGrpcServiceStub(channel)

    def call_environment_step(self):
        request = navops_service_pb2.EnvironmentStepRequest()
        response = self._stub.CallEnvironmentStep(request)
        # print(f'[{datetime.now().isoformat()}] [{self.__class__.__name__}] CallEnvironmentStep(response={response})')


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
    main()
