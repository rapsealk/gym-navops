#!/usr/bin/python3
# -*- coding: utf-8 -*-
import uuid

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.side_channel import (
    SideChannel,
    IncomingMessage,
    OutgoingMessage
)
import numpy as np


class AimChannel(SideChannel):

    def __init__(self) -> None:
        super().__init__(uuid.UUID("0a6d3d29-0130-475c-a98c-ae665a752cbc"))

    def on_message_received(self, msg: IncomingMessage) -> None:
        print(msg.read_string())

    def send_string(self, data: str) -> None:
        msg = OutgoingMessage()
        msg.write_string(data)
        super().queue_message_to_send(msg)
