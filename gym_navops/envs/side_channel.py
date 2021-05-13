#!/usr/bin/python3
# -*- coding: utf-8 -*-
import uuid
from queue import Queue

from mlagents_envs.side_channel.side_channel import SideChannel, IncomingMessage


class EpisodeSideChannel(SideChannel):

    def __init__(self):
        super().__init__(uuid.UUID("0a6d3d29-0130-475c-a98c-ae665a752cbc"))
        self._event_queue = Queue()

    def on_message_received(self, msg: IncomingMessage):
        blue_wins = msg.read_bool()
        self.event_queue.put(blue_wins)
        print(f'[EpisodeSideChannel] on_message_received: {blue_wins}')

    @property
    def event_queue(self):
        return self._event_queue


if __name__ == "__main__":
    pass
