#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import sys

from urllib import request


def get_build_dist(platform):
    dist = {
        "win32": "Windows",
        "linux": "Linux",
        "darwin": "MacOS"
    }
    return dist.get(platform, None)


class NavOpsDownloader:

    def download(self, build_name: str, path: str, version='v0.1.0', unity_version='2020.3.4f1'):
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
            width = os.get_terminal_size().columns
            message = '\rDownload' + f'{url}: ({block_size / size * 100:.2f}%)'
            if len(message) > width:
                message = message[:width//2-3] + '...' + message[-width//2:]
            sys.stdout.write(message)
            if block_size == size:
                sys.stdout.write('\n')
        return download_hook


if __name__ == "__main__":
    pass
