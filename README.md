# gym-rimpac
![GitHub Actions CI](https://github.com/rapsealk/gym-rimpac/workflows/Lint/badge.svg)
![Python 3.8](https://img.shields.io/badge/Python-3.8-blue.svg?logo=python)

```python
import gym
import gym_rimpac

env = gym.make('Rimpac-v0')
env = gym.make('RimpacDiscrete-v0')

# For more details..
env = gym.make(
    'Rimpac[Discrete]-v0',
    worker_id=0,
    base_port=None,
    seed=0,
    no_graphics=False,
    mock=False,
    _discrete=True
)
```
