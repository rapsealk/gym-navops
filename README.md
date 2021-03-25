# gym-navops
![GitHub Actions CI](https://github.com/rapsealk/gym-navops/workflows/Lint/badge.svg)
![Python 3.8](https://img.shields.io/badge/Python-3.8-blue.svg?logo=python)

```python
import gym
import gym_navops

env = gym.make('NavOps-v0')
env = gym.make('NavOpsDiscrete-v0')
env = gym.make('NavOpsMultiDiscrete-v0')

# For more details..
env = gym.make(
    'NavOps[[Multi]Discrete]-v0',
    worker_id=0,
    base_port=None,
    seed=0,
    no_graphics=False,
    mock=False
)
```
