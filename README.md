## Demo

```python
python gen_data.py gbpjpy15.csv
```

```python
from rl import dqn
from env import Env

env = Env(step_size=96, types=1, spread=10, pip_cost=1000, leverage=500, min_lots=0.01, assets=100000, available_assets_rate=0.8)

agent = dqn(restore=False, lr=1e-3, n=3, env=env, epsilon=0.05)

agent.run()
```



## rl agent

* sac

* dqn
* qr dqn
* noisy dense
* rainbow (no noisy)
