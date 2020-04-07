from new_rewards import Reward, Reward2
import numpy as np
from memory import Memory

#pip_cost = 1000 or 100000
#type : 1 = discrete action, 2 = continue action
class Env:
    def __init__(self, step_size=96, types=1, spread=10, pip_cost=1000, leverage=500, min_lots=0.01, assets=100000, available_assets_rate=0.8):
        self.step_size = step_size
        spread /= pip_cost

        self.data()
        self.memory = Memory(5000000)

        self.types = types
        reward = Reward if type==1 else Reward2
        self.rewards = reward(spread, leverage, pip_cost, min_lots, assets, available_assets_rate)
        self.rewards.max_los_cut = -np.mean(self.atr) * pip_cost

    def data(self):
        self.x = np.load("x.npy")
        self.y, self.atr, self.scale_atr, self.high, self.low = np.load("target.npy")