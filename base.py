import numpy as np
import pandas as pd
import tensorflow as tf
from IPython.display import clear_output

from env import Env


# huber loss

def se_block(x, r=16):
    sq = tf.keras.layers.GlobalAveragePooling1D()(x)
    sq = tf.keras.layers.Reshape((1, x.shape[-1]))(sq)

    ex1 = tf.keras.layers.Dense(x.shape[-1] // r, "elu")(sq)
    ex2 = tf.keras.layers.Dense(x.shape[-1], "sigmoid")(ex1)

    return tf.keras.layers.Multiply()([x, ex2])


def bese_net(i):
    x1 = tf.keras.layers.Conv1D(48, 3, 1, "same", dilation_rate=1, activation="elu")(i)
    x2 = tf.keras.layers.Conv1D(48, 3, 1, "same", dilation_rate=3, activation="elu")(i)
    x3 = tf.keras.layers.Conv1D(48, 3, 1, "same", dilation_rate=5, activation="elu")(i)
    x4 = tf.keras.layers.Conv1D(48, 3, 1, "same", dilation_rate=7, activation="elu")(i)
    x = tf.keras.layers.Add()([x1, x2, x3, x4])

    x5 = tf.keras.layers.Conv1D(48, 1, 1, "same", activation="elu")(i)
    x = tf.keras.layers.Concatenate()([x, x5])
    x = se_block(x)

    # x1 = tf.keras.layers.Conv1D(48, 3, 1, "same", dilation_rate=1, activation="elu")(x)
    # x2 = tf.keras.layers.Conv1D(48, 3, 1, "same", dilation_rate=3, activation="elu")(x)
    # x3 = tf.keras.layers.Conv1D(48, 3, 1, "same", dilation_rate=5, activation="elu")(x)
    # x4 = tf.keras.layers.Conv1D(48, 3, 1, "same", dilation_rate=7, activation="elu")(x)
    # x = tf.keras.layers.Add()([x1,x2,x3,x4])
    #
    # x5 = tf.keras.layers.Conv1D(48,1,1,"same", activation="elu")(x)
    # x = tf.keras.layers.Concatenate()([x,x5])
    # x = se_block(x)

    return x


# env = Env()


class Agent:
    def __init__(self, restore=False, lr=1e-3, n=1, env=None):
        self.restore = restore
        self.lr = lr
        self.n = n

        self.env = env
        self.step_size = self.env.step_size
        self.types = self.env.types
        self.x = self.env.x
        self.y, self.atr, self.scale_atr, self.high, self.low = self.env.y, self.env.atr, self.env.scale_atr, self.env.high, self.env.low
        self.rewards = self.env.rewards
        self.memory = self.env.memory

        self.gamma = 0

        self.build()

    def build(self):
        pass

    def train(self, i):
        pass

    def action(self, state, i):
        pass

    def save(self, i):
        pass

    def run(self, train=True):
        i = 10000000 if train else 5
        start = 0 if not self.restore else self.i
        start = start if train else 4
        h = np.random.randint(self.x.shape[0] - self.x.shape[0] * 0.2 - self.step_size)
        train_step = self.x.shape[0] - self.x.shape[0] * 0.2 - self.step_size
        test_step = self.x.shape[0] - self.x.shape[0] * 0.2, self.x.shape[0] - self.step_size * 5
        reset = 0

        for i in range(start, i):
            if (i + 1) % 5 == 0 and train:
                h = np.random.randint(test_step[0], test_step[1])
            elif i % 5 != 0 or not train:
                h = np.random.randint(train_step)

            df = self.x[h:h + self.step_size]
            trend = self.y[h:h + self.step_size]
            atr = self.atr[h:h + self.step_size]
            scale_atr = self.scale_atr[h:h + self.step_size]
            high = self.high[h:h + self.step_size]
            low = self.low[h:h + self.step_size]


            if self.types == 1:
                action = self.action(df, i)
                self.rewards.reward(trend, high, low, action, atr, scale_atr)
                q = np.array(action).reshape((-1, 1))

            elif self.types == 2:
                action, leverage, q = self.action(df, i)
                self.rewards.reward(trend, high, low, action, leverage, atr, scale_atr)

            # # memory append
            if (i + 1) % 5 != 0:
                rewards = np.array(self.rewards.total_gain)
                r1, r2 = rewards[1:], rewards[:-1]
                rewards = [0]
                rewards = np.append(rewards, np.log(r1 / r2) * 100)
                rewards = (rewards * 10 ** 3).astype(np.int32) * (10 ** -2)

                assert len(rewards) == df.shape[0]
                l = np.array(rewards != 0).reshape((-1,))
                rewards = rewards[l]
                df = df[l]
                q = q[l]
                #
                if len(rewards) > self.n:
                    memory = []
                    for t in range(len(rewards) - self.n):
                        if 0.2 > np.random.rand():
                            r = sum(rewards[t:t + self.n] * 0.99)
                            e = df[t], q[t], r, df[t + self.n]
                            memory.append(e)

                    # abs_error = self.sample(memory).reshape((-1,))
                    for t in range(len(memory)):
                        self.memory.store(memory[t])
                        # self.memory.store(memory[t], abs_error[t])

                if reset > 50:
                    self.train(i)
                self.gamma = 1 - (0.1 + (1 - 0.1) * (np.exp(-0.00001 * i)))
                reset += 1

            if i % 2000 == 0:
                clear_output()

            if (i + 1) % 5 == 0 or not train:
                prob = pd.Series(action)
                prob = prob.value_counts() / prob.value_counts().sum()
                if len(prob) != 3:
                    for p in range(3):
                        if p not in prob:
                            prob[p] = 0

                print('action probability: buy={}, sell={}, hold={}'.format(
                    prob[0], prob[1], prob[2]))
                print('epoch: {}, total assets:{}, growth_rate:{}'.format(
                    i + 1, self.rewards.assets, self.rewards.assets / self.rewards.initial_assets))
                print("")

            if (i + 1) % 500 == 0:
                self.save(i)