from collections import deque

import tensorflow as tf
from IPython.display import clear_output

from memory import Memory
from new_rewards import Reward, Reward2
import numpy as np
import random

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
    x = tf.keras.layers.Add()([x1,x2,x3,x4])

    x5 = tf.keras.layers.Conv1D(48,1,1,"same", activation="elu")(i)
    x = tf.keras.layers.Concatenate()([x,x5])
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


class Base_Agent:
    def __init__(self, spread, pip_cost, leverage=500, min_lots=0.01, assets=1000000, available_assets_rate=0.4,
                 restore=False, step_size=96, n=3, lr=1-3):
        self.step_size = step_size
        spread /= pip_cost
        self.restore = restore
        self.lr = lr
        self.n = n
        self.gamma = 0.4
        self.random = 0
        self.scale = 3

        self.gen_data()
        self.build()

        self.rewards = Reward(spread, leverage, pip_cost, min_lots, assets, available_assets_rate) if self.types == "DQN" else Reward2(spread, leverage, pip_cost, min_lots, assets, available_assets_rate)
        self.rewards.max_los_cut = -np.mean(self.atr) * pip_cost
        self.memory = Memory(5000000)

    def build(self):
        pass

    def gen_data(self):
        self.x = np.load("x.npy")
        self.y, self.atr, self.scale_atr, self.high, self.low = np.load("target.npy")

    def mse(self, q_backup, q):
        return tf.abs(q_backup - q) ** 2

    def huber_loss(self, q_backup, q, delta=4):
        error = tf.abs(q_backup - q)
        loss = tf.where(error < delta, error ** 2 * .5, delta * error - 0.5 * delta ** 2)
        # return tf.where(q_backup > 0, loss, loss*0)
        return loss

    def loss(self, states, new_states, rewards, actions):
        pass

    def sample(self, memory):
        pass

    def train(self, states=None):
        pass

    def lr_decay(self, i):
        pass

    def gamma_updae(self, i):
        pass

    def update_w(self, i):
        pass

    def nstep(self, r):
        discount_r = 0.0
        for r in r:
            discount_r += 0.99 * r
        return r

    def prob(self, history):
        prob = np.asanyarray(history)
        a = np.mean(prob == 0)
        b = np.mean(prob == 1)
        c = 1 - (a + b)
        prob = [a, b, c]
        return prob

    def policy(self, state, i):
        pass

    def pg_action(self, action):
        q = action[:]
        action, leverage = action[:, 0], [i * 2.5 if i > 0 else i * .5 for i in action[:, 1]]
        action = [0 if i > 0.5 else 1 for i in action]
        return action, leverage, q

    def save(self, i):
        pass

    def run(self, train=True):
        i = 10000000 if train else 5
        start = 0 if not self.restore else self.i
        start = start if train else 4
        h = np.random.randint(self.x.shape[0] - self.x.shape[0] * 0.2 - self.step_size)
        train_step = self.x.shape[0] - self.x.shape[0] * 0.2 - self.step_size
        test_step = self.x.shape[0] - self.x.shape[0] * 0.2, self.x.shape[0] - self.step_size * 5

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

            action = self.policy(df, i)
            if self.types == "PG":
                action, leverage, q = self.pg_action(action)
                self.rewards.reward(trend, high, low, action, leverage, atr, scale_atr)
            elif self.types == "DQN":
                self.rewards.reward(trend, high, low, action, atr, scale_atr)
                q = np.array(action).reshape((-1,1))

            # # memory append
            if (i + 1) % 5 != 0:
                if True:
                    rewards = np.array(self.rewards.total_gain)
                    r1, r2 = rewards[1:], rewards[:-1]
                    rewards = [0]
                    rewards = np.append(rewards, np.log(r1 / r2) * 100)
                    rewards = (rewards * 10 ** self.scale).astype(np.int32) * (10 ** -2)

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
                                e = df[t], q[t], self.nstep(rewards[t:t + self.n]), df[t + self.n]
                                memory.append(e)

                        abs_error = self.sample(memory).reshape((-1,))
                        for t in range(len(abs_error)):
                            self.memory.store(memory[t], abs_error[t])
                            #
            # #
                if i > 50:
                    self.train()
                self.lr_decay(i)
                self.gamma_updae(i)
                self.update_w(i)

            if i % 2000 == 0:
                clear_output()

            if (i + 1) % 5 == 0 or not train:
                prob = self.prob(action)

                if self.rewards.assets < 0:
                    break

                print('action probability: buy={}, sell={}, hold={}'.format(
                    prob[0], prob[1], prob[2]))
                print('epoch: {}, total assets:{}, growth_rate:{}'.format(
                    i + 1, self.rewards.assets, self.rewards.assets / self.rewards.initial_assets))
                print("")

            if (i + 1) % 500 == 0:
                self.save(i)