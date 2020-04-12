from distutils.dir_util import copy_tree

import tensorflow as tf

import base
import numpy as np
from gym.spaces import Box
from env import Env

env = Env(types=2)

def actor(input_shape):
    i = tf.keras.layers.Input(input_shape)

    x = tf.keras.layers.Dense(128, "relu")(i)
    x = tf.keras.layers.Dense(128, "relu")(x)

    x = tf.keras.layers.Dense(2, name="policy")(x)

    return tf.keras.Model(i, x)

def q(x, name):
    x = tf.keras.layers.Dense(128, "elu")(x)
    x = tf.keras.layers.Dense(128, "elu")(x)

    return tf.keras.layers.Dense(1, name=name)(x)

def critic(input_shape):
    s = tf.keras.layers.Input(input_shape, name="s")
    a = tf.keras.layers.Input((2,), name="a")

    x = base.bese_net(s)
    f = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Concatenate()([f, a])
    q1 = q(x, "q1")
    q2 = q(x, "q2")

    return tf.keras.Model([s, a], [q1, q2])

class Model(tf.keras.Model):
    def __init__(self, dim=(130, 4)):
        super(Model, self).__init__()
        self.actor = actor(dim)
        self.critic = critic(dim)


class Agent(base.Agent):
    def __init__(self, restore=False, lr=1e-3, n=1, env=env):
        super(Agent, self).__init__(
            restore=restore,
            lr=lr,
            env=env,
            n=n
        )

    def build(self):
        self.aciton_space = Box(-1, 1, (2,))
        self.model = Model()
        self.target_model = Model()

        if self.restore:
            self.i = np.load("td3/td3_epoch.npy")
            self.model.load_weights("td3/td3")
            self.target_model.load_weights("td3/td3")
        else:
            self.target_model.set_weights(self.model.get_weights())

        self.v_opt = tf.keras.optimizers.Nadam(1e-3)
        self.p_opt = tf.keras.optimizers.Nadam(self.lr)

        l = self.model.actor.get_layer
        self.policy = tf.keras.backend.function(l("inputs").input, l("policy").output)
        l = self.target_model.actor.get_layer
        self.target_policy = tf.keras.backend.function(l("inputs").input, l("policy").output)
        l = self.model.critic.get_layer
        self.q = tf.keras.backend.function([l("s").input, l("a").input],[l("q1").output, l("q2").output])
        l = self.target_model.critic.get_layer
        self.target_q = tf.keras.backend.function([l("s").input, l("a").input],[l("q1").output, l("q2").output])

        self.epoch = self.i if self.restore else 0

    def train(self, i):
        tree_idx, replay = self.memory.sample(128)

        states = np.array([a[0][0] for a in replay], np.float32)
        new_states = np.array([a[0][3] for a in replay], np.float32)
        actions = np.array([a[0][1] for a in replay], np.float32).reshape((-1, 2))
        rewards = np.array([a[0][2] for a in replay], np.float32).reshape((-1, 1))

        noise_policy = self.target_policy(new_states) + np.random.normal(0, 0.2, actions.shape).clip(-0.5, 0.5)
        q_backup = rewards + self.gamma * np.minimum(self.target_q([new_states, noise_policy]))

        with tf.GradientTape() as tape:
            q1, q2 = self.model.critic([states, actions])
            q1_error = q_backup - q1
            q2_erorr = q_backup - q2
            loss = tf.reduce_mean(q1_error ** 2) + tf.reduce_mean(q2_erorr ** 2)
        gradient = tape.gradient(loss, self.model.critic.trainable_variables)
        self.v_opt.apply_gradients(zip(gradient, self.model.critic.trainable_variables))

        if self.epoch % 2 == 0:
            with tf.GradientTape() as tape:
                policy = self.model.actor(states)
                q1, _ = self.model.critic([states, policy])
                loss = -tf.reduce_mean(q1)
            gradient = tape.gradient(loss, self.model.actor.trainable_variables)
            self.p_opt.apply_gradients(zip(gradient, self.model.actor.trainable_variables))

            self.target_model.set_weights(0.005 * np.array(self.model.get_weights()) + 0.995 * np.array(self.target_model.get_weights()))

        abs_error = np.array(np.abs(q1_error) + np.abs(q2_erorr))
        # print(print(abs_error))
        self.memory.batch_update(tree_idx, abs_error)
        self.gamma = 1 - (0.1 + (1 - 0.1) * (np.exp(-0.00001 * i)))

        self.epoch += 1

    def action(self, state, i):
        if i < 50:
            policy = np.array([self.aciton_space.sample() for _ in range(state.shape[0])])
        else:
            policy = self.policy(state)

        q = policy[:]
        '''
        lev = 400, max_lev=500, min_lev=200の場合
        400 + 400 * 0.25 = 500
        400 + 400 * -0.5 = 200
        '''
        action, leverage = policy[:, 0], [i * 0.25 if i > 0 else i * 0.5 for i in policy[:, 1]]
        action = [2 if i >= -1.5 and i < -0.5 else 0 if i >= -0.5 and i < 0.5 else 1 for i in action * 1.5]

        return action, leverage, q

    def save(self, i):
        self.restore = True
        self.i = i
        self.model.save_weights("td3/td3")
        np.save("td3/td3_epoch", i)
        try:
            copy_tree("/content/td3", "/content/drive/My Drive/td3")
        except:
            pass
