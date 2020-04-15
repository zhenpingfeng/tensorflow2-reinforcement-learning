import shutil

import tensorflow as tf

import base
import qr_dqn
import numpy as np
from env import Env
from noisy_dense import IndependentDense


def output(x, num):
    out = [IndependentDense(num)(x) for _ in range(3)]
    out = [tf.reshape(out, (-1, 1, num)) for out in out]

    out = tf.keras.layers.Concatenate(axis=1, name="Q")(out)
    return out


def build_model(n=200, dim=(130, 4)):
    inputs = tf.keras.layers.Input(dim, name="inputs")

    x = base.bese_net(inputs)
    x = tf.keras.layers.Flatten()(x)
    x = IndependentDense(512, "elu")(x)

    advantage = output(x, n)
    value = IndependentDense(n)(x)
    value = tf.reshape(value, (-1,1,n))
    sub = advantage - tf.reshape(tf.reduce_mean(advantage, 1), (-1,1,n))
    out = tf.keras.layers.Add(name="q")([value, sub])

    return tf.keras.Model(inputs, out)


class Agent(qr_dqn.Agent):
    def build(self):
        if self.restore:
            self.i = np.load("rainbow_epoch.npy")
            self.model = tf.keras.models.load_model("rainbow.h5")
            self.target_model = tf.keras.models.load_model("rainbow.h5")
        else:
            self.model = build_model(self.num)
            opt = tf.keras.optimizers.Nadam(self.lr)
            self.model.compile(opt, "mse")
            self.target_model = build_model(self.num)
            self.target_model.set_weights(self.model.get_weights())

        self.q = tf.keras.backend.function(self.model.get_layer("inputs").input, self.model.get_layer("q").output)
        self.targe_q = tf.keras.backend.function(self.target_model.get_layer("inputs").input, self.target_model.get_layer("q").output)


    # def action(self, state, i):
    #     q = np.sum(self.q(state), -1)
    #     return np.argmax(q, -1)

    def save(self, i):
        self.restore = True
        self.i = i
        self.model.save("rainbow.h5")
        np.save("rainbow_epoch", i)
        _ = shutil.copy("/content/rainbow.h5", "/content/drive/My Drive")
        _ = shutil.copy("/content/rainbow_epoch.npy", "/content/drive/My Drive")
