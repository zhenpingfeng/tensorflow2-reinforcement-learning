import shutil

import numpy as np
import tensorflow as tf

import base
from env import Env


def se_block(x):
    sq = tf.keras.layers.GlobalAveragePooling1D()(x)
    sq = tf.keras.layers.Reshape((1, x.shape[-1]))(sq)

    ex1 = tf.keras.layers.Dense(x.shape[-1] // 7, "elu")(sq)
    ex2 = tf.keras.layers.Dense(x.shape[-1], "sigmoid")(ex1)

    return tf.keras.layers.Multiply()([x, ex2])


def output(x, num):
    x = tf.keras.layers.Dense(512, "elu")(x)
    out = [tf.keras.layers.Dense(num)(x) for _ in range(3)]
    out = [tf.reshape(out, (-1, 1, num)) for out in out]

    out = tf.keras.layers.Concatenate(axis=1, name="q")(out)
    return out

    # return tf.reshape(out, (-1, 3, num))


def build_model(n=200, dim=(130, 4)):
    inputs = tf.keras.layers.Input(dim, name="inputs")

    x = base.bese_net(inputs)
    x = tf.keras.layers.Flatten()(x)

    out = output(x, n)

    return tf.keras.Model(inputs, out)


env = Env(types=1)


class Agent(base.Agent):
    def __init__(self, restore=False, lr=1e-3, n=1, env=env, num=200, epsilon=0.05):
        self.num = num
        self.epsilon = epsilon
        self.tau = np.array([i / self.num for i in range(self.num)])
        self.random = 0

        super(Agent, self).__init__(
            restore=restore,
            lr=lr,
            env=env,
            n=n
        )

    def build(self):
        if self.restore:
            self.i = np.load("qrdqn_epoch.npy")
            self.model = tf.keras.models.load_model("qrdqn.h5")
            self.target_model = tf.keras.models.load_model("qrdqn.h5")
        else:
            self.model = build_model(self.num)
            opt = tf.keras.optimizers.Nadam(self.lr)
            self.model.compile(opt, "mse")
            self.target_model = build_model(self.num)
            self.target_model.set_weights(self.model.get_weights())

        self.q = tf.keras.backend.function(self.model.get_layer("inputs").input, self.model.get_layer("q").output)
        self.targe_q = tf.keras.backend.function(self.target_model.get_layer("inputs").input, self.target_model.get_layer("q").output)


    def sample(self, memory):
        states = np.array([a[0] for a in memory], np.float32)
        new_states = np.array([a[3] for a in memory], np.float32)
        actions = np.array([a[1] for a in memory]).reshape((-1, 1))
        rewards = np.array([a[2] for a in memory], np.float32).reshape((-1, 1))

        q = self.q(states)
        target_q = self.targe_q(new_states)
        arg_q = np.sum(self.model(new_states), -1).reshape((-1, 3))
        arg_q = np.argmax(arg_q, -1)

        q_backup = q.numpy()

        for i in range(q.shape[0]):
            q_backup[i, actions[i]] = rewards[i] + self.gamma * target_q[i, arg_q[i]]

        error = q_backup - q
        q_error = tf.maximum(self.tau * error, (self.tau - 1) * error)
        loss = tf.where(q_error < 2, q_error ** 2 * .5, 2 * q_error - 0.5 * 2 ** 2)

        return tf.reduce_mean(tf.reduce_sum(loss, 2), 1).numpy()


    def train(self, i):
        tree_idx, replay = self.memory.sample(256)

        states = np.array([a[0][0] for a in replay], np.float32)
        new_states = np.array([a[0][3] for a in replay], np.float32)
        actions = np.array([a[0][1] for a in replay]).reshape((-1, 1))
        rewards = np.array([a[0][2] for a in replay], np.float32).reshape((-1, 1))


        target_q = self.targe_q(new_states)
        arg_q = np.sum(self.q(new_states), -1).reshape((-1, 3))
        arg_q = np.argmax(arg_q, -1)

        with tf.GradientTape() as tape:
            q = self.model(states)
            q_backup = q.numpy()
            for i in range(arg_q.shape[0]):
                q_backup[i, actions[i]] = rewards[i] + self.gamma * target_q[i, arg_q[i]]

            error = q_backup - q
            q_error = tf.maximum(self.tau * error, (self.tau - 1) * error)
            loss = tf.where(q_error < 2, q_error ** 2 * .5, 2 * q_error - 0.5 * 2 ** 2)

            error = tf.reduce_mean(tf.reduce_sum(loss, 2), 1)
            loss = tf.reduce_mean(error)

        ae = error.numpy().reshape((-1,)) + 1e-10
        self.memory.batch_update(tree_idx, ae)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        if (i + 1) % 200 == 0:
            self.target_model.set_weights(self.model.get_weights())

        # lr = self.lr * 0.0001 ** (i / 10000000)
        # self.model.optimizer.lr.assign(lr)

    def action(self, state, i):
        epsilon = self.epsilon + (1 - self.epsilon) * (np.exp(-0.0001 * i))
        q = np.sum(self.q(state), -1)
        q = np.abs(q) / np.sum(np.abs(q), 1).reshape((-1, 1)) * (np.abs(q) / q)

        if (i + 1) % 5 != 0:
            epsilon = epsilon if self.random % 5 != 0 else 1.
            q += epsilon * np.random.randn(q.shape[0], q.shape[1])
            action = [np.argmax(i) if 0.1 < np.random.rand() else np.random.randint(3) for i in q]
            self.random += 1
        else:
            action = np.argmax(q, -1)

        return action

    def save(self, i):
        self.restore = True
        self.i = i
        self.model.save("qrdqn.h5")
        np.save("qrdqn_epoch", i)
        _ = shutil.copy("/content/qrdqn.h5", "/content/drive/My Drive")
        _ = shutil.copy("/content/qrdqn_epoch.npy", "/content/drive/My Drive")
