import shutil

import tensorflow as tf

import base
import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def build_model(dim=(30, 4)):
    inputs = tf.keras.layers.Input(dim)

    x = base.bese_net(inputs)
    x = tf.keras.layers.Flatten()(x)
    # x = tf.keras.layers.GRU(128)(x)

    # tensor_action, tensor_validation = tf.split(x, 2, 1)
    # x = tf.keras.layers.Dense(512, "elu")(x)
    out = tf.keras.layers.Dense(2, name="out")(x)
    # v = tf.keras.layers.Dense(328, "elu", kernel_initializer="he_normal")(tensor_validation)
    # # v = tf.keras.layers.BatchNormalization()(v)
    # v = tf.keras.layers.Dense(1, name="v")(v)
    # a = tf.keras.layers.Dense(328, "elu", kernel_initializer="he_normal")(tensor_action)
    # # a = tf.keras.layers.BatchNormalization()(a)
    # a = tf.keras.layers.Dense(2, name="a")(a)
    # out = v + tf.subtract(a, tf.reduce_mean(a, axis=1, keepdims=True))

    return tf.keras.Model(inputs, out)

################################################################################################################################

class Agent(base.Base_Agent):
    def build(self):
        self.types = "DQN"
        self.gamma = 0.3
        self.epsilon = 0.05
        self.scale = 3

        if self.restore:
            self.i = np.load("dqn_epoch.npy")
            self.model = tf.keras.models.load_model("dqn.h5")
            self.target_model = tf.keras.models.load_model("dqn.h5")
        else:
            self.model = build_model()
            self.model.compile("nadam", "mse")
            self.target_model = build_model()
            self.target_model.set_weights(self.model.get_weights())

    def loss(self, states, new_states, rewards, actions):
        q = self.model(states)
        target_q = self.target_model(new_states).numpy()
        arg_q = self.model(new_states).numpy()
        arg_q = np.argmax(arg_q, -1)

        q_backup = q.numpy()

        for i in range(rewards.shape[0]):
            q_backup[i, actions[i]] = rewards[i] + self.gamma * target_q[i, arg_q[i]]

        return q_backup, q

    def sample(self, memory):
        states = np.array([a[0] for a in memory], np.float32)
        new_states = np.array([a[3] for a in memory], np.float32)
        actions = np.array([a[1] for a in memory]).reshape((-1, 1))
        rewards = np.array([a[2] for a in memory], np.float32).reshape((-1, 1))

        q_backup, q = self.loss(states, new_states, rewards, actions)

        return tf.reduce_sum(self.mse(q_backup, q), -1).numpy().reshape((-1,))

    def train(self):
        tree_idx, replay = self.memory.sample(128)

        states = np.array([a[0][0] for a in replay], np.float32)
        new_states = np.array([a[0][3] for a in replay], np.float32)
        actions = np.array([a[0][1] for a in replay]).reshape((-1, 1))
        rewards = np.array([a[0][2] for a in replay], np.float32).reshape((-1, 1))

        with tf.GradientTape() as tape:
            q_backup, q = self.loss(states, new_states, rewards, actions)
            error = self.mse(q_backup, q)
            loss = tf.reduce_mean(error)

        ae = tf.reduce_sum(error, -1).numpy().reshape((-1,)) + 1e-10

        self.memory.batch_update(tree_idx, ae)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        # gradients = [(tf.clip_by_value(grad, -100.0, 100.0))
        #              for grad in gradients]
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    def lr_decay(self, i):
        lr = self.lr * 0.0001 ** (i / 10000000)
        self.model.optimizer.lr.assign(lr)

    # def gamma_updae(self, i):
    #     self.gamma = 1 - (0.6 + (1 - 0.6) * (np.exp(-0.0001 * i)))

    def policy(self, state, i):
        epsilon = self.epsilon + (1 - self.epsilon) * (np.exp(-0.0001 * i))
        q = self.model(state).numpy()

        if (i + 1) % 5 != 0:
            q = np.abs(q) / np.sum(np.abs(q), 1).reshape((-1, 1)) * (np.abs(q) / q)
            epsilon = epsilon if self.random % 5 != 0 else 1.
            q += epsilon * np.random.randn(q.shape[0], q.shape[1])
            # action = np.argmax(q, 1)
            action = [np.argmax(i) if 0.1 < np.random.rand() else np.random.randint(2) for i in q]
            self.random += 1
        else:
            action = np.argmax(q, -1)

        return action

    def save(self, i):
        self.restore = True
        self.i = i
        self.model.save("dqn.h5")
        np.save("dqn_epoch", i)
        _ = shutil.copy("/content/dqn.h5", "/content/drive/My Drive")
        _ = shutil.copy("/content/dqn_epoch.npy", "/content/drive/My Drive")
