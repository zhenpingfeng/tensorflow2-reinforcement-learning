from distutils.dir_util import copy_tree

import tensorflow as tf

from new_rewards import *
import base
from gym.spaces import Box

EPS = 1e-10


def gaussian_entropy(log_std):
    return tf.reduce_sum(log_std + 0.5 * np.log(2.0 * np.pi * np.e), axis=-1, name="entropy", keepdims=True)


def gaussian_likelihood(input_, mu_, log_std):
    pre_sum = -0.5 * (((input_ - mu_) / (tf.exp(log_std) + EPS))
                      ** 2 + 2 * log_std + np.log(2 * np.pi))
    return tf.reduce_sum(pre_sum, axis=1, keepdims=True)


def clip_but_pass_gradient(input_, lower=-1., upper=1.):
    clip_up = tf.cast(input_ > upper, tf.float32)
    clip_low = tf.cast(input_ < lower, tf.float32)
    return input_ + tf.stop_gradient((upper - input_) * clip_up + (lower - input_) * clip_low)


def apply_squashing_func(mu_, pi_, logp_pi):
    # Squash the output
    deterministic_policy = tf.keras.layers.Activation("tanh", name="deterministic_policy")(mu_)
    policy = tf.keras.layers.Activation("tanh", name="policy")(pi_)
    # OpenAI Variation:
    # To avoid evil machine precision error, strictly clip 1-pi**2 to [0,1] range.
    logp_pi -= tf.reduce_sum(tf.math.log(clip_but_pass_gradient(1 - policy ** 2, lower=0, upper=1) + EPS), axis=1,
                             keepdims=True)
    # Squash correction (from original implementation)
    # logp_pi -= tf.reduce_sum(tf.log(1 - policy ** 2 + EPS),
    #  axis=1, name="logp_pi")
    return deterministic_policy, policy, logp_pi


def output(x, name):
    x = tf.keras.layers.Dense(1280, "elu")(x)
    return tf.keras.layers.Dense(1, name=name)(x)


def build_actor(dim):
    inputs = tf.keras.layers.Input(dim)

    x = tf.keras.layers.GlobalAveragePooling1D()(inputs)
    x = tf.keras.layers.Dense(32, "elu")(x)
    x = tf.keras.layers.Dropout(.3)(x)
    # x = tf.keras.layers.AlphaDropout(0.1)(x)
    x = tf.keras.layers.Dense(32, "elu")(x)
    x = tf.keras.layers.Dropout(.3)(x)
    # x = tf.keras.layers.AlphaDropout(0.1)(x)

    log_std = tf.keras.layers.Dense(2)(x)
    mu = tf.keras.layers.Dense(2)(x)

    # mu = tf.clip_by_value(tf.abs(mu), 0.1, 10) * (tf.abs(mu) / mu)
    log_std = tf.clip_by_value(log_std, -20, 2.)
    std = tf.exp(log_std)

    pi = mu + tf.random.normal(tf.shape(mu)) * std
    logp_pi = gaussian_likelihood(pi, mu, log_std)
    entropy = gaussian_entropy(log_std)
    deterministic_policy, policy, logp_pi = apply_squashing_func(mu, pi, logp_pi)

    return tf.keras.Model(inputs, [deterministic_policy, policy, logp_pi])


def build_critic(dim):
    states = tf.keras.layers.Input(dim, name="states")
    action = tf.keras.layers.Input((2,), name="action")

    x = base.bese_net(states)

    x_action = tf.keras.layers.Concatenate()([x, action])
    q1 = output(x_action, "q1")
    q2 = output(x_action, "q2")
    v = output(x, "v")

    return tf.keras.Model([states, action], [q1, q2, v])


class Model(tf.keras.Model):
    def __init__(self, dim=(10, 4)):
        super(Model, self).__init__()
        self.actor = build_actor(dim)
        self.critic = build_critic(dim)
        self.log_ent_coef = tf.Variable(tf.math.log(1.0), dtype=tf.float32, name="log_ent_coef")
        self.target_entropy = -np.prod(3).astype(np.float32)


class Agent(base.Base_Agent):
    def build(self):
        self.gamma = 0.
        self.epock = 0.
        self.types = "PG"
        self.aciton_space = Box(np.array([-1.,-1.]), np.array([1., 1.]))

        self.model = Model()
        self.target_model = Model()

        if self.restore:
            self.i = np.load("sac_epoch.npy")
            self.model.load_weights("sac")
            self.target_model.load_weights("sac")
        else:
            self.target_model.set_weights(self.model.get_weights())

        self.v_opt = tf.keras.optimizers.Nadam(1e-3)
        self.p_opt = tf.keras.optimizers.Nadam(self.lr)
        self.e_opt = tf.keras.optimizers.Nadam(self.lr)

    def sample(self, memory):
        states = np.array([a[0] for a in memory], np.float32)
        new_states = np.array([a[3] for a in memory], np.float32)
        actions = np.array([a[1] for a in memory]).reshape((-1, 2))
        rewards = np.array([a[2] for a in memory], np.float32).reshape((-1, 1))

        _, _, target_v = self.target_model.critic.predict_on_batch([new_states, actions])
        q1, _, _ = self.model.critic.predict_on_batch([states, actions])

        q_backup = rewards + self.gamma * target_v.numpy()

        return np.abs(q_backup - q1)

    def train(self):
        tree_idx, replay = self.memory.sample(128)

        states = np.array([a[0][0] for a in replay], np.float32)
        new_states = np.array([a[0][3] for a in replay], np.float32)
        actions = np.array([a[0][1] for a in replay]).reshape((-1, 2))
        rewards = np.array([a[0][2] for a in replay], np.float32).reshape((-1, 1))

        ent_coef = tf.exp(self.model.log_ent_coef)
        ################################################################################
        with tf.GradientTape() as p_tape:
            d, policy, logp_pi = self.model.actor.predict_on_batch(states)
            q1_pi, q2_pi, _ = self.model.critic.predict_on_batch([states, policy])
            # min_q_pi = tf.minimum(q1_pi, q2_pi)
            # q1, q2, v = self.model.critic.predict_on_batch([states, actions])
            p_loss = tf.reduce_mean(ent_coef * logp_pi - q1_pi)
        ################################################################################
        with tf.GradientTape() as v_tape:
            _, _, target_v = self.target_model.critic.predict_on_batch([new_states, actions])
            q1, q2, v = self.model.critic.predict_on_batch([states, actions])
            min_q_pi = tf.minimum(q1_pi, q2_pi)

            q_backup = rewards + self.gamma * target_v

            q1_loss = tf.reduce_mean((q_backup - q1) ** 2) * .5
            q2_loss = tf.reduce_mean((q_backup - q2) ** 2) * .5

            v_backup = min_q_pi - ent_coef * logp_pi
            v_loss = tf.reduce_mean((v_backup - v) ** 2) * .5

            v_loss += q1_loss + q2_loss
        ################################################################################
        with tf.GradientTape() as e_tape:
            e_loss = -tf.reduce_mean(self.model.log_ent_coef * logp_pi + self.model.target_entropy)
        ################################################################################

        ae = np.abs(np.array(q_backup) - q1.numpy()).reshape((-1,))
        # print(ae)
        self.memory.batch_update(tree_idx, ae)

        gradients = v_tape.gradient(v_loss, self.model.critic.trainable_variables)
        gradients = [(tf.clip_by_value(grad, -10.0, 10.0))
                     for grad in gradients]
        self.v_opt.apply_gradients(zip(gradients, self.model.critic.trainable_variables))

        if self.epock >= 50 and self.epock % 5 == 0:
            gradients = p_tape.gradient(p_loss, self.model.actor.trainable_variables)
            gradients = [(tf.clip_by_value(grad, -1., 1.))
                         for grad in gradients]
            self.p_opt.apply_gradients(zip(gradients, self.model.actor.trainable_variables))

            self.target_model.set_weights(
                (1 - 0.005) * np.array(self.target_model.get_weights()) + 0.005 * np.array(
                    self.model.get_weights()))

        gradients = e_tape.gradient(e_loss, self.model.log_ent_coef)
        gradients = (tf.clip_by_value(gradients, -1.0, 1.0))
        self.e_opt.apply_gradients([[gradients, self.model.log_ent_coef]])

        self.epock += 1

    def lr_decay(self, i):
        lr = self.lr * 0.0001 ** (i / 10000000)
        self.e_opt.lr.assign(lr)
        self.p_opt.lr.assign(lr)
        lr = 1e-3 * 0.0001 ** (i / 10000000)
        self.v_opt.lr.assign(lr)

    def gamma_updae(self, i):
        self.gamma = 1 - (0.8 + (1 - 0.8) * (np.exp(-0.00001 * i)))

    def policy(self, state, i):
        if i > 100:
            deterministic_policy, policy, _ = self.model.actor.predict_on_batch(state)
            if (i + 1) % 5 != 0:
                # p = policy
                # deterministic_policy = deterministic_policy.numpy()
                # policy += 0.1 * np.random.randn(deterministic_policy.shape[0], deterministic_policy.shape[1])
                p = np.array([policy[i] if 0.1 < np.random.rand() else self.aciton_space.sample() for i in range(policy.shape[0])])
            else:
                p = deterministic_policy
        else:
            p = np.array([self.aciton_space.sample() for _ in range(state.shape[0])])

        return p

    def save(self, i):
        self.restore = True
        self.i = i
        self.model.save_weights("sac/sac")
        np.save("sac/sac_epoch", i)
        copy_tree("/content/sac", "/content/drive/My Drive")
