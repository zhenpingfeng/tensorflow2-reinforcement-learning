import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import standard_ops


class IndependentDense(tf.keras.layers.Dense):
    def build(self, input_shape):
        super(IndependentDense, self).build(input_shape)
        self.last_dim = tensor_shape.dimension_value(input_shape[-1])
        self.mu_init = tf.random_uniform_initializer(-((3 / self.last_dim) ** 0.5), (3 / self.last_dim) ** 0.5)
        self.sigma_init = tf.constant_initializer(0.017)

        self.w_mu = self.add_weight("w_mu", [self.last_dim, self.units], initializer=self.mu_init, dtype=self.dtype,
                                    trainable=True)
        self.w_sigma = self.add_weight("w_sigma", [self.last_dim, self.units], initializer=self.sigma_init,
                                       dtype=self.dtype, trainable=True)

        if self.use_bias:
            self.b_mu = self.add_weight("b_mu", [self.units, ], initializer=self.mu_init, dtype=self.dtype,
                                        trainable=True)
            self.b_sigma = self.add_weight("b_sigma", [self.units, ], initializer=self.sigma_init, dtype=self.dtype,
                                           trainable=True)

    def call(self, inputs):
        def rank(tensor):
            """Return a rank if it is a tensor, else return None."""
            if isinstance(tensor, ops.Tensor):
                return tensor._rank()  # pylint: disable=protected-access
            return None

        inputs = ops.convert_to_tensor(inputs)
        rank = rank(inputs)
        w_epsilon = tf.random.normal((self.last_dim, self.units))
        w = self.w_mu * self.kernel + self.w_sigma * self.kernel * (w_epsilon * self.kernel)

        if rank > 2:
            # Broadcasting is required for the inputs.
            outputs = standard_ops.tensordot(inputs, w, [[rank - 1], [0]])
            # Reshape the output back to the original ndim of the input.
            if not context.executing_eagerly():
                shape = inputs.get_shape().as_list()
                output_shape = shape[:-1] + [self.units]
                outputs.set_shape(output_shape)
        else:
            outputs = gen_math_ops.mat_mul(inputs, w)
        if self.use_bias:
            b_epsilon = tf.random.normal([self.units, ])
            b = self.b_mu + self.bias + self.b_sigma * self.bias * (b_epsilon * self.bias)
            outputs = nn.bias_add(outputs, b)
        if self.activation is not None:
            return self.activation(outputs)  # pylint: disable=not-callable
        return outputs


class FactorisedDense(tf.keras.layers.Dense):
    def build(self, input_shape):
        super(FactorisedDense, self).build(input_shape)
        self.last_dim = tensor_shape.dimension_value(input_shape[-1])
        mu = 1 / self.last_dim ** 0.5
        self.mu_init = tf.random_uniform_initializer(-mu, mu)
        self.sigma_init = tf.constant_initializer(0.5 / self.last_dim ** 0.5)

        self.w_mu = self.add_weight("w_mu", [self.last_dim, self.units], initializer=self.mu_init, dtype=self.dtype,
                                    trainable=True)
        self.w_sigma = self.add_weight("w_sigma", [self.last_dim, self.units], initializer=self.sigma_init,
                                       dtype=self.dtype, trainable=True)

        if self.use_bias:
            self.b_mu = self.add_weight("b_mu", [self.units, ], initializer=self.mu_init, dtype=self.dtype,
                                        trainable=True)
            self.b_sigma = self.add_weight("b_sigma", [self.units, ], initializer=self.sigma_init, dtype=self.dtype,
                                           trainable=True)

    def call(self, inputs):
        def rank(tensor):
            """Return a rank if it is a tensor, else return None."""
            if isinstance(tensor, ops.Tensor):
                return tensor._rank()  # pylint: disable=protected-access
            return None

        inputs = ops.convert_to_tensor(inputs)
        rank = rank(inputs)
        p, q = tf.random.normal((self.last_dim, 1)), tf.random.normal((1, self.last_dim))
        p, q = (tf.math.sign(p) * tf.abs(p) ** 0.5), (tf.math.sign(q) * tf.abs(q) ** 0.5)
        w_epsilon = p * q
        w = self.w_mu * self.kernel + self.w_sigma * self.kernel * (w_epsilon * self.kernel)

        if rank > 2:
            # Broadcasting is required for the inputs.
            outputs = standard_ops.tensordot(inputs, w, [[rank - 1], [0]])
            # Reshape the output back to the original ndim of the input.
            if not context.executing_eagerly():
                shape = inputs.get_shape().as_list()
                output_shape = shape[:-1] + [self.units]
                outputs.set_shape(output_shape)
        else:
            outputs = gen_math_ops.mat_mul(inputs, w)
        if self.use_bias:
            b_epsilon = tf.squeeze(q)
            b = self.b_mu + self.bias + self.b_sigma * self.bias * (b_epsilon * self.bias)
            outputs = nn.bias_add(outputs, b)
        if self.activation is not None:
            return self.activation(outputs)  # pylint: disable=not-callable
        return outputs