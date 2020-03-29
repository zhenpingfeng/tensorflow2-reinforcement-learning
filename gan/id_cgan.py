import tensorflow as tf
import numpy as np

def se_block(x, r=16):
    sq = tf.keras.layers.GlobalAveragePooling1D()(x)
    sq = tf.keras.layers.Reshape((1, x.shape[-1]))(sq)

    ex1 = tf.keras.layers.Dense(x.shape[-1] // r, "elu")(sq)
    ex2 = tf.keras.layers.Dense(x.shape[-1], "sigmoid")(ex1)

    return tf.keras.layers.Multiply()([x, ex2])

def dense_f(x, f, k, s):
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(f, k, s, "same", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.ELU()(x)
    return x


def dense_b(x, f, k, s, n):
    concatenated_inputs = x
    for i in range(n):
        x = dense_f(concatenated_inputs, f, k, s)
        concatenated_inputs = tf.keras.layers.Concatenate()(
            [concatenated_inputs, x])
    return x

def generator():
    inputs = tf.keras.layers.Input((30,4))

    # x = tf.keras.layers.Conv1D(128, 3, 1, "same", activation="elu")(inputs)
    # x = tf.keras.layers.Conv1D(128, 3, 1, "same", activation="elu")(x)
    # x = tf.keras.layers.Conv1D(128 * 9 * 9, 3, 1, "same", activation="elu")(x)
    # x = se_block(x)

    x = tf.keras.layers.Flatten()(inputs)
    # x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(128 * 9 * 9, "elu")(x)

    # x = tf.keras.layers.Conv1D(128 * 7 * 7, 3, 1, "same", activation="elu")(inputs)
    # x = tf.keras.layers.GlobalAveragePooling1D()(x)

    x = tf.keras.layers.Reshape((9,9,128))(x)

    x = tf.keras.layers.Conv2D(256, 3, 1, "same")(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.2)(x)
    x = tf.keras.layers.ELU()(x)
    x = tf.keras.layers.UpSampling2D()(x)

    x = tf.keras.layers.Conv2D(128, 3, 1, "same")(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.2)(x)
    x = tf.keras.layers.ELU()(x)
    x = tf.keras.layers.UpSampling2D()(x)

    x = tf.keras.layers.Conv2D(64, 3, 1, "same")(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.2)(x)
    x = tf.keras.layers.ELU()(x)
    x = tf.keras.layers.UpSampling2D()(x)

    # x = tf.keras.layers.Conv2D(256, 3, 1, "same")(x)
    # x = tf.keras.layers.BatchNormalization(momentum=0.2)(x)
    # x = tf.keras.layers.ELU()(x)
    # x = tf.keras.layers.UpSampling2D()(x)
    #
    # x = tf.keras.layers.Conv2D(128, 3, 1, "same")(x)
    # x = tf.keras.layers.BatchNormalization(momentum=0.2)(x)
    # x = tf.keras.layers.ELU()(x)
    # x = tf.keras.layers.UpSampling2D()(x)
    #
    # x = tf.keras.layers.Conv2D(64, 3, 1, "same")(x)
    # x = tf.keras.layers.BatchNormalization(momentum=0.2)(x)
    # x = tf.keras.layers.ELU()(x)

    x = tf.keras.layers.Conv2D(3, 3, 1, "same", activation="tanh")(x)


    # inputs = tf.keras.layers.Input((224,224,3))
    #
    # x = tf.keras.layers.Conv2D(64,4,1,"same", kernel_initializer="he_normal")(inputs)
    # x = tf.keras.layers.ELU()(x)
    # p = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
    # # p = tf.keras.layers.LeakyReLU(0.2)(x)
    # x = tf.keras.layers.MaxPool2D()(p)
    #
    # x1 = dense_b(x,128,4,1,2)
    # x = tf.keras.layers.MaxPool2D()(x1)
    # # x2 = dense_b(x,256,3,1,2)
    # # x = tf.keras.layers.MaxPool2D()(x2)
    #
    # x = dense_b(x, 256, 4, 1, 2)
    # #
    # # x = dense_b(x, 256, 3, 1, 2)
    # # x = tf.keras.layers.UpSampling2D()(x)
    # # x = tf.keras.layers.Add()([x,x2])
    #
    # x = dense_b(x, 128, 4, 1, 2)
    # x = tf.keras.layers.UpSampling2D()(x)
    # x = tf.keras.layers.Add()([x,x1])
    #
    # x = dense_b(x, 64, 4, 1, 2)
    # x = tf.keras.layers.UpSampling2D()(x)
    # x = tf.keras.layers.Add()([x,p])
    #
    # x = dense_b(x, 16, 4, 1, 2)
    #
    # # x = tf.keras.layers.C(128, return_sequences=True)(x)
    #
    # x = tf.keras.layers.Conv2D(3,4,1,"same",activation="tanh")(x)

    return tf.keras.Model(inputs, x)

def discriminator():
    inputs = tf.keras.layers.Input((72, 72, 3))

    x = tf.keras.layers.Conv2D(64,3,1,"same", activation="elu", kernel_initializer="he_normal")(inputs)
    x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(128,3,1,"same", activation="elu", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
    x = tf.keras.layers.MaxPool2D()(x)
    # x = tf.keras.layers.Conv2D(256,3,1,"same", activation="elu", kernel_initializer="he_normal")(x)
    # x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
    # x = tf.keras.layers.MaxPool2D()(x)
    # x = tf.keras.layers.Conv2D(128,3,1,"same", activation="elu", kernel_initializer="he_normal")(x)
    # x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
    # x = tf.keras.layers.MaxPool2D()(x)

    x1 = tf.keras.layers.MaxPool2D(3)(x)
    x1 = tf.keras.layers.UpSampling2D(3)(x1)
    x2 = tf.keras.layers.MaxPool2D(6)(x)
    x2 = tf.keras.layers.UpSampling2D(6)(x2)
    x3 = tf.keras.layers.MaxPool2D(9)(x)
    x3 = tf.keras.layers.UpSampling2D(9)(x3)
    x4 = tf.keras.layers.MaxPool2D(18)(x)
    x4 = tf.keras.layers.UpSampling2D(18)(x4)

    x = tf.keras.layers.Concatenate()([x1,x2,x3,x])

    x = tf.keras.layers.Conv2D(1,1,1,"same",activation="sigmoid")(x)

    return tf.keras.Model(inputs,x)



class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.g = generator()
        self.d = discriminator()
        self.v = tf.keras.applications.VGG19(include_top=False)
        self.g_opt = tf.keras.optimizers.Adam(1e-4)
        self.d_opt = tf.keras.optimizers.Adam(1e-4)


    def optimize(self, inputs_g, inputs_d, batch_size=128):
        with tf.GradientTape(persistent=True) as tape:
            fake_image = self.g(inputs_g)
            x = self.d(inputs_d)
            loss_d_1 = tf.reduce_mean((np.ones((batch_size,18,18,1)) - x) ** 2)
            x = self.d(fake_image)
            loss_d_2 = tf.reduce_mean((np.zeros((batch_size, 18, 18, 1)) - x) ** 2)

            loss_d = loss_d_1 + loss_d_2

        self.d_gradients = gradients = tape.gradient(loss_d, self.d.trainable_variables)
        self.d_opt.apply_gradients(zip(gradients, self.d.trainable_variables))

        with tf.GradientTape() as tape:
            fake_image = self.g(inputs_g)
            fake_y = np.ones((batch_size, 14, 14, 1))
            fake_x = self.d(fake_image)
            loss_e = tf.reduce_mean((inputs_d - fake_image) ** 2)
            loss_a = tf.reduce_mean((self.v(inputs_d) - self.v(fake_image)) ** 2) * 1e-1
            loss_p = -tf.reduce_mean(tf.math.log(fake_x))
            loss_g = loss_p + loss_a + loss_e
            # loss_g = loss_p
        # print(loss_d)
        self.g_gradients = gradients = tape.gradient(loss_g, self.g.trainable_variables)
        self.g_opt.apply_gradients(zip(gradients, self.g.trainable_variables))
