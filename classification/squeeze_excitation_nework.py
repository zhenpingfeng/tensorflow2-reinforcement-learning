import tensorflow as tf

def se_bloack(x, r=16):
    sq = tf.keras.layers.GlobalAveragePooling2D()(x)
    sq = tf.keras.layers.Reshape((1,1,x.shape[-1]))(sq)

    ex1 = tf.keras.layers.Dense(x.shape[-1] // r, "relu")(sq)
    ex2 = tf.keras.layers.Dense(x.shape[-1], "sigmoid")(ex1)

    return tf.keras.layers.Multiply()([x, ex2])

def stem(i):
    x = tf.keras.layers.Conv2D(32, 3, 2, activation="elu")(i)
    x = tf.keras.layers.Conv2D(32, 3, 1, activation="elu")(x)
    x = tf.keras.layers.Conv2D(64, 3, 1, "same", activation="elu")(x)

    x1 = tf.keras.layers.Conv2D(96, 3, 2, activation="elu")(x)
    x2 = tf.keras.layers.MaxPool2D(3, 2)(x)
    x = tf.keras.layers.Concatenate()([x1, x2])
    # x = se_block(x)

    x1 = tf.keras.layers.Conv2D(64, 1, 1, "same", activation="elu")(x)
    x1 = tf.keras.layers.Conv2D(96, 3, 1, activation="elu")(x1)

    x2 = tf.keras.layers.Conv2D(64, 1, 1, "same", activation="elu")(x)
    x2 = tf.keras.layers.Conv2D(64, (7,1), 1, "same", activation="elu")(x2)
    x2 = tf.keras.layers.Conv2D(64, (1,7), 1, "same", activation="elu")(x2)
    x2 = tf.keras.layers.Conv2D(96, 3, 1, activation="elu")(x2)
    return tf.keras.layers.Concatenate()([x1, x2])

def resnet_a(i):
    x1 = tf.keras.layers.Conv2D(32,3,1,"same",activation="elu")(i)
    x1 = tf.keras.layers.Conv2D(48,3,1,"same",activation="elu")(x1)
    x1 = tf.keras.layers.Conv2D(62,3,1,"same",activation="elu")(x1)

    x2 = tf.keras.layers.Conv2D(32,1,1,"same",activation="elu")(i)
    x2 = tf.keras.layers.Conv2D(32,3,1,"same",activation="elu")(x2)

    x3 = tf.keras.layers.Conv2D(32,1,1,"same",activation="elu")(i)

    x = tf.keras.layers.Concatenate()([x1,x2,x3])
    x = tf.keras.layers.Conv2D(192,1,1,"same")(x)
    x = tf.keras.layers.Add()([i,x])

    return tf.keras.layers.ELU()(x)
def resnet_b(i):
    x1 = tf.keras.layers.Conv2D(128,1,1,"same",activation="elu")(i)
    x1 = tf.keras.layers.Conv2D(160,(1,7),1,"same",activation="elu")(x1)
    x1 = tf.keras.layers.Conv2D(192,(7,1),1,"same",activation="elu")(x1)

    x2 = tf.keras.layers.Conv2D(192,1,1,"same",activation="elu")(i)

    x = tf.keras.layers.Concatenate()([x1,x2])
    x = tf.keras.layers.Conv2D(960,1,1,"same")(x)
    x = tf.keras.layers.Add()([i,x])

    return tf.keras.layers.ELU()(x)

def resnet_c(i):
    x1 = tf.keras.layers.Conv2D(192,1,1,"same",activation="elu")(i)
    x1 = tf.keras.layers.Conv2D(224,(1,7),1,"same",activation="elu")(x1)
    x1 = tf.keras.layers.Conv2D(256,(7,1),1,"same",activation="elu")(x1)

    x2 = tf.keras.layers.Conv2D(192,1,1,"same",activation="elu")(i)

    x = tf.keras.layers.Concatenate()([x1,x2])
    x = tf.keras.layers.Conv2D(1952,1,1,"same")(x)
    x = tf.keras.layers.Add()([i,x])

    return tf.keras.layers.ELU()(x)

def reduction_a(i):
    x1 = tf.keras.layers.Conv2D(256,1,1,"same",activation="elu")(i)
    x1 = tf.keras.layers.Conv2D(256,3,1,"same",activation="elu")(x1)
    x1 = tf.keras.layers.Conv2D(384,3,2,"valid",activation="elu")(x1)

    x2 = tf.keras.layers.Conv2D(384,3,2,activation="elu")(i)

    x3 = tf.keras.layers.MaxPool2D(3,2)(i)

    return tf.keras.layers.Concatenate()([x1,x2,x3])

def reduction_b(i):
    x1 = tf.keras.layers.Conv2D(256,1,1,"same",activation="elu")(i)
    x1 = tf.keras.layers.Conv2D(288,3,1,"same",activation="elu")(x1)
    x1 = tf.keras.layers.Conv2D(320,3,2,"valid",activation="elu")(x1)

    x2 = tf.keras.layers.Conv2D(256,1,1,"same",activation="elu")(i)
    x2 = tf.keras.layers.Conv2D(288,3,2,"valid",activation="elu")(x2)

    x3 = tf.keras.layers.Conv2D(256,1,1,"same",activation="elu")(i)
    x3 = tf.keras.layers.Conv2D(384,3,2,"valid",activation="elu")(x3)

    x4 = tf.keras.layers.MaxPool2D(3,2)(i)

    return tf.keras.layers.Concatenate()([x1,x2,x3,x4])


def f(x, f):
    x = f(x)
    return x

def se_inception_resnet_v2(input_shape=(224,224,3), output_size=2):
    inputs = tf.keras.layers.Input(input_shape)

    x = f(inputs, stem)
    x = se_bloack(x)

    for i in range(5):
        x = f(x, resnet_a)
    x = se_bloack(x)
    x = f(x, reduction_a)
    x = se_bloack(x)

    for i in range(10):
        x = f(x, resnet_b)
    x = se_bloack(x)
    x = f(x, reduction_b)
    x = se_bloack(x)

    for i in range(5):
        x = f(x, resnet_c)
    x = se_bloack(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    if output_size == 2:
        x = tf.keras.layers.Dense(1, "sigmoid")(x)
        loss = "binary_crossentropy"
    else:
        x = tf.keras.layers.Dense(output_size, "softmax")(x)
        loss = "categorical_crossentropy"

    model = tf.keras.Model(inputs, x)
    opt = tf.keras.optimizers.RMSprop(0.001,epsilon=1.,decay=0.5)
    model.compile(opt, loss, ["accuracy"])
    return model
