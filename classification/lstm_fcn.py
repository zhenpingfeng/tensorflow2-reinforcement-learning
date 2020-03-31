import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def model(shape=(130,4)):
    i = tf.keras.layers.Input(shape)
    x1 = tf.keras.layers.Conv1D(128, 3, 1, "same")(i)
    x1 = tf.keras.layers.BatchNormalization(epsilon=0.001, momentum=0.99)(x1)
    x1 = tf.keras.layers.ELU()(x1)
    x1 = tf.keras.layers.Conv1D(256, 3, 1, "same")(x1)
    x1 = tf.keras.layers.BatchNormalization(epsilon=0.001, momentum=0.99)(x1)
    x1 = tf.keras.layers.ELU()(x1)
    x1 = tf.keras.layers.Conv1D(128, 3, 1, "same")(x1)
    x1 = tf.keras.layers.BatchNormalization(epsilon=0.001, momentum=0.99)(x1)
    x1 = tf.keras.layers.ELU()(x1)
    x1 = tf.keras.layers.GlobalAveragePooling1D()(x1)

    x2 = tf.keras.layers.Permute((2,1))(i)
    x2 = tf.keras.layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2)(x2)

    x = tf.keras.layers.Concatenate()([x1,x2])
    x = tf.keras.layers.Dense(1, "sigmoid")(x)

    model = tf.keras.Model(i,x)
    opt = tf.keras.optimizers.Adam(decay=1e-3)
    model.compile(opt, "binary_crossentropy", ["accuracy"])

    return model

def data():
    df = pd.read_csv("gbpjpy15.csv")
    x = np.array(df[["Close", "Open", "High", "Low"]])
    y = np.array(df[["Close"]])

    X = []
    Y = []

    window_size = 130
    for i in range(len(y)-window_size):
        X.append(x[i:i+window_size])
        Y.append(0 if y[i+window_size] >= y[i+window_size - 1] else 1)

    X, Y = np.array(X), np.array(Y)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    return X, Y, x_train, x_test, y_train, y_test
