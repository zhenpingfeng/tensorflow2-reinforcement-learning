import tensorflow as tf
from mpl_finance import candlestick2_ohlc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

df = pd.read_csv("gbpjpy15.csv")
x = np.array(df[["Close", "Open", "High", "Low"]])

# g = tf.keras.preprocessing.sequence.TimeseriesGenerator(x,x,20)
# x, y = [], []
# for i1,i2 in g:
#     x += i1.tolist()
#     y += i2.tolist()
x = np.array(x)[-15000:]
y = x[:]

fig1 = plt.figure(figsize=(4.03, 4.13))
fig2 = plt.figure(figsize=(1, 1))
ax1 = fig1.add_subplot(1, 1, 1)
ax2 = fig2.add_subplot(1, 1, 1)

image1 = []
image_time_x = []
image2 = []
image_label = []

window_size = 30
history_size = 2

for i in range(0, len(y) - window_size - history_size):
    # for i in range(1):
    candlestick2_ohlc(ax1, x[i:i + window_size, 1], x[i:i + window_size, 2], x[i:i + window_size, 3],
                      x[i:i + window_size, 0], width=0.5, alpha=1, colorup="r", colordown="b")
    image_time_x.append(x[i:i + window_size])
    i += window_size

    candlestick2_ohlc(ax2, x[i:i + history_size, 1], x[i:i + history_size, 2], x[i:i + history_size, 3],
                      x[i:i + history_size, 0], width=0.1, alpha=1, colorup="k", colordown="y")

    label = 1 if x[i, 0] > x[i - 1, 0] else 0
    image_label.append(label)

    ax1.axis("off")
    ax2.axis("off")

    # if x[i-1,-1,0] > x[i,-1,0]:
    #   color = "royalblue"
    # else:
    #   color = "salmon"

    fig1.savefig('forex1.jpg', bbox_inches='tight', pad_inches=0.0)
    fig2.savefig('forex2.jpg', pad_inches=0.0, facecolor="r", alpha=0.5)
    #
    ax1.cla()
    ax2.cla()

    im1 = cv2.imread("forex1.jpg")
    image1.append(im1)
    im2 = cv2.imread("forex2.jpg")
    image2.append(im2)

np.save("image_x", image1)
np.save("image_time_x", image_time_x)
np.save("image_y", image2)
np.save("image_label", image_label)