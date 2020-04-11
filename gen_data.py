import sys

import numpy as np
import pandas as pd
import ta


def gen_data(file_path="gbpjpy15.csv"):
    try:
        print("load file")
        df = pd.read_csv(file_path)
    except:
        print("Use 'python gen_data.py 'file_path''")
        return

    df["Close1"] = df["Close"] * 100

    ma = np.array(ta.trend.ema(df["Close1"], 14) - ta.trend.ema(df["Close1"], 7)).reshape((-1, 1))
    macd = np.array(ta.trend.macd_diff(df["Close1"])).reshape((-1, 1))
    rsi = np.array(ta.momentum.rsi(df["Close"]) - ta.momentum.rsi(df["Close"], 7)).reshape((-1, 1))
    stoch = np.array(
        ta.momentum.stoch_signal(df["High"], df["Low"], df["Close"]) - ta.momentum.stoch(df["High"], df["Low"],
                                                                                         df["Close"])).reshape((-1, 1))

    x = np.concatenate([ma, macd, rsi, stoch], -1)

    y = np.array(df[["Open"]])
    atr = np.array(ta.volatility.average_true_range(df["High"], df["Low"], df["Close"]))
    high = np.array(df[["High"]])
    low = np.array(df[["Low"]])

    print("gen time series data")
    x = x[100:]
    y = y[100:]

    window_size = 130
    time_x = []
    time_y = []

    for i in range(len(y) - window_size):
        time_x.append(x[i:i + window_size])
        i += window_size
        time_y.append(y[i])

    x = np.array(time_x).reshape((-1, window_size, x.shape[-1]))
    y = np.array(time_y).reshape((-1, y.shape[-1]))

    atr = atr[-len(y):].reshape((-1, 1))
    scale_atr = atr
    high = high[-len(y):].reshape((-1, 1))
    low = low[-len(y):].reshape((-1, 1))

    x = (x * 10 ** 5).astype(np.int32) * (10 ** -5)

    np.save("x", x)
    np.save("target", np.array([y, atr, scale_atr, high, low]))

    print("done\n")


if __name__ == "__main__":
    argv = sys.argv
    print(argv[1])
    gen_data(argv[1])
