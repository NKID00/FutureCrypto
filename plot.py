from typing import Tuple
from os import walk

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from config import TEST_DATASET_SIZE, PREDICTION_SIZE, SHOW_DIFFERENCE


def load_plot_data() -> Tuple[
    tf.data.Dataset, tf.data.Dataset, Tuple[np.float64, np.float64], np.ndarray
]:
    print('load data ...')
    data = np.load('./preprocessed/BTCUSDT.npz')
    xp = data['xp']
    raw = data['raw']
    return xp, raw


def load_model() -> Tuple[tf.keras.Model, str]:
    print('load model ...')
    name = sorted(next(walk("model"))[1])[-1]
    print(f'model name = {name}')
    model = tf.keras.models.load_model(f'./model/{name}')
    model.summary()
    return model, name


def plot_one(ax, real, t, c, l):
    ax.plot(range(len(real) - len(t), len(real)), t, c=c, label=l)
    # ax.scatter(range(len(real) - len(t), len(real)), t, c=c)


def plot_one_prediction(ax, x, y, t):
    real = np.concatenate((x, y))
    ax.plot(range(len(real)), real, c='gray', label='real', zorder=1)
    # ax.scatter(range(len(real)), real, c='gray', zorder=1)
    # plot_one(ax, real, t, 'deepskyblue', 'output')
    # if SHOW_DIFFERENCE:
    #     plot_one(ax.twinx(), real, y[-len(t):] - t, 'salmon', 'difference')


def plot_prediction(
    model: tf.keras.Model, xp: Tuple[np.float64, np.float64], raw: np.ndarray
):
    test_data = raw[-TEST_DATASET_SIZE-1:]
    x_iter = test_data[:-1]
    y_iter = test_data[1:]
    axes = plt.subplots(TEST_DATASET_SIZE // 2, 2)[1].flatten()
    t_iter = x_iter[:]
    for i in range(1, PREDICTION_SIZE+1):
        print('%-30s' % f'predict {i}/{PREDICTION_SIZE} ...', end='\r')
        prediction = model.predict(
            t_iter.reshape(t_iter.shape[0], t_iter.shape[1], 1)
        ).reshape(-1, 1)
        t_iter = np.hstack((t_iter[:, 1:], prediction))
    print()
    for i, (x, y, t, ax) in enumerate(zip(x_iter, y_iter, t_iter, axes), 1):
        print(f'plot {i}/{len(x_iter)} ...', end='\r')
        plot_one_prediction(
            ax,
            np.interp(x[-PREDICTION_SIZE*3:], [-1.0, +1.0], xp),
            np.interp(y[-PREDICTION_SIZE*3:], [-1.0, +1.0], xp),
            np.interp(t[-PREDICTION_SIZE:], [-1.0, +1.0], xp)
        )
    print()
    plt.show()


if __name__ == '__main__':
    model = load_model()[0]
    xp, raw = load_plot_data()
    plot_prediction(model, xp, raw)
